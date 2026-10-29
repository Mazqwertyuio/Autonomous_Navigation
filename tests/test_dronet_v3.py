"""Check official inputs, weights and the collision veto through the pipeline."""
import hashlib
import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms

from nomad_pipeline.config import Config
from nomad_pipeline.models import dronet_assets
from nomad_pipeline.models.base import PolicyOutput
from nomad_pipeline.models import pulp_dronet
from nomad_pipeline.models.pulp_dronet import collision_info, preprocess_bgr
from nomad_pipeline.pipeline import NavPipeline, build_policy


def frame():
    return np.random.default_rng(12).integers(0, 220, (301, 403, 3), dtype=np.uint8)


def test_preprocessing_matches_official_center_crop_not_resize():
    bgr = frame()
    pil = Image.fromarray(bgr[:, :, ::-1]).convert("L")
    expected = transforms.Compose([transforms.CenterCrop(200), transforms.ToTensor()])(pil)
    actual = preprocess_bgr(bgr)
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
    assert actual.shape == (1, 1, 200, 200)
    assert 0 <= actual.min() <= actual.max() <= 1


@pytest.mark.parametrize("bad", [np.zeros((199, 400, 3), np.uint8),
                                np.zeros((300, 400, 3), np.float32),
                                np.zeros((3, 300, 400), np.uint8)])
def test_preprocessing_rejects_small_or_normalized_inputs(bad):
    with pytest.raises(ValueError):
        preprocess_bgr(bad)


def test_missing_and_corrupt_checkpoints_fail_before_model_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(pulp_dronet, "dronet", lambda **kw: pytest.fail("Model was created"))
    path = tmp_path / "missing.pth"
    with pytest.raises(FileNotFoundError, match="download_dronet_v3.py"):
        pulp_dronet.DroNetV3Inference("cpu", weights_path=path)
    path.write_bytes(b"not the official model")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        pulp_dronet.DroNetV3Inference("cpu", weights_path=path)


def test_incompatible_state_dict_is_not_partially_loaded(tmp_path, monkeypatch):
    path = tmp_path / "incompatible.pth"
    torch.save({"unexpected": torch.ones(1)}, path)
    # Use an intentionally accepted test digest to reach strict architecture validation.
    monkeypatch.setitem(dronet_assets.CHECKPOINTS, "v3", {
        "filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
    with pytest.raises(RuntimeError, match="state_dict"):
        pulp_dronet.DroNetV3Inference("cpu", weights_path=path)


@pytest.mark.parametrize("values", [(float("nan"), .2), (.1, float("inf")), (.1, -0.1)])
def test_nonfinite_predictions_fail_closed(values):
    with pytest.raises(RuntimeError):
        collision_info(*values)


def test_risk_slows_then_stops_without_claiming_a_distance():
    risks = [collision_info(.2, risk) for risk in [0, .2, .6, .79, .8, 1]]
    scales = [info["speed_scale"] for info in risks]
    assert scales == sorted(scales, reverse=True)
    assert scales[-2:] == [0, 0]
    assert risks[2]["state"] == "CAUTION_SLOW"
    assert risks[-1]["state"] == "EMERGENCY_BRAKE"
    assert risks[-1]["min_dist_m"] is None


@pytest.fixture
def fake_detector(monkeypatch):
    class Detector:
        def __init__(self, *args, **kwargs):
            self.device = torch.device("cpu")
            self.params_m = 0
            self.steer, self.risk = .5, 0.0

        def evaluate(self, bgr):
            # The policy must receive the original camera pixels.
            assert bgr.shape == (301, 403, 3)
            return collision_info(self.steer, self.risk)

        def reset(self):
            pass

    monkeypatch.setattr(pulp_dronet, "DroNetV3Inference", Detector)


def test_reactive_commands_keep_sign_and_brake_after_filter(fake_detector):
    cfg = Config(model="dronet-v3", device="cpu", v_max=.35, w_max=.45, ema_alpha=.1)
    pipe = NavPipeline(cfg)
    pipe.obstacle_shield.evaluate = lambda *args, **kwargs: {"speed_scale": 1.0}
    _, wp, clear, _ = pipe.step(frame())
    assert pipe.frontier_mgr is None
    assert clear[0] == pytest.approx(.35)
    assert clear[1] > 0 and wp[-1, 1] > 0  # positive model yaw is leftward
    pipe.policy.detector.risk = .4
    _, _, slowed, _ = pipe.step(frame())
    assert slowed[0] == pytest.approx(.21)
    pipe.policy.detector.risk = .95
    img, _, stopped, _ = pipe.step(frame())
    assert stopped[0] == 0.0  # no residual forward speed from EMA
    assert img.shape == frame().shape
    assert pipe.last_shield_info["state"] == "EMERGENCY_BRAKE"
    pipe.policy.detector.risk = 0.0
    assert pipe.step(frame())[2][0] > 0


def test_reactive_policy_rejects_image_goal(fake_detector):
    policy = build_policy("dronet-v3", device="cpu")
    with pytest.raises(ValueError, match="image goal"):
        policy.predict_bgr(frame(), goal=np.zeros((3, 96, 96)))


class FixedGoalPolicy:
    context = 0
    img_size = (85, 64)
    device = torch.device("cpu")

    def predict(self, frames, goal=None):
        return PolicyOutput(np.array([[.3, 0], [.6, 0], [1., 0]]), dist=5.0)


def test_all_forbidden_frontiers_veto_forward_motion():
    pipe = NavPipeline(Config(model="vint"), policy=FixedGoalPolicy())
    # Every crop overlaps this zone, but the projected waypoints don't: the
    # frontier veto must work independently of the trajectory point test.
    pipe.forbidden_mgr.add_zone(100, 70, 300, 90)
    _, _, vw, _ = pipe.step(frame())
    assert pipe.frontier_mgr.is_trapped
    assert pipe.last_hazard_info["hazard"] == "TRAPPED"
    assert vw[0] == 0.0


def test_independent_shield_stops_existing_policy():
    class Shield:
        risk = 0.0

        def evaluate(self, bgr, **kwargs):
            return collision_info(-.5, self.risk)

    shield = Shield()
    pipe = NavPipeline(Config(model="nomad"), policy=FixedGoalPolicy(), shield=shield)
    pipe.obstacle_shield.evaluate = lambda *args, **kwargs: {"speed_scale": 1.0}
    assert pipe.step(frame())[2][0] > 0
    shield.risk = .9
    _, _, vw, _ = pipe.step(frame())
    assert vw[0] == 0
    assert vw[1] < 0


@pytest.mark.parametrize("name,variant", [("dronet-v3", "v3"), ("dronet-v3-tiny", "tiny-v3")])
def test_official_checkpoint_inference_when_installed(name, variant):
    path = dronet_assets.WEIGHTS_DIR / dronet_assets.CHECKPOINTS[variant]["filename"]
    if not path.is_file():
        pytest.skip("Run download_dronet_v3.py to enable official-checkpoint integration checks")
    policy = build_policy(name, device="cpu")
    output = policy.predict_bgr(frame())
    assert output.waypoints.shape == (8, 2)
    assert np.isfinite(output.waypoints).all()
    assert 0 <= output.shield_info["coll_prob"] <= 1
    assert policy.detector.weights_path == path.resolve()
