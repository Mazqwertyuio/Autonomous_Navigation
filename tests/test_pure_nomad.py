"""Isolation checks for the goal-free NoMaD baseline, including a fake live run."""
import csv
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nomad_pipeline.config import Config
from nomad_pipeline.controller import linear_tello_rc, pure_pursuit
from nomad_pipeline.models.base import PolicyOutput
from nomad_pipeline import pipeline as pipeline_module
from nomad_pipeline.pipeline import NavPipeline
import tello_nomad_autopilot as app


class FixedPolicy:
    context = 1
    img_size = (96, 96)
    device = torch.device("cpu")

    def __init__(self):
        self.goals = []

    def predict(self, frames, goal=None):
        self.goals.append(goal)
        points = np.array([[[.3, .1], [.6, .2], [1., .3], [1.5, .4]]])
        return PolicyOutput(points)


def no_addons(*args, **kwargs):
    pytest.fail("An autonomous add-on ran in pure NoMaD mode")


def test_pure_nomad_never_initializes_or_calls_addons(monkeypatch):
    monkeypatch.setattr(pipeline_module, "MonocularObstacleShield", no_addons)
    monkeypatch.setattr(pipeline_module, "ForbiddenZoneManager", no_addons)
    monkeypatch.setattr(pipeline_module, "select_best_trajectory", no_addons)
    cfg = Config(pure_nomad=True, device="cpu", num_diffusion_iters=4)
    policy = FixedPolicy()
    pipe = NavPipeline(cfg, policy=policy)
    pipe.cmd_filter.update = no_addons
    image = np.full((300, 400, 3), 150, np.uint8)  # would trip the wall heuristic
    for _ in range(2):
        overlay, wp, command, output = pipe.step(image)
        assert command == pure_pursuit(wp, cfg.lookahead, cfg.v_max, cfg.w_max, max_points=0)
        assert command[0] > 0
        np.testing.assert_array_equal(output.waypoints[0], wp)
        assert overlay.shape == image.shape
    assert all(goal is None for goal in policy.goals)
    assert pipe.shield is pipe.obstacle_shield is pipe.frontier_mgr is pipe.novelty_engine is None
    assert pipe.forbidden_mgr is None
    assert pipe.last_shield_info is None
    assert pipe.last_hazard_info["hazard"] == "DISABLED"
    assert np.all(image == 150)  # visualization does not contaminate the camera frame


@pytest.mark.parametrize("kw", [dict(model="vint"), dict(use_dronet_shield=True),
                                dict(use_shield=True), dict(use_vlfly=True),
                                dict(use_novelty_explorer=True), dict(num_samples=2)])
def test_pure_config_rejects_mixed_modes(kw):
    with pytest.raises(ValueError):
        Config(pure_nomad=True, **kw)


def test_pure_mode_rejects_goals_and_injected_shields():
    cfg = Config(pure_nomad=True)
    with pytest.raises(ValueError, match="injected shield"):
        NavPipeline(cfg, policy=FixedPolicy(), shield=object())
    pipe = NavPipeline(cfg, policy=FixedPolicy())
    with pytest.raises(ValueError, match="goal mask=1"):
        pipe.set_goal(np.zeros((300, 400, 3), np.uint8))


def test_real_nomad_checkpoint_in_pure_goal_free_mode():
    torch.manual_seed(0)
    pipe = NavPipeline(Config(pure_nomad=True, device="cpu", num_diffusion_iters=4))
    image = np.full((300, 400, 3), 150, np.uint8)
    _, wp, command, output = pipe.step(image)
    assert pipe.goal is None and output.dist is None
    assert output.waypoints.shape == (1, pipe.policy.cfg["len_traj_pred"], 2)
    assert pipe.policy.iters == 4 and pipe.policy.num_samples == 1
    assert np.isfinite(wp).all()
    assert command == pure_pursuit(wp, pipe.cfg.lookahead, pipe.cfg.v_max, pipe.cfg.w_max, max_points=0)


def test_linear_rc_preserves_signs_zero_and_small_commands():
    assert linear_tello_rc(0, 0, .35, .45) == (0, 0, 0, 0)
    assert linear_tello_rc(.35, .45, .35, .45) == (0, 22, 0, -28)
    assert linear_tello_rc(.35, -.45, .35, .45) == (0, 22, 0, 28)
    small = linear_tello_rc(.035, .045, .35, .45)
    assert small == (0, 2, 0, -3)  # no added minimum forward or yaw boost
    assert linear_tello_rc(.35, .45, .35, .45)[1] == 22  # no turn slowdown


@pytest.mark.parametrize("args", [[], ["--pure-nomad"], ["--model", "nomad"]])
def test_cli_defaults_to_pure_nomad(args, monkeypatch):
    captured = []
    def offline(cfg, *args, **kwargs):
        captured.append(cfg)
        return {}
    monkeypatch.setattr(app, "run_video_simulation", offline)
    app.main(args)
    cfg = captured[0]
    assert cfg.pure_nomad and cfg.model == "nomad"
    assert cfg.num_diffusion_iters == 4 and cfg.num_samples == 1
    assert not cfg.use_obstacle_shield
    assert cfg.ema_alpha == 1 and cfg.max_pursuit_points == 0


@pytest.mark.parametrize("args", [["--pure-nomad", "--dronet"],
                                ["--pure-nomad", "--model", "vint"],
                                ["--pure-nomad", "--num-samples", "2"]])
def test_cli_rejects_conflicting_pure_options(args):
    with pytest.raises(SystemExit) as error:
        app.main(args)
    assert error.value.code == 2


def test_pure_live_commands_logging_and_manual_stop(tmp_path, monkeypatch):
    """Exercise the live orchestration without networking or physical motors."""
    monkeypatch.chdir(tmp_path)
    policy = FixedPolicy()
    cfg = Config(pure_nomad=True, num_diffusion_iters=4, v_max=.35, w_max=.45)
    pipe = NavPipeline(cfg, policy=policy)
    monkeypatch.setattr(app, "NavPipeline", lambda _: pipe)
    monkeypatch.setattr(app, "StuckRecoveryController", no_addons)
    monkeypatch.setattr(app, "compute_altitude_ud", no_addons)
    monkeypatch.setattr(app, "map_vw_to_tello_rc", no_addons)
    for method in ("namedWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(app.cv2, method, lambda *a, **kw: None)
    monkeypatch.setattr(app.cv2, "setMouseCallback", no_addons)
    keys = iter(map(ord, ["t", "n", "w", "g", " ", "n", "q"]))
    monkeypatch.setattr(app.cv2, "waitKey", lambda _: next(keys))

    writers = {}
    class Writer:
        def __init__(self, path, *args):
            self.frames = []
            writers[path] = self
        def isOpened(self):
            return True
        def write(self, image):
            self.frames.append(image.copy())
        def release(self):
            pass
    monkeypatch.setattr(app.cv2, "VideoWriter", Writer)

    class Drone:
        def __init__(self):
            self.commands = []
            self.landed = 0
        def connect(self):
            pass
        def streamon(self):
            pass
        def streamoff(self):
            pass
        def get_battery(self):
            return 80
        def get_frame_read(self):
            return SimpleNamespace(frame=np.full((300, 400, 3), 70, np.uint8))
        def get_current_state(self):
            return dict(tof=10, pitch=60, roll=60)  # baseline logs these, no altitude/recovery overrides
        def send_rc_control(self, *command):
            self.commands.append(command)
        def takeoff(self):
            pass
        def land(self):
            self.landed += 1
    drone = Drone()
    monkeypatch.setitem(sys.modules, "djitellopy", SimpleNamespace(Tello=lambda: drone))
    app.run_live_tello(cfg)
    assert drone.landed == 1
    assert drone.commands[-1] == (0, 0, 0, 0)
    assert any(command[1] > 0 for command in drone.commands)
    assert all(command[0] == command[2] == 0 and command[1] >= 0 for command in drone.commands)
    assert all(goal is None for goal in policy.goals)
    logs = list((tmp_path / "flights").glob("telemetry*.csv"))
    with logs[0].open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 7
    assert all(row["control_profile"] == "pure-nomad" and row["hazard"] == "DISABLED" for row in rows)
    assert all(row["recovery"] == "NORMAL" and row["rc_ud"] == "0" for row in rows)
    metadata = json.loads(next((tmp_path / "flights").glob("config*.json")).read_text(encoding="utf-8"))
    assert metadata["profile"] == "pure-nomad" and metadata["config"]["pure_nomad"]
    raw = next(writer for path, writer in writers.items() if "flight_raw_" in path)
    assert len(raw.frames) == 7 and all(np.all(image == 70) for image in raw.frames)
