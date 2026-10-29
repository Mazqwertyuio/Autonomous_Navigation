import numpy as np, pytest, torch
from nomad_pipeline.config import Config
from nomad_pipeline.pipeline import NavPipeline, build_policy


@pytest.mark.parametrize("name", ["gnm", "vint", "nomad"])
def test_e2e(name):
    pipe = NavPipeline(Config(model=name), build_policy(name))
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    if pipe.policy.needs_goal:
        pipe.set_goal(frame)
    for _ in range(3):
        img, wp, vw, out = pipe.step(frame)
    assert img.shape == frame.shape
    assert wp.shape == (pipe.policy.cfg["len_traj_pred"], 2)
    assert np.isfinite(wp).all()
    if name == "nomad":
        assert out.waypoints.shape[0] == pipe.policy.num_samples and out.dist is None
        pipe.set_goal(frame)
        _, _, _, out2 = pipe.step(frame)
        assert out2.dist is not None
