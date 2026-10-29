import numpy as np
from nomad_pipeline.config import Config
from nomad_pipeline.projection import project_points
from nomad_pipeline.controller import pure_pursuit
from nomad_pipeline.frame_buffer import FrameBuffer

K = Config().intrinsics(640, 480)


def test_projection_forward():
    uv, ok = project_points([[1.0, 0.0]], K, 0.4)
    assert ok[0]
    assert np.isclose(uv[0, 0], K[0, 2])
    assert np.isclose(uv[0, 1], K[1, 2] + K[1, 1] * 0.4)


def test_projection_horizon_and_side():
    uv, _ = project_points([[1, 0], [5, 0]], K, 0.4)
    assert uv[1, 1] < uv[0, 1] and uv[1, 1] > K[1, 2]
    uv, _ = project_points([[2, 0.5]], K, 0.4)
    assert uv[0, 0] < K[0, 2]


def test_projection_behind_camera():
    _, ok = project_points([[0.0, 0.0], [-1.0, 0.0]], K, 0.4)
    assert not ok.any()


def test_pursuit():
    v, w = pure_pursuit([[0.3, 0], [0.6, 0], [1.0, 0]])
    assert v > 0 and abs(w) < 1e-9
    assert pure_pursuit([[1, 0.5], [2, 1.0]])[1] > 0
    assert pure_pursuit([[1, -0.5], [2, -1.0]])[1] < 0
    assert pure_pursuit([]) == (0.0, 0.0)


def test_buffer():
    b = FrameBuffer(4, (96, 96))
    f = np.zeros((480, 640, 3), np.uint8)
    b.push(f)
    assert b.get().shape == (4, 3, 96, 96)
    for i in range(6):
        b.push(np.full((480, 640, 3), i * 30, np.uint8))
    a = b.get()
    assert a.shape == (4, 3, 96, 96) and len(b.frames) == 4
    assert a[0].mean() < a[3].mean()

def test_pitch_moves_points_up():
    from nomad_pipeline.projection import project_points
    uv0, _ = project_points([[2.0, 0.0]], K, 0.4)
    uv1, _ = project_points([[2.0, 0.0]], K, 0.4, pitch=0.3)
    assert uv1[0, 1] < uv0[0, 1] and np.isclose(uv1[0, 0], K[0, 2])


def test_hfov_intrinsics():
    k = Config(hfov_deg=90).intrinsics(640, 480)
    assert np.isclose(k[0, 0], 320.0)


def test_bimodality_metric():
    from nomad_pipeline.metrics import bimodality
    rng = np.random.default_rng(1)
    t = np.linspace(0.1, 1, 8)[None, :, None]
    left = np.concatenate([t * 2, t * 0.8], -1) + rng.normal(0, .02, (4, 8, 2))
    right = np.concatenate([t * 2, -t * 0.8], -1) + rng.normal(0, .02, (4, 8, 2))
    bi, sizes = bimodality(np.concatenate([left, right]))
    uni, _ = bimodality(left.repeat(2, 0) + rng.normal(0, .02, (8, 8, 2)))
    assert sorted(sizes) == [4, 4] and bi > 2 * uni


def test_select_best_trajectory_and_filter():
    from nomad_pipeline.controller import select_best_trajectory, FlightCommandFilter
    t = np.linspace(0.2, 1.6, 8)[:, None]
    left_branch = np.hstack([t, 0.5 * t])
    right_branch = np.hstack([t, -0.5 * t])
    zig_zag = np.hstack([t, np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5])[:, None]])
    samples = np.stack([zig_zag, right_branch, left_branch], axis=0)

    # Якщо попередній крок був ліворуч — має обрати ліву гілку
    chosen = select_best_trajectory(samples, prev_wp=left_branch)[0]
    assert np.allclose(chosen, left_branch)

    # Перевірка EMA фільтра, deadband та аварійного гальма щита
    flt = FlightCommandFilter(alpha=0.5, w_deadband=0.05)
    v1, w1 = flt.update(0.5, 0.02, shield_scale=1.0)
    assert np.isclose(v1, 0.5) and w1 == 0.0  # deadband прибрав мікротремтіння
    v2, w2 = flt.update(0.5, 0.40, shield_scale=0.0)
    assert v2 == 0.0 and np.isclose(w2, 0.21)  # EMERGENCY_BRAKE зупинив v, але зберіг плавний поворот w
