import numpy as np
import pytest
from nomad_pipeline.frame_buffer import FrameBuffer
from nomad_pipeline.frontier_manager import FrontierSubgoalManager


def test_frontier_subgoal_sampling():
    mgr = FrontierSubgoalManager(crop_ratio=0.5, jitter=0.1, auto_advance_dist=2.0, max_subgoal_frames=10)
    buf = FrameBuffer(size=4, img_size=(85, 64))

    frame = np.full((720, 960, 3), 100, dtype=np.uint8)
    goal_tensor = mgr.sample_new_goal(frame, buf, random_jitter=False)

    assert goal_tensor.shape == (3, 64, 85)
    assert mgr.subgoal_id == 1
    assert mgr.current_goal_bgr is not None
    assert mgr.current_goal_bgr.shape == (360, 480, 3)


def test_frontier_auto_advance_on_distance():
    mgr = FrontierSubgoalManager(crop_ratio=0.5, jitter=0.0, auto_advance_dist=2.0, max_subgoal_frames=50)
    buf = FrameBuffer(size=4, img_size=(85, 64))
    frame = np.full((720, 960, 3), 100, dtype=np.uint8)

    # Initial step
    mgr.step(frame, buf, current_dist=5.0)
    assert mgr.subgoal_id == 1

    # Distance still far
    mgr.step(frame, buf, current_dist=3.5)
    assert mgr.subgoal_id == 1

    # Distance reached
    mgr.step(frame, buf, current_dist=1.8)
    assert mgr.subgoal_id == 2
    assert "REACHED" in mgr.last_advance_reason


def test_frontier_auto_advance_on_timeout():
    mgr = FrontierSubgoalManager(crop_ratio=0.5, jitter=0.0, auto_advance_dist=2.0, max_subgoal_frames=5)
    buf = FrameBuffer(size=4, img_size=(85, 64))
    frame = np.full((720, 960, 3), 100, dtype=np.uint8)

    # Steps within TTL
    for _ in range(5):
        mgr.step(frame, buf, current_dist=5.0)
    assert mgr.subgoal_id == 1

    # Exceeding TTL
    mgr.step(frame, buf, current_dist=5.0)
    assert mgr.subgoal_id == 2
    assert "TTL_EXPIRED" in mgr.last_advance_reason
