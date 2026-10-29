import numpy as np
import pytest
from nomad_pipeline.obstacle_shield import ForbiddenZoneManager, MonocularObstacleShield


def test_forbidden_zone_manager_overlap():
    fzm = ForbiddenZoneManager()
    fzm.add_zone(100, 100, 200, 200, "WINDOW")

    assert len(fzm.zones) == 1
    # Overlapping box
    assert fzm.overlaps_box(150, 150, 250, 250) is True
    # Non-overlapping box
    assert fzm.overlaps_box(10, 10, 50, 50) is False

    fzm.clear()
    assert len(fzm.zones) == 0


def test_obstacle_shield_blank_wall():
    shield = MonocularObstacleShield(wall_lap_thresh=20.0)

    # Completely flat gray image (blank wall)
    blank_wall = np.full((300, 400, 3), 150, dtype=np.uint8)
    res = shield.evaluate(blank_wall)

    assert res["state"] == "EMERGENCY_BRAKE"
    assert res["hazard"] == "BLANK_WALL"
    assert res["speed_scale"] == 0.0


def test_obstacle_shield_window_glare():
    shield = MonocularObstacleShield(window_glare_thresh=8.0)

    # Image with high brightness in the center (window daylight)
    glare_img = np.full((300, 400, 3), 100, dtype=np.uint8)
    glare_img[50:250, 100:300] = 250 # Pure white overexposed window
    res = shield.evaluate(glare_img)

    assert res["state"] == "EMERGENCY_BRAKE"
    assert res["hazard"] == "WINDOW_GLARE"
    assert res["speed_scale"] == 0.0
