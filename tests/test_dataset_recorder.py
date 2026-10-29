import os
import shutil
import pickle
import numpy as np
import pytest

from nomad_pipeline.dataset_recorder import (
    FlightDatasetRecorder,
    draw_flight_minimap,
    draw_recording_hud,
    draw_handheld_hud,
)
from record_flight_dataset import convert_existing_flight_to_dataset


def test_odometry_integration():
    recorder = FlightDatasetRecorder(output_dir="tmp_test_dataset")
    # Дрон летить вперед зі швидкістю 0.5 м/с при курсі 0 рад (вздовж осі X)
    x, y, dist = recorder.update_odometry(dt=1.0, vgx_mps=0.5, vgy_mps=0.0, yaw_rad=0.0)
    assert np.isclose(x, 0.5, atol=1e-3)
    assert np.isclose(y, 0.0, atol=1e-3)
    assert np.isclose(dist, 0.5, atol=1e-3)

    # Розворот на 90 градусів (pi/2) і політ вперед: тепер рух вздовж осі Y
    x2, y2, dist2 = recorder.update_odometry(dt=1.0, vgx_mps=0.5, vgy_mps=0.0, yaw_rad=np.pi / 2)
    assert np.isclose(x2, 0.5, atol=1e-3)
    assert np.isclose(y2, 0.5, atol=1e-3)
    assert np.isclose(dist2, 1.0, atol=1e-3)


def test_dataset_save_and_vint_format(tmp_path):
    out_dir = str(tmp_path / "train")
    recorder = FlightDatasetRecorder(
        output_dir=out_dir,
        traj_name="traj_unit_test",
        save_video=False,
        save_csv=False,
        image_size=(160, 120),
    )
    recorder.start()

    dummy_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    for i in range(15):
        recorder.record_step(
            frame_bgr=dummy_frame,
            vgx_mps=0.3,
            vgy_mps=0.0,
            yaw_deg=10.0 * i,
        )

    res = recorder.save()
    assert res["status"] == "success"
    assert res["frames"] == 15
    assert os.path.exists(res["pkl_path"])
    assert os.path.exists(res["traj_names_file"])

    # Перевірка структури traj_data.pkl
    with open(res["pkl_path"], "rb") as f:
        data = pickle.load(f)
    assert "position" in data
    assert "yaw" in data
    assert data["position"].shape == (15, 2)
    assert data["yaw"].shape == (15,)

    # Перевірка наявності окремих кадрів 0.jpg ... 14.jpg
    for i in range(15):
        img_p = os.path.join(res["traj_dir"], f"{i}.jpg")
        assert os.path.exists(img_p)

    # Перевірка запису в traj_names.txt
    with open(res["traj_names_file"], "r", encoding="utf-8") as f:
        names = f.read().splitlines()
    assert "traj_unit_test" in names


def test_recorder_reset_for_consecutive_clips():
    recorder = FlightDatasetRecorder(output_dir="tmp_test_dataset", traj_name="traj_1")
    recorder.start()
    recorder.record_step(np.zeros((100, 100, 3), dtype=np.uint8), vgx_mps=0.4, vgy_mps=0.0, yaw_deg=0.0)
    assert recorder.frame_count == 1
    assert len(recorder.frames) == 1

    recorder.reset(new_traj_name="traj_2")
    assert recorder.frame_count == 0
    assert len(recorder.frames) == 0
    assert recorder.x == 0.0
    assert recorder.traj_name == "traj_2"


def test_hud_and_minimap_rendering():
    recorder = FlightDatasetRecorder(output_dir="tmp_test_dataset")
    recorder.start()
    recorder.update_odometry(dt=0.1, vgx_mps=0.4, vgy_mps=0.1, yaw_rad=0.3)

    # 1. Міні-карта
    minimap = draw_flight_minimap(
        positions=[(0.0, 0.0), (recorder.x, recorder.y)],
        current_pos=(recorder.x, recorder.y),
        current_yaw_rad=recorder.yaw_rad,
        size=100,
    )
    assert minimap.shape == (100, 100, 3)

    # 2. Повний польотний HUD
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hud = draw_recording_hud(
        frame_bgr=frame,
        recorder=recorder,
        rc_commands=(10, 20, 0, 5),
        battery_pct=88,
        tof_cm=90,
        temp_c=52,
        anchor_info=(25.0, 3, 0.4, 0.2),  # тривога стіни
        flying=True,
    )
    assert hud.shape == (480, 640, 3)

    # 3. Handheld 15s HUD для різних станів
    for st in ("READY", "COUNTDOWN", "RECORDING", "SAVED"):
        h_hud = draw_handheld_hud(
            frame_bgr=frame,
            recorder=recorder,
            state=st,
            remaining_sec=12.5 if st == "RECORDING" else 2.1,
            target_sec=15.0,
            clip_idx=1,
            saved_count=0,
            battery_pct=95,
            temp_c=45,
            tof_cm=105,
            anchor_info=(40.0, 5, 1.2, 0.8),
        )
        assert h_hud.shape == (480, 640, 3)
