"""Генерація порівняльної структури та спліт-відео (Side-by-Side) для 3 версій автопілота.
Версія 1: MonoNav Metric Ultimate (Hypersim Metric LiDAR)
Версія 2: HKUST EGO-Planner v2 (B-Spline Gradient Trajectory Optimization)
Версія 3: DiffusionPolicy4Drone (Conditional DDPM Reverse Diffusion)
"""

import os
import shutil
import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(ROOT_DIR, "autopilots_benchmark")

V1_DIR = os.path.join(BENCH_DIR, "v1_mononav_metric_ultimate")
V2_DIR = os.path.join(BENCH_DIR, "v2_hkust_ego_planner")
V3_DIR = os.path.join(BENCH_DIR, "v3_diffusion_policy")

for d in [V1_DIR, V2_DIR, V3_DIR]:
    os.makedirs(d, exist_ok=True)


def setup_folders():
    # V1
    src_v1_video = os.path.join(ROOT_DIR, "mononav_metric_ultimate_output.mp4")
    src_v1_snap = os.path.join(ROOT_DIR, "mononav_metric_snapshot.jpg")
    if os.path.exists(src_v1_video):
        shutil.copy2(src_v1_video, os.path.join(V1_DIR, "flight_video.mp4"))
    if os.path.exists(src_v1_snap):
        shutil.copy2(src_v1_snap, os.path.join(V1_DIR, "snapshot.jpg"))
    with open(os.path.join(V1_DIR, "info.txt"), "w", encoding="utf-8") as f:
        f.write("Версія 1: MonoNav Metric Ultimate\n"
                "- Архітектура: Depth-Anything-V2 Metric Hypersim (до 20м)\n"
                "- Принцип: Псевдо-LiDAR хмара точок, динамічний вибір безпечних секторів, 180° Emergency Escape.\n"
                "- Запуск на дроні: tello_live_autopilot.py --mode ultimate\n")

    # V2
    src_v2_video = os.path.join(ROOT_DIR, "ego_planner_tttt_flight_output.mp4")
    src_v2_snap = os.path.join(ROOT_DIR, "ego_planner_tttt_snapshot.jpg")
    if os.path.exists(src_v2_video):
        shutil.copy2(src_v2_video, os.path.join(V2_DIR, "flight_video.mp4"))
    if os.path.exists(src_v2_snap):
        shutil.copy2(src_v2_snap, os.path.join(V2_DIR, "snapshot.jpg"))
    with open(os.path.join(V2_DIR, "info.txt"), "w", encoding="utf-8") as f:
        f.write("Версія 2: HKUST EGO-Planner v2\n"
                "- Архітектура: Неперервна градієнтна оптимізація B-сплайнів (L-BFGS-B)\n"
                "- Принцип: Сили відштовхування від перешкод + мінімізація ривка (Jerk) + динамічна гладкість.\n"
                "- Запуск на дроні: tello_live_autopilot.py --mode ego\n")

    # V3
    src_v3_video = os.path.join(ROOT_DIR, "diffusion_policy_tttt_flight_output.mp4")
    src_v3_snap = os.path.join(ROOT_DIR, "diffusion_policy_tttt_snapshot.jpg")
    if os.path.exists(src_v3_video) and os.path.getsize(src_v3_video) > 100000:
        shutil.copy2(src_v3_video, os.path.join(V3_DIR, "flight_video.mp4"))
    if os.path.exists(src_v3_snap):
        shutil.copy2(src_v3_snap, os.path.join(V3_DIR, "snapshot.jpg"))
    with open(os.path.join(V3_DIR, "info.txt"), "w", encoding="utf-8") as f:
        f.write("Версія 3: DiffusionPolicy4Drone\n"
                "- Архітектура: 1D Conditional U-Net + ResNet18-GN + DDPM Reverse Diffusion\n"
                "- Принцип: Генерація 32 тривимірних вейпоінтів з шуму під умовою карти глибини та цілі.\n"
                "- Запуск на дроні: tello_live_autopilot.py --mode dp\n")

    print("[BENCHMARK] Структуру папок по версіях успішно створено у /autopilots_benchmark!")


def make_split_video():
    v1_path = os.path.join(ROOT_DIR, "mononav_metric_ultimate_output.mp4")
    v2_path = os.path.join(ROOT_DIR, "ego_planner_tttt_flight_output.mp4")
    v3_path = os.path.join(ROOT_DIR, "diffusion_policy_tttt_flight_output.mp4")

    if not (os.path.exists(v1_path) and os.path.exists(v2_path) and os.path.exists(v3_path)):
        print("[SPLIT] Чекаємо на готовність усіх трьох відео...")
        return False

    cap1 = cv2.VideoCapture(v1_path)
    cap2 = cv2.VideoCapture(v2_path)
    cap3 = cv2.VideoCapture(v3_path)

    out_split = os.path.join(BENCH_DIR, "comparison_split_3_autopilots.mp4")
    # Кожен суб-кадр 426 x 320 -> Загальний спліт: 1280 x 320 (3 в ряд)
    sub_w, sub_h = 426, 320
    split_w, split_h = sub_w * 3, sub_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_split, fourcc, 15.0, (split_w, split_h))

    print(f"[SPLIT] Створення 3-віконного відео порівняння: {out_split}...")
    frame_count = 0

    while True:
        r1, f1 = cap1.read()
        r2, f2 = cap2.read()
        r3, f3 = cap3.read()

        if not (r1 and r2 and r3):
            break

        f1_s = cv2.resize(f1, (sub_w, sub_h))
        f2_s = cv2.resize(f2, (sub_w, sub_h))
        f3_s = cv2.resize(f3, (sub_w, sub_h))

        # Написи зверху кожного вікна
        cv2.rectangle(f1_s, (0, 0), (sub_w, 28), (10, 10, 10), -1)
        cv2.putText(f1_s, "V1: MonoNav Metric Ultimate", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 1, cv2.LINE_AA)

        cv2.rectangle(f2_s, (0, 0), (sub_w, 28), (10, 10, 10), -1)
        cv2.putText(f2_s, "V2: HKUST EGO-Planner v2", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 50), 1, cv2.LINE_AA)

        cv2.rectangle(f3_s, (0, 0), (sub_w, 28), (10, 10, 10), -1)
        cv2.putText(f3_s, "V3: DiffusionPolicy4Drone", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 50), 1, cv2.LINE_AA)

        # Роздільники
        cv2.line(f1_s, (sub_w - 1, 0), (sub_w - 1, sub_h), (80, 80, 80), 2)
        cv2.line(f2_s, (sub_w - 1, 0), (sub_w - 1, sub_h), (80, 80, 80), 2)

        combined = np.hstack([f1_s, f2_s, f3_s])
        out_writer.write(combined)
        frame_count += 1

    cap1.release()
    cap2.release()
    cap3.release()
    out_writer.release()

    print(f"[SPLIT] Порівняльне спліт-відео успішно збережено ({frame_count} кадрів):")
    print(f"  -> {out_split}")
    return True


if __name__ == "__main__":
    setup_folders()
    make_split_video()
