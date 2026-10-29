"""Тестовий запуск автопілота на відеофайлі (Offline Video Benchmark / Simulation).

Підтримувані режими:
  --mode ultimate : MonoNav Metric Ultimate (обхід кутів стін за кінематикою Delta-yaw)
  --mode ego      : HKUST EGO-Planner v2 (B-сплайн планування в коридорах)
  --mode follow   : MonoNav MediaPipe Person Follower (супровід та трекінг людини)

Приклад запуску:
  py -3.11 run_video_test.py --mode ultimate
  py -3.11 run_video_test.py --mode ego
  py -3.11 run_video_test.py --mode follow
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(description="Run Autopilot Test on Video")
    parser.add_argument("--mode", type=str, default="ultimate", choices=["ultimate", "ego", "follow"],
                        help="Режим автопілота: ultimate, ego, або follow")
    parser.add_argument("--video", type=str, default=None,
                        help="Шлях до відеофайлу (за замовчуванням береться відповідний політ з real_flights_signed)")
    parser.add_argument("--max-frames", type=int, default=150,
                        help="Максимальна кількість тестових кадрів для прогону")
    parser.add_argument("--save-output", action="store_true", default=True,
                        help="Зберегти оброблене відео з накладеною навігацією")
    args = parser.parse_args()

    # Автоматичний вибір відповідного відео, якщо не задано вручну
    signed_dir = os.path.join(ROOT_DIR, "real_flights_signed")
    if args.video is None:
        if args.mode == "ultimate":
            video_path = os.path.join(signed_dir, "Flight_3_MonoNav_Ultimate_Full_Flight_12-35-29.mp4")
        elif args.mode == "ego":
            video_path = os.path.join(signed_dir, "Flight_5_HKUST_EGO_Planner_v2_Full_Flight_12-54-54.mp4")
        else: # follow
            video_path = os.path.join(signed_dir, "Flight_11_MediaPipe_Follow_Success_15-16-39.mp4")
    else:
        video_path = args.video

    if not os.path.exists(video_path):
        print(f"[ERROR] Відеофайл не знайдено: {video_path}")
        sys.exit(1)

    print("=" * 72)
    print("      🚀 ТЕСТОВИЙ ЗАПУСК АВТОПІЛОТА НА ВІДЕОФАЙЛІ")
    print("=" * 72)
    print(f"  Режим:            {args.mode.upper()}")
    print(f"  Відео:            {os.path.basename(video_path)}")
    print(f"  Ліміт кадрів:     {args.max_frames}")
    print("=" * 72)

    # Ініціалізація обраного автопілота
    print(f"\n[INIT] Завантаження модуля {args.mode.upper()}...")
    t0 = time.time()
    if args.mode == "ultimate":
        from mononav_metric_ultimate import MonoNavMetricUltimate
        planner = MonoNavMetricUltimate(device="cuda")
    elif args.mode == "ego":
        from ego_planner_v2 import EgoPlannerV2
        planner = EgoPlannerV2(device="cuda")
    else: # follow
        from mononav_person_follower import MonoNavPersonFollower
        planner = MonoNavPersonFollower()
    print(f"[INIT] Модель завантажено за {time.time() - t0:.2f} сек!\n")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_writer = None
    output_path = os.path.join(ROOT_DIR, f"test_output_{args.mode}.mp4")
    if args.save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    print(f"[RUN] Старт обробки (всього у відео {total_frames} кадрів)...")
    frame_idx = 0
    inference_times = []

    try:
        while cap.isOpened() and frame_idx < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            t_start = time.time()
            if args.mode == "ultimate":
                res = planner.plan(frame)
                vis_frame = planner.render(frame, res)
                mode_str = res.get("mode", "FLY")
                dist = res.get("front_dist", 0.0)
                status_info = f"MODE: {mode_str} | Front Dist: {dist:.2f}m"
            elif args.mode == "ego":
                res = planner.plan(frame)
                vis_frame = planner.render(frame, res)
                n_obs = res.get("n_obs", 0)
                status_info = f"EGO B-SPLINE | Traj: {len(res.get('trajectory', []))} pts | Obs: {n_obs}"
            else: # follow
                res = planner.plan_follow(frame)
                vis_frame = planner.render_hud(frame, res)
                status_info = res.get("status_text", "")

            step_time_ms = (time.time() - t_start) * 1000.0
            inference_times.append(step_time_ms)

            # Накладання метрики FPS
            cur_fps = 1000.0 / max(step_time_ms, 1.0)
            cv2.putText(vis_frame, f"Frame {frame_idx}/{args.max_frames} | {cur_fps:.1f} FPS ({step_time_ms:.1f}ms)",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if out_writer:
                out_writer.write(vis_frame)

            if frame_idx % 25 == 0 or frame_idx == 1:
                print(f"  [Кадр {frame_idx:03d}/{args.max_frames}] {step_time_ms:.1f} мс ({cur_fps:.1f} FPS) -> {status_info}")

    finally:
        cap.release()
        if out_writer:
            out_writer.release()

    avg_time = np.mean(inference_times) if inference_times else 0.0
    avg_fps = 1000.0 / avg_time if avg_time > 0 else 0.0

    print("\n" + "=" * 72)
    print("      ✅ ТЕСТУВАННЯ УСПІШНО ЗАВЕРШЕНО!")
    print("=" * 72)
    print(f"  Опрацьовано кадрів:    {frame_idx}")
    print(f"  Середній час кроку:   {avg_time:.2f} мс")
    print(f"  Середня швидкість:    {avg_fps:.1f} FPS (РЕАЛЬНИЙ ЧАС)")
    if args.save_output and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Результат збережено:  {os.path.basename(output_path)} ({size_mb:.2f} МБ)")
    print("=" * 72)

if __name__ == "__main__":
    main()
