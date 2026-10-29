"""Модуль запису польотів та формування датасетів для дотренування NoMaD / ViNT / DroNet.

Формат датасету повністю сумісний зі специфікацією VisualNav-Transformer (ViNT/NoMaD):
  dataset/
    <split>/ (наприклад train або test)
      traj_names.txt  <- список папок траєкторій
      traj_<id>/
        0.jpg, 1.jpg, ... N-1.jpg  <- RGB кадри (4:3)
        traj_data.pkl             <- {"position": np.ndarray (N, 2), "yaw": np.ndarray (N,)}

Також автоматично зберігає MP4 відео для перегляду людиною та детальну CSV телеметрію.
"""

import os
import time
import pickle
import csv
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any, List


class FlightDatasetRecorder:
    """Клас реєстратора траєкторій польоту для датасету NoMaD / ViNT."""

    def __init__(
        self,
        output_dir: str = "dataset/train",
        dataset_name: str = "tello",
        traj_name: Optional[str] = None,
        save_video: bool = True,
        save_csv: bool = True,
        flights_dir: str = "flights",
        image_size: Tuple[int, int] = (640, 480),  # (W, H)
    ):
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.save_video = save_video
        self.save_csv = save_csv
        self.flights_dir = flights_dir
        self.image_size = image_size

        ts_str = time.strftime("%Y%m%d_%H%M%S")
        self.traj_name = traj_name or f"traj_{ts_str}"
        self.traj_dir = os.path.join(self.output_dir, self.traj_name)

        # Одометрія в метричній системі (метрах) та радіанах
        self.x = 0.0
        self.y = 0.0
        self.yaw_rad = 0.0
        self.total_distance = 0.0
        self.last_step_time: Optional[float] = None

        # Буфери даних
        self.frames: List[np.ndarray] = []
        self.positions: List[Tuple[float, float]] = []
        self.yaws: List[float] = []
        self.telemetry_records: List[Dict[str, Any]] = []

        self.is_recording = False
        self.start_time: Optional[float] = None
        self.frame_count = 0

    def reset(self, new_traj_name: Optional[str] = None):
        """Скидає буфери для запису наступного кліпу (наприклад наступних 15 секунд)."""
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        self.traj_name = new_traj_name or f"traj_{ts_str}"
        self.traj_dir = os.path.join(self.output_dir, self.traj_name)
        self.x = 0.0
        self.y = 0.0
        self.yaw_rad = 0.0
        self.total_distance = 0.0
        self.last_step_time = None
        self.frames.clear()
        self.positions.clear()
        self.yaws.clear()
        self.telemetry_records.clear()
        self.is_recording = False
        self.start_time = None
        self.frame_count = 0

    def start(self):
        """Початок або відновлення запису."""
        self.is_recording = True
        if self.start_time is None:
            self.start_time = time.time()
        self.last_step_time = time.perf_counter()

    def pause(self):
        """Тимчасова пауза запису."""
        self.is_recording = False

    def toggle(self) -> bool:
        """Перемкнути стан запису."""
        if self.is_recording:
            self.pause()
        else:
            self.start()
        return self.is_recording

    def update_odometry(
        self,
        dt: float,
        vgx_mps: float,
        vgy_mps: float,
        yaw_rad: float,
    ) -> Tuple[float, float, float]:
        """Інтегрує швидкість дрона у глобальні координати (X, Y) у метрах.

        Args:
            dt: крок часу (сек)
            vgx_mps: поздовжня швидкість дрона вперед (м/с)
            vgy_mps: поперечна швидкість дрона вбік (м/с)
            yaw_rad: поточний курс дрона (радіани)

        Returns:
            (x, y, total_distance)
        """
        self.yaw_rad = float(yaw_rad)

        # Перетворення швидкостей зв'язаної системи дрона у світову систему координат
        vx_world = vgx_mps * np.cos(self.yaw_rad) - vgy_mps * np.sin(self.yaw_rad)
        vy_world = vgx_mps * np.sin(self.yaw_rad) + vgy_mps * np.cos(self.yaw_rad)

        dx = vx_world * dt
        dy = vy_world * dt

        self.x += float(dx)
        self.y += float(dy)
        step_dist = float(np.hypot(dx, dy))
        self.total_distance += step_dist

        return self.x, self.y, self.total_distance

    def record_step(
        self,
        frame_bgr: np.ndarray,
        vgx_mps: float,
        vgy_mps: float,
        yaw_deg: float,
        extra_telemetry: Optional[Dict[str, Any]] = None,
    ):
        """Записує черговий крок польоту у пам'ять."""
        now = time.perf_counter()
        if self.last_step_time is None:
            dt = 0.05
        else:
            dt = max(1e-4, now - self.last_step_time)
        self.last_step_time = now

        yaw_rad = np.deg2rad(yaw_deg)
        self.update_odometry(dt, vgx_mps, vgy_mps, yaw_rad)

        if not self.is_recording:
            return

        # Зберігаємо кадр у буфер (якщо розмір відрізняється, підганяємо під image_size)
        H, W = frame_bgr.shape[:2]
        if (W, H) != self.image_size:
            frame_to_store = cv2.resize(frame_bgr, self.image_size, interpolation=cv2.INTER_AREA)
        else:
            frame_to_store = frame_bgr.copy()

        self.frames.append(frame_to_store)
        self.positions.append((float(self.x), float(self.y)))
        self.yaws.append(float(self.yaw_rad))

        record = {
            "timestamp": time.time(),
            "frame_idx": self.frame_count,
            "x_m": self.x,
            "y_m": self.y,
            "yaw_rad": self.yaw_rad,
            "yaw_deg": yaw_deg,
            "vgx_mps": vgx_mps,
            "vgy_mps": vgy_mps,
            "dt_s": dt,
        }
        if extra_telemetry:
            record.update(extra_telemetry)
        self.telemetry_records.append(record)

        self.frame_count += 1

    def save(self) -> Dict[str, Any]:
        """Зберігає всі зібрані кадри та одометрію у форматі датасету NoMaD / ViNT."""
        n_frames = len(self.frames)
        if n_frames == 0:
            return {"status": "empty", "frames": 0, "message": "Немає записаних кадрів"}

        os.makedirs(self.traj_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        if self.flights_dir:
            os.makedirs(self.flights_dir, exist_ok=True)

        # 1. Збереження кадрів 0.jpg, 1.jpg, ...
        for idx, frame in enumerate(self.frames):
            frame_path = os.path.join(self.traj_dir, f"{idx}.jpg")
            cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # 2. Збереження traj_data.pkl
        traj_data = {
            "position": np.asarray(self.positions, dtype=np.float32),  # (N, 2)
            "yaw": np.asarray(self.yaws, dtype=np.float32),            # (N,)
        }
        pkl_path = os.path.join(self.traj_dir, "traj_data.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(traj_data, f)

        # 3. Оновлення traj_names.txt (список активних траєкторій у сплиті)
        traj_names_file = os.path.join(self.output_dir, "traj_names.txt")
        existing_names = set()
        if os.path.exists(traj_names_file):
            with open(traj_names_file, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        existing_names.add(stripped)

        if self.traj_name not in existing_names:
            with open(traj_names_file, "a", encoding="utf-8") as f:
                f.write(f"{self.traj_name}\n")

        # 4. Опціонально: зберігаємо MP4 відео для швидкого візуального огляду людиною
        video_path = None
        if self.save_video and self.flights_dir:
            video_path = os.path.join(self.flights_dir, f"{self.traj_name}.mp4")
            H, W = self.frames[0].shape[:2]
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                20.0,
                (W, H),
            )
            for f in self.frames:
                writer.write(f)
            writer.release()

        # 5. Опціонально: зберігаємо CSV файл телеметрії
        csv_path = None
        if self.save_csv and self.flights_dir and self.telemetry_records:
            csv_path = os.path.join(self.flights_dir, f"{self.traj_name}_telemetry.csv")
            fieldnames = list(self.telemetry_records[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.telemetry_records)

        duration = (time.time() - self.start_time) if self.start_time else 0.0

        return {
            "status": "success",
            "traj_name": self.traj_name,
            "traj_dir": self.traj_dir,
            "frames": n_frames,
            "duration_s": duration,
            "distance_m": self.total_distance,
            "pkl_path": pkl_path,
            "traj_names_file": traj_names_file,
            "video_path": video_path,
            "csv_path": csv_path,
        }


def draw_flight_minimap(
    positions: List[Tuple[float, float]],
    current_pos: Tuple[float, float],
    current_yaw_rad: float,
    size: int = 150,
    meters_window: float = 6.0,
) -> np.ndarray:
    """Малює 2D міні-карту траєкторії дрона (вигляд зверху)."""
    m = np.zeros((size, size, 3), dtype=np.uint8)
    m[:] = (26, 26, 26)  # темно-сірий фон

    center = size // 2
    scale = (size / 2.0) / (meters_window / 2.0)

    # Сітка координат
    cv2.line(m, (center, 0), (center, size), (45, 45, 45), 1)
    cv2.line(m, (0, center), (size, center), (45, 45, 45), 1)

    # Початкова точка (0,0) - зелений хрестик
    cv2.circle(m, (center, center), 3, (0, 255, 0), -1)

    # Траєкторія руху
    if len(positions) > 1:
        pts = []
        for x, y in positions:
            # У NoMaD: X - вперед (+вгору на мапі), Y - вправо/вліво (+вправо на мапі)
            px = int(center + y * scale)
            py = int(center - x * scale)
            px = np.clip(px, 0, size - 1)
            py = np.clip(py, 0, size - 1)
            pts.append([px, py])
        pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(m, [pts], isClosed=False, color=(255, 200, 0), thickness=2)

    # Поточне положення дрона (червоне коло з вектором орієнтації)
    cx = int(center + current_pos[1] * scale)
    cy = int(center - current_pos[0] * scale)
    cx = np.clip(cx, 4, size - 5)
    cy = np.clip(cy, 4, size - 5)

    cv2.circle(m, (cx, cy), 5, (0, 0, 255), -1)
    # Стрілка курсу (yaw)
    arrow_len = 12
    # yaw_rad = 0 вказує вздовж +X (вгору на екрані, dy = -len, dx = 0)
    ax = int(cx + arrow_len * np.sin(current_yaw_rad))
    ay = int(cy - arrow_len * np.cos(current_yaw_rad))
    cv2.line(m, (cx, cy), (ax, ay), (0, 255, 255), 2)

    # Окантовка
    cv2.rectangle(m, (0, 0), (size - 1, size - 1), (80, 80, 80), 1)
    cv2.putText(m, "TOP-DOWN (X,Y)", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
    return m


def draw_recording_hud(
    frame_bgr: np.ndarray,
    recorder: FlightDatasetRecorder,
    rc_commands: Tuple[int, int, int, int],  # (lr, fb, ud, yaw)
    battery_pct: int = 100,
    tof_cm: int = 0,
    temp_c: int = 50,
    anchor_info: Optional[Tuple[float, int, float, float]] = None,
    flying: bool = False,
) -> np.ndarray:
    """Малює детальний інформаційний HUD для запису тренувального датасету."""
    vis = frame_bgr.copy()
    H, W = vis.shape[:2]

    # Верхня панель (Header bar)
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (W, 64), (16, 16, 16), -1)
    # Нижня панель (Footer bar)
    cv2.rectangle(overlay, (0, H - 36), (W, H), (16, 16, 16), -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    # 1. Значок запису REC / PAUSE
    t_sec = time.time()
    blinking = (int(t_sec * 2) % 2) == 0
    if recorder.is_recording:
        badge_color = (0, 0, 255) if blinking else (0, 0, 180)
        cv2.circle(vis, (22, 22), 8, badge_color, -1)
        cv2.putText(vis, "REC", (38, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 255), 2)
    else:
        cv2.circle(vis, (22, 22), 8, (0, 200, 255), -1)
        cv2.putText(vis, "PAUSED", (38, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 200, 255), 2)

    # 2. Інформація про тривалість та кадри
    dur_sec = (time.time() - recorder.start_time) if recorder.start_time else 0.0
    mm, ss = int(dur_sec // 60), int(dur_sec % 60)
    fps_est = (recorder.frame_count / max(dur_sec, 0.1)) if dur_sec > 0 else 0.0
    time_str = f"{mm:02d}:{ss:02d}"
    stats_str = f"Time: {time_str} | Frames: {recorder.frame_count:04d} | {fps_est:.1f} FPS"
    cv2.putText(vis, stats_str, (130, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # 3. ID траєкторії та статус польоту
    flight_mode_str = "AIRBORNE" if flying else "GROUNDED"
    flight_color = (0, 255, 120) if flying else (180, 180, 180)
    cv2.putText(vis, f"Traj: {recorder.traj_name} [{flight_mode_str}]", (130, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, flight_color, 1)

    # 4. Телеметрія Tello (справа зверху)
    tello_str = f"Bat: {battery_pct}% | Alt: {tof_cm}cm | Temp: {temp_c}C"
    cv2.putText(vis, tello_str, (W - 270, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

    lr, fb, ud, yaw_cmd = rc_commands
    rc_str = f"RC: lr={lr:+03d} fb={fb:+03d} ud={ud:+03d} yaw={yaw_cmd:+03d}"
    cv2.putText(vis, rc_str, (W - 270, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # 5. Метрична позиція (X, Y) та орієнтація
    deg = int(np.rad2deg(recorder.yaw_rad)) % 360
    pos_str = f"Pos: X={recorder.x:+.2f}m  Y={recorder.y:+.2f}m  Dist={recorder.total_distance:.2f}m  Yaw={deg} deg"
    cv2.putText(vis, pos_str, (20, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    # 6. Індикатор геометричних орієнтирів стін (Wall Anchor Indicator)
    if anchor_info is not None:
        anchor_score, n_lines, canny_pct, floor_pct = anchor_info
        if anchor_score < 35.0:
            warn_color = (0, 0, 255)
            warn_text = f"WALL HAZARD (Anchor={anchor_score:.0f}) - TURN TO OPEN SPACE!"
        elif anchor_score < 90.0:
            warn_color = (0, 180, 255)
            warn_text = f"CAUTION: NEAR WALL (Anchor={anchor_score:.0f})"
        else:
            warn_color = (0, 255, 100)
            warn_text = f"OPEN SPACE (Anchor={anchor_score:.0f}, Lines={n_lines})"

        cv2.rectangle(vis, (W - 380, H - 32), (W - 10, H - 6), (20, 20, 20), -1)
        cv2.rectangle(vis, (W - 380, H - 32), (W - 10, H - 6), warn_color, 1)
        cv2.putText(vis, warn_text, (W - 370, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, warn_color, 1)

    # 7. Міні-мапа в кутку (Top-Down view)
    map_size = 140
    minimap = draw_flight_minimap(
        recorder.positions,
        (recorder.x, recorder.y),
        recorder.yaw_rad,
        size=map_size,
    )
    # Розміщуємо міні-карту в правому нижньому кутку (трохи вище footer)
    mx1 = W - map_size - 12
    my1 = H - map_size - 44
    vis[my1:my1 + map_size, mx1:mx1 + map_size] = minimap

    return vis


def draw_handheld_hud(
    frame_bgr: np.ndarray,
    recorder: FlightDatasetRecorder,
    state: str = "READY",               # READY | COUNTDOWN | RECORDING | SAVED
    remaining_sec: float = 15.0,
    target_sec: float = 15.0,
    clip_idx: int = 1,
    saved_count: int = 0,
    battery_pct: int = 100,
    temp_c: int = 50,
    tof_cm: int = 0,
    anchor_info: Optional[Tuple[float, int, float, float]] = None,
) -> np.ndarray:
    """Малює спеціальний ергономічний інтерфейс для збору датасету дроном у руках."""
    vis = frame_bgr.copy()
    H, W = vis.shape[:2]

    # Верхня і нижня напівпрозорі плашки
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (W, 76), (16, 16, 16), -1)
    cv2.rectangle(overlay, (0, H - 40), (W, H), (16, 16, 16), -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    # 1. Заголовок режиму в руках (Handheld Mode)
    cv2.putText(vis, f"HANDHELD MODE | CLIP #{clip_idx}", (18, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 1)

    tello_info = f"Bat: {battery_pct}% | Temp: {temp_c}C | ToF: {tof_cm}cm | Total Saved: {saved_count}"
    cv2.putText(vis, tello_info, (W - 380, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    # 2. Стан таймера та статус
    if state == "RECORDING":
        elapsed = target_sec - remaining_sec
        pct = np.clip(elapsed / max(target_sec, 0.1), 0.0, 1.0)
        # Червоний миготливий індикатор запису
        t_sec = time.time()
        blink_red = (0, 0, 255) if (int(t_sec * 3) % 2 == 0) else (0, 0, 160)
        cv2.circle(vis, (30, 52), 9, blink_red, -1)
        time_text = f"RECORDING: {remaining_sec:04.1f}s / {target_sec:.0f}s"
        cv2.putText(vis, time_text, (48, 58), cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 0, 255), 2)
        cv2.putText(vis, "WALK FORWARD / TURN NATURALLY", (340, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1)

        # Смуга прогресу (Progress bar) зверху
        bar_w = int(W * pct)
        cv2.rectangle(vis, (0, 72), (bar_w, 76), (0, 0, 255), -1)

    elif state == "COUNTDOWN":
        # Зворотний відлік 3... 2... 1...
        cnt_val = int(np.ceil(remaining_sec))
        cv2.putText(vis, f"GET READY: STARTING IN {cnt_val}s...", (24, 58),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 215, 255), 2)
        # Великий відлік по центру екрана
        cv2.putText(vis, str(cnt_val), (W // 2 - 25, H // 2 + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 255, 255), 4)

    elif state == "SAVED":
        cv2.circle(vis, (30, 52), 9, (0, 255, 100), -1)
        cv2.putText(vis, f"SAVED CLIP #{clip_idx}! PRESS SPACE FOR NEXT", (48, 58),
                    cv2.FONT_HERSHEY_DUPLEX, 0.68, (0, 255, 100), 2)

    else:  # READY
        cv2.circle(vis, (30, 52), 9, (0, 200, 255), -1)
        cv2.putText(vis, "READY: PRESS [SPACE] TO START 15s CLIP", (48, 58),
                    cv2.FONT_HERSHEY_DUPLEX, 0.68, (0, 255, 255), 2)
        cv2.putText(vis, "[Hold Tello at chest height]", (W - 320, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

    # 3. Нижня панель: позиція та орієнтація
    deg = int(np.rad2deg(recorder.yaw_rad)) % 360
    pos_str = f"Walk: X={recorder.x:+.2f}m  Y={recorder.y:+.2f}m  Dist={recorder.total_distance:.2f}m  Yaw={deg} deg"
    cv2.putText(vis, pos_str, (18, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    # 4. Індикатор стін перед камерою
    if anchor_info is not None:
        anchor_score, n_lines, _, _ = anchor_info
        if anchor_score < 35.0:
            w_color = (0, 0, 255)
            w_msg = "WALL HAZARD - GOOD SPOT TO DEMONSTRATE TURN!"
        elif anchor_score < 90.0:
            w_color = (0, 180, 255)
            w_msg = "CAUTION: APPROACHING WALL"
        else:
            w_color = (0, 255, 120)
            w_msg = "OPEN CORRIDOR / ROOM"

        cv2.rectangle(vis, (W - 390, H - 36), (W - 10, H - 6), (20, 20, 20), -1)
        cv2.rectangle(vis, (W - 390, H - 36), (W - 10, H - 6), w_color, 1)
        cv2.putText(vis, w_msg, (W - 380, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, w_color, 1)

    # 5. Міні-мапа пройденого шляху в правому нижньому кутку
    map_size = 140
    minimap = draw_flight_minimap(
        recorder.positions,
        (recorder.x, recorder.y),
        recorder.yaw_rad,
        size=map_size,
    )
    mx1 = W - map_size - 12
    my1 = H - map_size - 48
    vis[my1:my1 + map_size, mx1:mx1 + map_size] = minimap

    return vis

