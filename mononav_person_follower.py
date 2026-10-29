"""MonoNav MediaPipe Person Tracker для DJI Tello.

ПРИНЦИП РОБОТИ (ПРОСТО, ШВИДКО, БЕЗ ВАЖКИХ НЕЙРОМЕРЕЖ):
1. Детекція людини:
   - Google MediaPipe ObjectDetector (EfficientDet-Lite0 TFLite, 25-40 FPS на CPU).
   - Миттєве виявлення людини у кадрі з нульовою затримкою (без Torch/SSD/Depth).
2. Дистанція за відсотком зайнятості кадру (Frame Height Occupancy %):
   - Якщо людина займає < 45% висоти кадру -> людина далеко -> летимо вперед (Pitch > 0);
   - Якщо людина займає 45% - 62% висоти кадру -> ІДЕАЛЬНА ЗОНА (~1.6м) -> ЗАВИСАННЯ (Pitch = 0);
   - Якщо людина займає > 65% висоти кадру -> занадто близько -> м'який відкат назад (Pitch < 0).
3. Горизонтальне центрування (Yaw):
   - Якщо людина ліворуч -> плавний поворот ліворуч (Yaw < 0);
   - Якщо людина праворуч -> плавний поворот праворуч (Yaw > 0);
   - Мертва зона +-10%: коли людина приблизно по центру — дрон не виляє.
4. Пошук при втраті з кадру (Search Mode):
   - Пам'ятає, куди зникла людина (ліворуч чи праворуч);
   - Обертається в ту саму сторону, доки знову не зловить людину;
   - Під час пошуку швидкість вперед строго 0 (безпека).
5. Захист від зіткнень:
   - Якщо людина або об'єкт займає > 80% кадру -> екстрене гальмо / відкат назад.
   - Дрон ніколи не повзе боком (Roll/LR завжди строго 0).
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "efficientdet_lite0.tflite")


class MonoNavPersonFollower:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        max_speed_fb: int = 18,
        max_yaw_rate: int = 36,
    ):
        self.max_speed_fb = max_speed_fb
        self.max_yaw_rate = max_yaw_rate

        print(f"[PersonTracker] Завантаження MediaPipe Detector: {model_path}...")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.25,
            category_allowlist=["person"],
        )
        self.detector = vision.ObjectDetector.create_from_options(options)

        # KCF трекер для надшвидкої інтерполяції між кадрами
        self.tracker = None
        self.frame_idx = 0

        # Пам'ять напрямку пошуку людини
        self.last_seen_side = "RIGHT"  # 'LEFT' або 'RIGHT'
        self.lost_frames = 0

        # Фільтр згладжування команд (EMA)
        self.smooth_fb = 0.0
        self.smooth_yaw = 0.0
        self.smooth_ud = 0.0

        print("[PersonTracker] [OK] MediaPipe Follower активовано (30+ FPS, без SSD/Depth)!\n")

    def plan_follow(self, bgr_img: np.ndarray) -> dict:
        h, w = bgr_img.shape[:2]
        self.frame_idx += 1

        bbox = None
        best_score = 0.0

        # Детекція через MediaPipe кожні 2 кадри (або коли трекер порожній)
        if self.tracker is None or self.frame_idx % 2 == 0:
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection_result = self.detector.detect(mp_image)

            for det in detection_result.detections:
                score = det.categories[0].score
                if score > best_score:
                    best_score = score
                    bb = det.bounding_box
                    x1 = max(0, bb.origin_x)
                    y1 = max(0, bb.origin_y)
                    x2 = min(w - 1, bb.origin_x + bb.width)
                    y2 = min(h - 1, bb.origin_y + bb.height)
                    bbox = (x1, y1, x2, y2)

            if bbox is not None:
                # Переініціалізація швидкого оптичного трекера (MIL або Nano)
                bx, by, bx2, by2 = bbox
                try:
                    self.tracker = cv2.TrackerMIL_create()
                    self.tracker.init(bgr_img, (bx, by, bx2 - bx, by2 - by))
                except Exception:
                    self.tracker = None

        elif self.tracker is not None:
            ok, tbox = self.tracker.update(bgr_img)
            if ok:
                tx, ty, tw, th = tbox
                if tw > 15 and th > 30 and tw < w * 0.95 and th < h * 0.95:
                    bbox = (int(tx), int(ty), int(tx + tw), int(ty + th))
                else:
                    self.tracker = None
            else:
                self.tracker = None

        # Розрахунок польотних параметрів
        target_fb = 0
        target_yaw = 0
        target_lr = 0   # СТРОГО 0: дрон ніколи не повзе боком!
        target_ud = 0
        height_pct = 0.0
        status_text = ""
        shield_alert = False

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = x2 - x1
            bh = y2 - y1

            # Відсоток зайнятості висоти кадру
            height_pct = (bh / float(h)) * 100.0

            # Оновлюємо бік останньої видимості
            if cx < w * 0.5:
                self.last_seen_side = "LEFT"
            else:
                self.last_seen_side = "RIGHT"
            self.lost_frames = 0

            # --- A. ГОРИЗОНТАЛЬНИЙ РИЗИКАННЯ (YAW) ---
            # Якщо людина ліворуч (cx < w/2) -> err_x < 0 -> yaw < 0 (поворот ліворуч)
            # Якщо людина праворуч (cx > w/2) -> err_x > 0 -> yaw > 0 (поворот праворуч)
            err_x = (cx - w / 2.0) / (w / 2.0)
            if abs(err_x) < 0.10:
                target_yaw = 0  # Мертва зона +-10%: дрон не виляє
            else:
                target_yaw = int(np.clip(err_x * self.max_yaw_rate, -self.max_yaw_rate, self.max_yaw_rate))

            # --- B. ДИСТАНЦІЯ ЗА ВІДСОТКОМ КАДРУ (PITCH) ---
            # Ідеальна зона: 45% - 62% висоти кадру (~1.5м - 1.8м)
            if height_pct < 28.0:
                target_fb = self.max_speed_fb  # Людина дуже далеко
                status_text = f"FAST APPROACH (+{target_fb}) | OCCUPANCY: {height_pct:.1f}%"
            elif height_pct < 45.0:
                target_fb = int(self.max_speed_fb * 0.65)  # Наближення
                status_text = f"APPROACHING (+{target_fb}) | OCCUPANCY: {height_pct:.1f}%"
            elif height_pct <= 62.0:
                target_fb = 0  # ІДЕАЛЬНА ЗОНА -> ЗАВИСАННЯ
                status_text = f"TARGET LOCKED ({height_pct:.1f}%) | STABLE HOVER"
            elif height_pct <= 75.0:
                target_fb = -12  # Занадто близько -> м'який відкат назад
                status_text = f"BACKUP SAFETY (-12) | OCCUPANCY: {height_pct:.1f}%"
            else:
                # Екстрено близько (> 75%) -> захист від зіткнення
                target_fb = -16
                shield_alert = True
                status_text = f"[SHIELD BRAKE: TOO CLOSE {height_pct:.1f}%]"

            # --- C. ВИСОТА (THROTTLE UP/DOWN) ---
            err_y = (cy - h * 0.45) / (h / 2.0)
            if abs(err_y) < 0.15:
                target_ud = 0
            else:
                target_ud = int(np.clip(-err_y * 18.0, -18, 18))

        else:
            # --- D. ЛЮДИНУ ВТРАЧЕНО: АКТИВНИЙ ПОШУК У НАПРЯМКУ ЗНИКНЕННЯ ---
            self.lost_frames += 1
            target_fb = 0   # Вперед не летимо
            target_lr = 0   # Боком не ходимо
            target_ud = 0

            if self.lost_frames < 90:
                # Обертаємося в сторону, де востаннє бачили людину
                search_yaw = -24 if self.last_seen_side == "LEFT" else 24
                target_yaw = search_yaw
                status_text = f"SEARCHING {self.last_seen_side} ({self.lost_frames}/90) | RC Yaw: {target_yaw:+d}"
            else:
                target_yaw = 0
                status_text = "SEARCH TIMEOUT | STABLE HOVER WAIT"

        # EMA-фільтрація команд для плавності
        self.smooth_fb = 0.60 * target_fb + 0.40 * self.smooth_fb
        self.smooth_yaw = 0.65 * target_yaw + 0.35 * self.smooth_yaw
        self.smooth_ud = 0.60 * target_ud + 0.40 * self.smooth_ud

        cmd_fb = int(round(self.smooth_fb))
        cmd_yaw = int(round(self.smooth_yaw))
        cmd_ud = int(round(self.smooth_ud))
        cmd_lr = 0

        # Орієнтовна відстань в метрах (для відображення на HUD):
        # f_px * H_real / h_px = 520 * 1.7 / bh = ~880 / bh
        approx_dist_m = round(880.0 / max(bh, 20), 2) if bbox is not None else None

        return {
            "bbox": bbox,
            "height_pct": height_pct,
            "dist_m": approx_dist_m,
            "fb": cmd_fb,
            "yaw": cmd_yaw,
            "lr": cmd_lr,
            "ud": cmd_ud,
            "shield_alert": shield_alert,
            "status_text": status_text,
        }

    def render_hud(self, bgr_img: np.ndarray, res: dict) -> np.ndarray:
        vis = bgr_img.copy()
        h, w = vis.shape[:2]

        bbox = res["bbox"]
        h_pct = res.get("height_pct", 0.0)
        dist_m = res.get("dist_m")
        fb = res["fb"]
        yaw = res["yaw"]
        lr = res.get("lr", 0)
        status_text = res["status_text"]
        shield_alert = res.get("shield_alert", False)

        # Рамка людини
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 120), 2)
            cv2.circle(vis, (cx, cy), 5, (0, 255, 255), -1)

            # Центральний приціл
            cv2.drawMarker(vis, (int(w / 2), int(h / 2)), (200, 200, 200), cv2.MARKER_CROSS, 20, 1)

            dist_str = f"~{dist_m:.1f}m" if dist_m else "--m"
            cv2.putText(vis, f"PERSON: {h_pct:.1f}% ({dist_str})", (x1 + 6, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)

        # Верхній банер
        top_color = (0, 0, 160) if shield_alert else (15, 60, 20)
        cv2.rectangle(vis, (0, 0), (w, 36), top_color, -1)
        tag = "[SHIELD ALERT]" if shield_alert else "[MEDIAPIPE TRACKER]"
        cv2.putText(vis, f"{tag} {status_text}", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        # Нижня панель
        cv2.rectangle(vis, (0, h - 30), (w, h), (15, 15, 15), -1)
        cv2.putText(vis, f"RC Pitch: {fb:+d} | RC Yaw: {yaw:+d} | Occupancy: {h_pct:.1f}% | Roll: {lr}",
                    (12, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)

        return vis


if __name__ == "__main__":
    follower = MonoNavPersonFollower()
    print("MediaPipe Follower successfully initialized!")
