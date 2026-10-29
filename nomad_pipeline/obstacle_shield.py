"""Тактичний монокулярний щит захисту від вікон, дзеркал та однотонних стін (Monocular Obstacle Shield).

Реалізує фундаментальний принцип:
  «Якщо ми втрачаємо геометричні вказівники (лінія підлоги, кути стін, столи, двері) —
   ми втрачаємо орієнтацію та наближаємося до стіни. Дрон повинен завчасно триматися подалі!»

Рівні безпеки:
  1. ANCHORS_RICH (> 90): Відкритий простір кімнати з чіткими орієнтирами -> 100% швидкості.
  2. CAUTION_SLOW (35..90): Орієнтири розмиваються, наближаємося до стіни -> завчасне сповільнення на 55% і відворот.
  3. EMERGENCY_BRAKE (< 35 або lap_var < 28): Втрата геометричних ліній, плоска стіна -> повний стоп тяги (fb = 0).
  4. FORBIDDEN_ZONE: Користувацькі мітки (закриті вікна, телевізори, дзеркала).
"""

import cv2
import numpy as np
from .projection import project_points


def compute_geometric_anchors(crop_bgr: np.ndarray) -> tuple:
    """Обчислює щільність геометричних орієнтирів: лінії підлоги, кути кімнати, меблі."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=15, maxLineGap=8)
    n_lines = len(lines) if lines is not None else 0
    canny_pct = float(np.mean(edges > 0)) * 100.0

    # Орієнтири нижньої площини (лінія стику підлоги зі стіною, плінтуси)
    floor_pct = float(np.mean(edges[int(H * 0.70):, :] > 0)) * 100.0

    # Енергія градієнтів Sobel (вертикальні та горизонтальні контури)
    sx = float(np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))))
    sy = float(np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))))

    anchor_score = (n_lines * 2.0) + (canny_pct * 12.0) + (floor_pct * 15.0) + (sx + sy) * 0.8
    return anchor_score, n_lines, canny_pct, floor_pct


class ForbiddenZoneManager:
    """Менеджер користувацьких міток заборонених зон (вікна, дзеркала, небезпечні зони)."""

    def __init__(self):
        self.zones = []  # Список кортежів: (x1, y1, x2, y2, label)
        self.drawing = False
        self.x0, self.y0 = 0, 0
        self.current_drag = None

    def add_zone(self, x1: int, y1: int, x2: int, y2: int, label: str = "WINDOW / NO-FLY"):
        xa, xb = min(x1, x2), max(x1, x2)
        ya, yb = min(y1, y2), max(y1, y2)
        if (xb - xa) >= 12 and (yb - ya) >= 12:
            self.zones.append((xa, ya, xb, yb, label))

    def add_center_zone(self, w_img: int, h_img: int, span_pct: float = 0.40, label: str = "WINDOW / NO-FLY"):
        """Швидка мітка центрального сектора як забороненої зони (за клавішею 'w')."""
        cw = int(w_img * span_pct)
        ch = int(h_img * span_pct)
        x1 = (w_img - cw) // 2
        y1 = (h_img - ch) // 2
        self.add_zone(x1, y1, x1 + cw, y1 + ch, label)

    def clear(self):
        self.zones.clear()
        self.current_drag = None

    def overlaps_box(self, rx1: int, ry1: int, rx2: int, ry2: int) -> bool:
        """Перевіряє, чи перетинається прямокутна область із будь-якою забороненою зоною."""
        for zx1, zy1, zx2, zy2, _ in self.zones:
            if not (rx2 < zx1 or rx1 > zx2 or ry2 < zy1 or ry1 > zy2):
                return True
        return False

    def on_mouse(self, event, x, y, flags, param):
        """Інтерактивна розмітка зон мишкою у вікні OpenCV:
        - Затискання ЛКМ + перетягування: малювання червоної рамки забороненої зони.
        - ПКМ (права кнопка миші): очистити всі мітки.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x0, self.y0 = x, y
            self.current_drag = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_drag = (self.x0, self.y0, x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.add_zone(self.x0, self.y0, x, y)
            self.current_drag = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clear()

    def check_trajectory_collision(self, waypoints_xy: np.ndarray, K: np.ndarray, cfg) -> tuple:
        """Проєктує траєкторію на кадр і перевіряє, чи веде вона у заборонену зону."""
        if not self.zones or len(waypoints_xy) == 0:
            return False, 0.0

        uv, ok = project_points(
            waypoints_xy, K,
            h_cam=getattr(cfg, "h_cam", 0.15),
            pitch=np.radians(getattr(cfg, "pitch_deg", 0.0))
        )

        for (u, v), valid in zip(uv, ok):
            if valid:
                for zx1, zy1, zx2, zy2, _ in self.zones:
                    if zx1 <= u <= zx2 and zy1 <= v <= zy2:
                        zone_center_x = (zx1 + zx2) / 2.0
                        img_center_x = K[0, 2]
                        steer_dir = +1.0 if zone_center_x >= img_center_x else -1.0
                        return True, steer_dir

        return False, 0.0

    def draw_hud_zones(self, vis: np.ndarray) -> np.ndarray:
        """Малює червоні напівпрозорі зони відчуження з діагональними лініями безпеки."""
        h, w = vis.shape[:2]
        all_to_draw = list(self.zones)
        if self.current_drag is not None:
            dx1, dy1, dx2, dy2 = self.current_drag
            all_to_draw.append((min(dx1, dx2), min(dy1, dy2), max(dx1, dx2), max(dy1, dy2), "DRAWING..."))

        if not all_to_draw:
            return vis

        overlay = vis.copy()
        for x1, y1, x2, y2, label in all_to_draw:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 220), -1)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (40, 40, 255), 2)
            tag = f"[FORBIDDEN: {label}]"
            cv2.putText(vis, tag, (x1 + 4, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 60, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.28, vis, 0.72, 0, vis)
        return vis


class MonocularObstacleShield:
    """Швидкий монокулярний щит: геометричні орієнтири, детекція стін, вікон та завчасне гальмування."""

    def __init__(self, wall_lap_thresh: float = 28.0, window_glare_thresh: float = 8.5):
        self.wall_lap_thresh = wall_lap_thresh
        self.window_glare_thresh = window_glare_thresh
        self.wall_counter = 0
        self.last_hazard = "CLEAR"

    def evaluate(self, frame_bgr: np.ndarray, fzm: ForbiddenZoneManager = None,
                 waypoints_xy: np.ndarray = None, K: np.ndarray = None, cfg = None) -> dict:
        H, W = frame_bgr.shape[:2]

        # 1. Перевірка ручних заборонених зон (користувацькі мітки вікон)
        if fzm is not None and waypoints_xy is not None and K is not None and cfg is not None:
            collides, steer_dir = fzm.check_trajectory_collision(waypoints_xy, K, cfg)
            if collides:
                self.last_hazard = "FORBIDDEN_ZONE"
                return {
                    "state": "EMERGENCY_BRAKE",
                    "hazard": "FORBIDDEN_ZONE",
                    "speed_scale": 0.0,
                    "steer_bias": steer_dir * 0.35,
                    "need_backoff": False,
                    "anchor_score": 0.0,
                    "label": "BLOCKED: FORBIDDEN ZONE (WINDOW/WALL)",
                }

        # 2. Аналіз центрального сектора камери
        ch1, ch2 = int(H * 0.20), int(H * 0.80)
        cw1, cw2 = int(W * 0.20), int(W * 0.80)
        center = frame_bgr[ch1:ch2, cw1:cw2]

        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_brightness = float(gray.mean())

        # Оцінка геометричних ліній (підлога, плінтуси, кути, столи)
        score_anchors, n_lines, canny_pct, floor_pct = compute_geometric_anchors(center)

        # Відсоток переосвітлених пікселів віконного відблиску
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        glare_pct = float(np.mean(hsv[:, :, 2] > 230)) * 100.0

        # Визначаємо, з якого боку більше текстури та відкритого простору
        full_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        l_half = full_gray[:, :W // 2]
        r_half = full_gray[:, W // 2:]
        lap_l = float(cv2.Laplacian(l_half, cv2.CV_64F).var())
        lap_r = float(cv2.Laplacian(r_half, cv2.CV_64F).var())
        optimal_steer = 0.35 if lap_l >= lap_r else -0.35

        # А) Детекція сонячного відблиску закритого вікна
        if glare_pct >= self.window_glare_thresh or (mean_brightness > 185.0 and lap_var < 35.0):
            self.last_hazard = "WINDOW_GLARE"
            return {
                "state": "EMERGENCY_BRAKE",
                "hazard": "WINDOW_GLARE",
                "speed_scale": 0.0,
                "steer_bias": -optimal_steer,  # Відвертаємося від сліпучого вікна
                "need_backoff": False,
                "anchor_score": score_anchors,
                "label": f"BLOCKED: WINDOW GLARE ({glare_pct:.1f}%)",
            }

        # Б) Екстрене блокування: повна втрата геометричних орієнтирів (стіна)
        is_blank_wall = (lap_var < self.wall_lap_thresh) or (canny_pct < 0.60) or (score_anchors < 35.0)
        if is_blank_wall:
            self.wall_counter += 1
            self.last_hazard = "BLANK_WALL"
            need_backoff = (self.wall_counter >= 3)
            return {
                "state": "EMERGENCY_BRAKE",
                "hazard": "BLANK_WALL",
                "speed_scale": 0.0,
                "steer_bias": optimal_steer,  # Повертаємо у бік більшої текстури кімнати
                "need_backoff": need_backoff,
                "anchor_score": score_anchors,
                "label": f"BLOCKED: NO ANCHORS / WALL (lines={n_lines}, canny={canny_pct:.2f}%)",
            }

        # В) Завчасне попередження: орієнтири слабшають, тримаємося подалі від стіни!
        if score_anchors < 90.0:
            self.wall_counter = max(0, self.wall_counter - 1)
            self.last_hazard = "DECAYING_ANCHORS"
            return {
                "state": "CAUTION_SLOW",
                "hazard": "DECAYING_ANCHORS",
                "speed_scale": 0.45,  # Знижуємо швидкість на 55% завчасно!
                "steer_bias": optimal_steer * 0.4,  # Плавно відхиляємося до структурованого боку
                "need_backoff": False,
                "anchor_score": score_anchors,
                "label": f"CAUTION: LOSING ANCHORS ({score_anchors:.0f}) -> SLOWING DOWN",
            }

        self.wall_counter = max(0, self.wall_counter - 1)
        self.last_hazard = "CLEAR"
        return {
            "state": "CLEAR_FORWARD",
            "hazard": "CLEAR",
            "speed_scale": 1.0,
            "steer_bias": 0.0,
            "need_backoff": False,
            "anchor_score": score_anchors,
            "label": "FORWARD PATH: CLEAR",
        }
