"""Модуль 3D-ліфтингу траєкторій для квадрокоптера DJI Tello.

Інтегрує 2D-планувальник NoMaD (Goal-Masked Diffusion Policy)
з монокулярною картою глибини Depth-Anything-V2:
  Вхід:
    - 2D-траєкторія NoMaD: wp_2d = [8, 2] (x_forward, y_lateral)
    - Оцінка глибини Depth-Anything-V2: depth_map [H, W]
  Вихід:
    - Повноцінна 3D-траєкторія: wp_3d = [8, 3] (x_forward, y_lateral, z_altitude)
    - Висота безпечного кліренсу над меблями / підлогою
    - Аварійне вертикальне уникнення перешкод (Vertical Climb / Dive)
"""

import os
import sys
import numpy as np
import cv2
import torch

DEPTH_DIR = r"C:\Users\User\Desktop\bd_3d\Depth-Anything-V2"
if DEPTH_DIR not in sys.path:
    sys.path.insert(0, DEPTH_DIR)

from depth_anything_v2.dpt import DepthAnythingV2


class TrajectoryLifter3D:
    def __init__(
        self,
        checkpoint_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights", "depth_anything_v2_vits.pth"),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        drone_cruise_alt: float = 0.85, # Базова крейсерська висота Tello над підлогою (м)
        fov_h_deg: float = 82.6,        # Горизонтальний кут огляду камери Tello
        fov_v_deg: float = 65.0,        # Вертикальний кут огляду камери Tello
    ):
        self.device = device
        self.drone_cruise_alt = drone_cruise_alt
        self.fov_h = np.deg2rad(fov_h_deg)
        self.fov_v = np.deg2rad(fov_v_deg)

        print(f"[3D-LIFTER] Ініціалізація Depth-Anything-V2 (vits) на {device}...")
        self.depth_net = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        self.depth_net.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        self.depth_net = self.depth_net.to(self.device).eval()

    def infer_depth(self, bgr_img: np.ndarray) -> np.ndarray:
        """Оцінює нормалізовану відносну карту глибини (disparity/depth)."""
        with torch.no_grad():
            raw_depth = self.depth_net.infer_image(bgr_img, input_size=252)
        # Нормалізуємо в діапазон [0.0, 1.0], де 1.0 = найближче (диспаритет), 0.0 = далечінь
        d_min, d_max = raw_depth.min(), raw_depth.max()
        if d_max > d_min:
            norm_disp = (raw_depth - d_min) / (d_max - d_min)
        else:
            norm_disp = np.zeros_like(raw_depth)
        return norm_disp

    def lift_2d_to_3d(self, wp_2d: np.ndarray, bgr_img: np.ndarray) -> dict:
        """Перетворює 2D вейпоінти [8, 2] (x, y) у повноцінні 3D вейпоінти [8, 3] (x, y, z).
        
        Аналізує 3 зони по висоті кадру (Floor, Obstacle Level, Ceiling)
        вздовж коридору проходження дрона.
        """
        disp = self.infer_depth(bgr_img)
        h, w = disp.shape
        fx = (w / 2.0) / np.tan(self.fov_h / 2.0)
        fy = (h / 2.0) / np.tan(self.fov_v / 2.0)
        cx, cy = w / 2.0, h / 2.0

        wp_3d = np.zeros((len(wp_2d), 3), dtype=np.float32)

        # Аналіз вертикального профілю попереду:
        # 1. Нижня зона (Floor / Low obstacles): h * 0.70 .. h
        # 2. Середня зона (Center corridor / Tables): h * 0.35 .. h * 0.70
        # 3. Верхня зона (Ceiling clearance): 0 .. h * 0.35
        floor_roi = disp[int(h * 0.70):, int(w * 0.25):int(w * 0.75)]
        mid_roi = disp[int(h * 0.35):int(h * 0.70), int(w * 0.25):int(w * 0.75)]
        top_roi = disp[:int(h * 0.35), int(w * 0.25):int(w * 0.75)]

        floor_proximity = float(np.percentile(floor_roi, 80)) if floor_roi.size > 0 else 0.5
        mid_proximity = float(np.percentile(mid_roi, 85)) if mid_roi.size > 0 else 0.5
        top_proximity = float(np.percentile(top_roi, 80)) if top_roi.size > 0 else 0.2

        # Обчислення оптимального вертикального маневру dz
        vertical_adjustment = 0.0
        climb_reason = "CRUISE"

        # Якщо середина заблокована низькою перешкодою (наприклад стіл/стілець), але зверху вільно:
        if mid_proximity > 0.65 and top_proximity < 0.45:
            # Набір висоти для прольоту над перешкодою
            vertical_adjustment = +0.25 * (mid_proximity - 0.50) / 0.50
            climb_reason = "CLIMB_OVER_OBSTACLE"
        # Якщо внизу наближається підлога (дрон просідає або підйом):
        elif floor_proximity > 0.75 and top_proximity < 0.50:
            vertical_adjustment = +0.15
            climb_reason = "GROUND_CLEARANCE_CLIMB"
        # Якщо зверху низька стеля чи балка:
        elif top_proximity > 0.65 and floor_proximity < 0.60:
            vertical_adjustment = -0.20
            climb_reason = "CEILING_CLEARANCE_DIVE"

        # Формуємо 3D точки (x_fwd, y_lat, z_alt)
        for i, pt in enumerate(wp_2d):
            x_fwd = float(pt[0])
            y_lat = float(pt[1])
            
            # Плавний перехід висоти вздовж 8 кроків траєкторії
            step_ratio = (i + 1) / float(len(wp_2d))
            z_alt = self.drone_cruise_alt + vertical_adjustment * step_ratio
            
            wp_3d[i] = [x_fwd, y_lat, z_alt]

        return {
            "wp_3d": wp_3d,
            "disparity_map": disp,
            "climb_reason": climb_reason,
            "vertical_adjustment": float(vertical_adjustment),
            "floor_prox": floor_proximity,
            "mid_prox": mid_proximity,
            "top_prox": top_proximity
        }
