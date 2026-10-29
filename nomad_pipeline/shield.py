"""Тактичний щит безпеки (Collision Shield) на базі Depth Anything V2 + Visual LiDAR.

Працює виключно як пасивне аварійне гальмо (не втручається в кут повороту yaw,
щоб не створювати ривків):
  - CLEAR_FORWARD (> 1.35 м)  -> speed_scale = 1.0 (100% швидкості NoMaD)
  - CAUTION_SLOW (0.85..1.35м)-> speed_scale = 0.40 (плавне пригальמוвання)
  - EMERGENCY_BRAKE (< 0.85 м)-> speed_scale = 0.0 (стоп руху вперед, розворот на місці)
"""

import os
import sys
import numpy as np

TELLO_VISION_DIR = r"C:\Users\User\Desktop\tello\tello-vision"
DA2_REPO_DIR = r"C:\Users\User\Desktop\bd_3d\Depth-Anything-V2"


class CollisionShield:
    def __init__(self, brake_dist_m=0.45, caution_dist_m=1.0, stride=3, emergency_only=True):
        for p in (TELLO_VISION_DIR, DA2_REPO_DIR):
            if os.path.exists(p) and p not in sys.path:
                sys.path.insert(0, p)

        from da2_depth import DA2DepthEstimator
        from visual_lidar import VisualLiDARScanner

        ckpt = os.path.join(TELLO_VISION_DIR, "checkpoints", "depth_anything_v2_vits.pth")
        self.da2 = DA2DepthEstimator(checkpoint_path=ckpt if os.path.exists(ckpt) else None)
        self.emergency_only = emergency_only
        self.brake_dist_m = brake_dist_m
        self.scanner = VisualLiDARScanner(
            brake_dist_m=brake_dist_m,
            caution_dist_m=caution_dist_m,
        )
        self.stride = max(1, int(stride))
        self._step = 0
        self.last_info = {
            "state": "CLEAR_FORWARD",
            "min_dist_m": 4.0,
            "speed_scale": 1.0,
        }

    def reset(self):
        self.da2.reset()
        self.scanner.reset()
        self._step = 0

    def evaluate(self, frame_bgr: np.ndarray, forward_speed_mps: float = 0.3) -> dict:
        self._step += 1
        if self._step % self.stride != 1 and self._step > 1:
            return self.last_info

        norm_depth, _, _, _, _ = self.da2.predict(frame_bgr)
        scan = self.scanner.scan(norm_depth, forward_speed_mps=forward_speed_mps)
        c_min = float(scan["sectors"]["center"]["min_m"])

        # Якщо режим "Аварійка": повний політ без сповільнень, ТІЛЬКИ екстрений стоп перед стіною
        if self.emergency_only:
            if c_min < self.brake_dist_m:
                state = "EMERGENCY_BRAKE"
                scale = 0.0
            else:
                state = "CLEAR_FORWARD"
                scale = 1.0
        else:
            state = scan["shield_state"]
            if state == "EMERGENCY_BRAKE":
                scale = 0.0
            elif state == "CAUTION_SLOW":
                ratio = (c_min - self.scanner.brake_dist_m) / max(
                    self.scanner.caution_dist_m - self.scanner.brake_dist_m, 0.1
                )
                scale = float(np.clip(0.25 + 0.65 * ratio, 0.20, 0.85))
            else:
                scale = 1.0

        self.last_info = {
            "state": state,
            "min_dist_m": c_min,
            "speed_scale": scale,
        }
        return self.last_info
