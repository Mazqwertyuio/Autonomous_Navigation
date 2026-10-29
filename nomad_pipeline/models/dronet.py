"""PULP-DroNet / DroNet v2 (ETH Zurich / RPG Scaramuzza)

Ultra-lightweight ResNet-8 CNN for autonomous drone navigation:
- Parameters: ~0.32 million (in 60x smaller than NoMaD)
- Speed: 300+ FPS on CPU
- Outputs:
    1. steer [-1.0, 1.0]: Continuous steering / yaw command
    2. collision [0.0, 1.0]: Collision risk probability

References:
- Loquercio et al., "DroNet: Learning to Fly by Driving", IEEE RA-L 2018
- Palossi et al., "A 64mW DNN-based Visual Navigation Engine for Autonomous Mini-Drones", 2019
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from .base import NavPolicy, PolicyOutput


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class DroNet(nn.Module):
    """Офіційна архітектура DroNet (ResNet-8) з двома головами:

    - Steering head: регресія кута повороту [-1, 1]
    - Collision head: бінарна класифікація ймовірності зіткнення [0, 1]
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        # Initial stem
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 3 Residual blocks
        self.layer1 = ResidualBlock(32, 32, stride=1)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Steering Head: Regression
        self.steering_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Collision Head: Sigmoid Probability
        self.collision_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # Prior for collision: initial risk low (safe forward)
        if hasattr(self.collision_fc[-2], "bias") and self.collision_fc[-2].bias is not None:
            nn.init.constant_(self.collision_fc[-2].bias, -1.8)

    def forward(self, x: torch.Tensor):
        feat = self.conv_stem(x)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)

        steer = self.steering_fc(feat)
        collision = self.collision_fc(feat)
        return steer, collision


class DroNetPolicy(NavPolicy):
    """Політика навігації DroNet для БПЛА.

    - Приймає RGB/BGR кадри
    - Оцінює ризик зіткнення та напрямок обходу
    - Будує кінематичну дугу траєкторії (len_traj_pred точок) для контролера Pure Pursuit
    """
    name = "dronet"
    needs_goal = False

    def __init__(self, device, weights_path: str = None, len_traj_pred: int = 8,
                 v_nominal: float = 0.5, w_gain: float = 1.0):
        self.device = device
        self.len_traj_pred = len_traj_pred
        self.v_nominal = v_nominal
        self.w_gain = w_gain
        self.context = 0
        self.img_size = (200, 200)
        self.cfg = {
            "len_traj_pred": len_traj_pred,
            "context_size": 0,
            "image_size": (200, 200),
            "normalize": False,
        }

        self.model = DroNet(in_channels=1).to(self.device).eval()
        self.params_m = sum(p.numel() for p in self.model.parameters()) / 1e6

        # Спроба завантажити ваги з можливих локацій
        self._load_weights(weights_path)

        self.last_steer = 0.0
        self.last_coll_prob = 0.0
        self.v_smooth = self.v_nominal
        self.w_smooth = 0.0

    def _load_weights(self, weights_path: str = None):
        candidates = []
        if weights_path:
            candidates.append(weights_path)
        candidates.extend([
            os.path.join("weights", "dronet.pth"),
            os.path.join("weights", "dronet_v2_gapflow_original.pth"),
            os.path.join("third_party", "pulp-dronet", "pulp-dronet-v2", "model", "dronet_v2_gapflow_original.pth"),
        ])

        loaded = False
        for p in candidates:
            if os.path.isfile(p):
                try:
                    sd = torch.load(p, map_location=self.device, weights_only=False)
                    if isinstance(sd, dict) and "state_dict" in sd:
                        sd = sd["state_dict"]
                    self.model.load_state_dict(sd, strict=False)
                    print(f"[DroNet] Завантажено ваги з '{p}' ({self.params_m:.2f}M параметрів).")
                    loaded = True
                    break
                except Exception as e:
                    print(f"[DroNet] Попередження: не вдалося завантажити '{p}': {e}")
        if not loaded:
            print(f"[DroNet] Ваги не знайдено в weights/. Ініціалізовано ResNet-8 ({self.params_m:.2f}M параметрів).")

    @torch.no_grad()
    def predict(self, frames: np.ndarray, goal=None) -> PolicyOutput:
        # frames: (context+1, 3, H, W) або (3, H, W) або (H, W, 3)
        if frames.ndim == 4:
            frame = frames[-1]  # останній актуальний кадр
        else:
            frame = frames

        if frame.ndim == 3 and frame.shape[0] == 3:
            # (3, H, W) -> Grayscale (H, W)
            # frame у діапазоні [0, 1] або [0, 255]
            if frame.dtype == np.uint8 or frame.max() > 1.0:
                frame_f = frame.astype(np.float32) / 255.0
            else:
                frame_f = frame.astype(np.float32)
            # Y = 0.299 R + 0.587 G + 0.114 B
            gray = 0.299 * frame_f[0] + 0.587 * frame_f[1] + 0.114 * frame_f[2]
        elif frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            gray = frame.astype(np.float32)
            if gray.max() > 1.0:
                gray /= 255.0

        if gray.shape != (200, 200):
            gray = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_AREA)

        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.device)
        steer_t, coll_t = self.model(tensor)

        steer_val = float(np.clip(steer_t.item(), -1.0, 1.0))
        coll_prob = float(np.clip(coll_t.item(), 0.0, 1.0))

        self.last_steer = steer_val
        self.last_coll_prob = coll_prob

        # Закон керування DroNet (Loquercio / Scaramuzza Lab):
        # Швидкість пропорційна безпеці: v = v_nominal * (1 - P_coll)
        v_target = max(0.08, self.v_nominal * (1.0 - coll_prob))
        w_target = steer_val * self.w_gain

        self.v_smooth = 0.75 * v_target + 0.25 * self.v_smooth
        self.w_smooth = 0.75 * w_target + 0.25 * self.w_smooth

        # Генерація точок траєкторії вперед
        t = np.linspace(0.2, 1.6, self.len_traj_pred)
        x_pts = np.maximum(0.15, self.v_smooth / max(self.v_nominal, 1e-3)) * t
        y_pts = 0.5 * self.w_smooth * (t ** 1.25)
        wp = np.stack([x_pts, y_pts], axis=1)

        return PolicyOutput(waypoints=wp, dist=None)
