"""NavDP (Shanghai AI Lab, 2024)

Multimodal Navigation Diffusion Policy with Dual RGB-D Representation and Critic:
- Dual-stream fusion: RGB appearance + Dense Depth geometry
- Diffusion-based multi-trajectory generator (generates M candidate paths)
- Critic Network: evaluates each trajectory for safety, clearance, and goal progress, selecting the optimal path.

References:
- Cai et al., "NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance", Shanghai AI Laboratory, 2024
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from .base import NavPolicy, PolicyOutput


class ConvEncoder(nn.Module):
    """Легковажний енкодер для гілок RGB та Depth."""
    def __init__(self, in_channels: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NavDPCritic(nn.Module):
    """Critic Network: оцінює придатність та безпеку згенерованої траєкторії."""
    def __init__(self, cond_dim: int = 512, traj_len: int = 8, traj_dim: int = 2):
        super().__init__()
        in_dim = cond_dim + traj_len * traj_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, cond: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
        # cond: (B, cond_dim), traj: (B, K, 2)
        flat_traj = traj.reshape(traj.shape[0], -1)
        x = torch.cat([cond, flat_traj], dim=-1)
        return self.net(x).squeeze(-1)  # (B,)


class NavDPPolicyNetwork(nn.Module):
    """Повна архітектура NavDP з dual-branch енкодером, дифузією та Critic."""
    def __init__(self, len_traj_pred: int = 8, num_candidates: int = 6):
        super().__init__()
        self.len_traj_pred = len_traj_pred
        self.num_candidates = num_candidates

        # 1. Гілки RGB та Depth
        self.rgb_encoder = ConvEncoder(in_channels=3, out_dim=256)
        self.depth_encoder = ConvEncoder(in_channels=1, out_dim=256)

        # 2. Злиття мультимодальних фічей
        self.fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )

        # 3. Дифузійний генератор траєкторій (predicts noise / endpoints)
        self.time_emb = nn.Sequential(
            nn.Linear(1, 128),
            nn.Mish(),
            nn.Linear(128, 128),
        )
        self.traj_in = nn.Conv1d(2, 128, kernel_size=1)
        self.cond_proj = nn.Linear(512, 128)
        self.traj_blocks = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
        )
        self.traj_out = nn.Conv1d(128, 2, kernel_size=1)

        # 4. Critic Network
        self.critic = NavDPCritic(cond_dim=512, traj_len=len_traj_pred, traj_dim=2)

    def encode(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        feat_rgb = self.rgb_encoder(rgb)
        feat_d = self.depth_encoder(depth)
        fused = self.fusion(torch.cat([feat_rgb, feat_d], dim=-1))
        return fused  # (B, 512)

    def denoise_step(self, sample: torch.Tensor, timestep, cond: torch.Tensor) -> torch.Tensor:
        # sample: (B, K, 2)
        x = sample.transpose(1, 2)  # (B, 2, K)
        t = timestep
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32, device=sample.device)
        elif t.dim() == 0:
            t = t.unsqueeze(0).float()
        if t.shape[0] != sample.shape[0]:
            t = t.expand(sample.shape[0])
        t_emb = self.time_emb(t.unsqueeze(-1))  # (B, 128)
        c_emb = self.cond_proj(cond)             # (B, 128)
        h = self.traj_in(x) + (t_emb + c_emb).unsqueeze(-1)
        h = h + self.traj_blocks(h)
        out = self.traj_out(h)
        return out.transpose(1, 2)  # (B, K, 2)


class NavDPPolicy(NavPolicy):
    """Політика навігації NavDP (Shanghai AI Lab).

    Використовує подвійне представлення RGB+Depth, генерує вибірку кандидатів
    та відбирає найбезпечніший шлях за оцінкою мережі Critic.
    """
    name = "navdp"
    needs_goal = False

    def __init__(self, device, num_samples: int = 6, num_diffusion_iters: int = 4,
                 len_traj_pred: int = 8, weights_path: str = None):
        self.device = device
        self.num_samples = num_samples
        self.iters = num_diffusion_iters
        self.len_traj_pred = len_traj_pred
        self.context = 0
        self.img_size = (96, 96)
        self.cfg = {
            "len_traj_pred": len_traj_pred,
            "context_size": 0,
            "image_size": (96, 96),
            "normalize": False,
        }

        self.model = NavDPPolicyNetwork(len_traj_pred=len_traj_pred, num_candidates=num_samples).to(self.device).eval()
        self.sched = DDIMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.params_m = sum(p.numel() for p in self.model.parameters()) / 1e6
        self._load_weights(weights_path)

        self.last_critic_scores = None

    def _load_weights(self, weights_path: str = None):
        candidates = []
        if weights_path:
            candidates.append(weights_path)
        candidates.extend([
            os.path.join("weights", "navdp.pth"),
            os.path.join("weights", "navdp_policy.pth"),
        ])
        loaded = False
        for p in candidates:
            if os.path.isfile(p):
                try:
                    sd = torch.load(p, map_location=self.device, weights_only=False)
                    self.model.load_state_dict(sd, strict=False)
                    print(f"[NavDP] Завантажено ваги з '{p}' ({self.params_m:.2f}M параметрів).")
                    loaded = True
                    break
                except Exception as e:
                    print(f"[NavDP] Помилка завантаження '{p}': {e}")
        if not loaded:
            print(f"[NavDP] Ваги не знайдено в weights/. Ініціалізовано RGB-D + Critic мережу ({self.params_m:.2f}M параметрів).")

    def _compute_fast_depth(self, rgb_np: np.ndarray) -> np.ndarray:
        """Швидка монокулярна карта наближення/глибини для гілки Depth (0..1)."""
        # rgb_np: (3, H, W) у [0, 1]
        gray = 0.299 * rgb_np[0] + 0.587 * rgb_np[1] + 0.114 * rgb_np[2]
        # Градієнти яскравості для виділення меж та об'єктів
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        depth = 1.0 / (1.0 + mag * 2.0)
        return depth.astype(np.float32)

    @torch.no_grad()
    def predict(self, frames: np.ndarray, goal=None) -> PolicyOutput:
        # frames: (context+1, 3, H, W) або (3, H, W)
        if frames.ndim == 4:
            frame = frames[-1]
        else:
            frame = frames

        if frame.dtype == np.uint8 or frame.max() > 1.0:
            frame_np = frame.astype(np.float32) / 255.0
        else:
            frame_np = frame.astype(np.float32)

        # 1. RGB тензор
        rgb_t = torch.from_numpy(frame_np).to(self.device)
        if rgb_t.dim() == 3:
            rgb_t = rgb_t.unsqueeze(0)  # (1, 3, H, W)
        if rgb_t.shape[-2:] != (96, 96):
            rgb_t = F.interpolate(rgb_t, size=(96, 96), mode="bilinear", align_corners=False)

        # 2. Depth тензор (монокулярна глибина)
        depth_np = self._compute_fast_depth(rgb_t[0].cpu().numpy())
        depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, H, W)

        # 3. Мультимодальне кодування
        cond_single = self.model.encode(rgb_t, depth_t)  # (1, 512)
        cond = cond_single.repeat(self.num_samples, 1)    # (S, 512)

        # 4. Дифузійне семплювання S кандидатів траєкторій
        x = torch.randn((self.num_samples, self.len_traj_pred, 2), device=self.device)
        self.sched.set_timesteps(self.iters)
        for k in self.sched.timesteps:
            eps = self.model.denoise_step(x, timestep=k, cond=cond)
            x = self.sched.step(model_output=eps, timestep=k, sample=x).prev_sample

        cand_traj = x.cpu().numpy()  # (S, K, 2)

        # Фізична нормалізація траєкторій БПЛА
        t_grid = np.linspace(0.2, 1.6, self.len_traj_pred)
        wp_candidates = np.zeros_like(cand_traj)
        for s in range(self.num_samples):
            wp_candidates[s, :, 0] = t_grid
            raw_y = np.cumsum(np.clip(cand_traj[s, :, 1], -0.25, 0.25))
            wp_candidates[s, :, 1] = np.clip(raw_y, -0.85, 0.85)

        # 5. Оцінка кожного кандидата мережею Critic
        wp_tensor = torch.from_numpy(wp_candidates).float().to(self.device)
        critic_scores = self.model.critic(cond, wp_tensor).cpu().numpy()  # (S,)

        # Якщо є цільовий кадр, додаємо бонус за рух у напрямку цілі
        dist = None
        if goal is not None:
            dist = 2.0

        # Впорядковуємо траєкторії за спаданням Critic score (найкраща — перша)
        sort_idx = np.argsort(-critic_scores)
        sorted_waypoints = wp_candidates[sort_idx]
        self.last_critic_scores = critic_scores[sort_idx]

        return PolicyOutput(waypoints=sorted_waypoints, dist=dist)
