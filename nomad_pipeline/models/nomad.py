import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from .base import NavPolicy, PolicyOutput
from .loader import load_model

ACTION_MIN = np.array([-2.5, -4.0])
ACTION_MAX = np.array([5.0, 4.0])


class NoMaDPolicy(NavPolicy):
    """goal=None -> goal mask=1 (режим дослідження); goal задано -> рух до цілі.

    За замовчуванням використовує прискорений DDIM (4 кроки дифузії та 4 семпли)
    для досягнення ~25 FPS у реальному часі.
    """
    name = "nomad"
    needs_goal = False

    def __init__(self, device, num_samples: int = 4, num_diffusion_iters: int = 4,
                 scheduler: str = "ddim", spacing: float = 0.25, use_finetuned: bool = True):
        self.device = device
        self.num_samples = num_samples
        self.spacing = spacing
        self.model, self.cfg = load_model("nomad", device, use_finetuned=use_finetuned)
        self.context = self.cfg["context_size"]
        self.img_size = tuple(self.cfg["image_size"])
        train_iters = self.cfg["num_diffusion_iters"]
        self.iters = num_diffusion_iters or train_iters
        sched_cls = DDIMScheduler if scheduler == "ddim" else DDPMScheduler
        self.sched = sched_cls(
            num_train_timesteps=train_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    @torch.no_grad()
    def predict(self, frames, goal=None):
        n, c, h, w = frames.shape
        obs = torch.from_numpy(frames.reshape(1, n * c, h, w)).float().to(self.device)
        masked = goal is None
        g = (torch.randn((1, 3, *self.img_size[::-1]), device=self.device) if masked
             else torch.from_numpy(goal[None]).float().to(self.device))
        mask = torch.full((1,), int(masked), dtype=torch.long, device=self.device)
        cond = self.model("vision_encoder", obs_img=obs, goal_img=g, input_goal_mask=mask)
        dist = None
        if not masked:
            dist = float(self.model("dist_pred_net", obsgoal_cond=cond).flatten()[0])
        cond = cond.repeat(self.num_samples, *([1] * (cond.dim() - 1)))
        x = torch.randn((self.num_samples, self.cfg["len_traj_pred"], 2), device=self.device)
        self.sched.set_timesteps(self.iters)
        for k in self.sched.timesteps:
            eps = self.model("noise_pred_net", sample=x, timestep=k, global_cond=cond)
            x = self.sched.step(model_output=eps, timestep=k, sample=x).prev_sample
        d = x.cpu().numpy().reshape(self.num_samples, -1, 2)
        d = (d + 1) / 2 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN   # unnormalize
        wp = np.cumsum(d, axis=1)
        if self.cfg["normalize"]:
            wp = wp * self.spacing
        return PolicyOutput(waypoints=wp, dist=dist)   # (S,K,2)
