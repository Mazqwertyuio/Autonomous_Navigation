import numpy as np
import torch
from .base import NavPolicy, PolicyOutput
from .loader import load_model


class _DistWaypointPolicy(NavPolicy):
    """GNM / ViNT: (obs-стек, ціль) -> (dist, waypoints). Потребує цільового кадру."""
    name = ""
    needs_goal = True

    def __init__(self, device, spacing: float = 0.25):
        self.device = device
        self.spacing = spacing
        self.model, self.cfg = load_model(self.name, device)
        self.context = self.cfg["context_size"]
        self.img_size = tuple(self.cfg["image_size"])

    @torch.no_grad()
    def predict(self, frames, goal=None):
        if goal is None:
            raise ValueError(f"{self.name} потребує цільового кадру (needs_goal=True)")
        n, c, h, w = frames.shape
        obs = torch.from_numpy(frames.reshape(1, n * c, h, w)).float().to(self.device)
        g = torch.from_numpy(goal[None]).float().to(self.device)
        dist, act = self.model(obs, g)
        wp = act[0, :, :2].cpu().numpy()
        if self.cfg["normalize"]:
            wp = wp * self.spacing
        return PolicyOutput(waypoints=wp, dist=float(dist[0, 0]))


class GNMPolicy(_DistWaypointPolicy):
    name = "gnm"


class ViNTPolicy(_DistWaypointPolicy):
    name = "vint"
