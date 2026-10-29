"""Official PULP-DroNet v3 float inference and a reactive navigation adapter."""
import cv2
import numpy as np
from PIL import Image
import torch

from .base import NavPolicy, PolicyOutput
from .dronet_assets import verified_checkpoint
from .vendor.pulp_dronet_v3 import dronet, ResBlock, Depthwise_Separable


def preprocess_bgr(frame_bgr):
    """Official grayscale CenterCrop(200), ToTensor; no resize or normalization."""
    if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
            or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3):
        raise ValueError("DroNet expects a raw uint8 BGR frame with shape (H,W,3)")
    height, width = frame_bgr.shape[:2]
    if min(height, width) < 200:
        raise ValueError("DroNet requires a camera frame at least 200x200 pixels")
    gray = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("L")
    left, top = int(round((width - 200) / 2)), int(round((height - 200) / 2))
    crop = np.array(gray.crop((left, top, left + 200, top + 200)), copy=True)
    return torch.from_numpy(crop).float().div_(255.0)[None, None]


def collision_info(steer, collision, caution_thresh=0.60, stop_thresh=0.80):
    if not 0 <= caution_thresh < stop_thresh <= 1:
        raise ValueError("Require 0 <= caution threshold < stop threshold <= 1")
    if not np.isfinite([steer, collision]).all() or not 0 <= collision <= 1:
        raise RuntimeError("DroNet returned invalid steering or collision probability")
    steer = float(np.clip(steer, -1, 1))
    if collision >= stop_thresh:
        state, scale = "EMERGENCY_BRAKE", 0.0
    elif collision >= caution_thresh:
        state = "CAUTION_SLOW"
        scale = min(1.0 - collision, (stop_thresh - collision) / (stop_thresh - caution_thresh))
    else:
        state, scale = "CLEAR_FORWARD", 1.0 - collision
    return {
        "state": state, "coll_prob": float(collision), "steer": steer,
        "speed_scale": float(scale), "min_dist_m": None,
    }


class DroNetV3Inference:
    def __init__(self, device=None, variant="v3", weights_path=None,
                 caution_thresh=0.60, stop_thresh=0.80):
        collision_info(0.0, 0.0, caution_thresh, stop_thresh)
        path = verified_checkpoint(variant, weights_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu")
        self.variant = variant
        self.weights_path = path.resolve()
        self.caution_thresh, self.stop_thresh = caution_thresh, stop_thresh
        if variant == "v3":
            model = dronet(depth_mult=1.0, block_class=ResBlock, bypass=True)
        else:
            model = dronet(depth_mult=0.125, block_class=Depthwise_Separable, bypass=False)
        # No arbitrary pickle classes, partial state dicts or random-weight fallbacks.
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()
        self.params_m = sum(p.numel() for p in model.parameters()) / 1e6
        self.last_info = None

    @torch.inference_mode()
    def evaluate(self, frame_bgr, forward_speed_mps=0.0):
        tensor = preprocess_bgr(frame_bgr).to(self.device)
        steering, collision = self.model(tensor)
        self.last_info = collision_info(
            float(steering.item()), float(collision.item()),
            self.caution_thresh, self.stop_thresh,
        )
        return self.last_info.copy()

    def reset(self):
        self.last_info = None


class DroNetV3Policy(NavPolicy):
    """Reactive yaw/speed control; contains no goal navigation or scene memory."""
    name = "dronet-v3"
    needs_goal = False
    context = 0
    img_size = (200, 200)

    def __init__(self, device=None, variant="v3", weights_path=None, len_traj_pred=8,
                 v_nominal=0.35, w_max=0.45, caution_thresh=0.60, stop_thresh=0.80):
        if len_traj_pred < 2 or v_nominal <= 0 or w_max <= 0:
            raise ValueError("Require at least two waypoints and positive velocity limits")
        self.detector = DroNetV3Inference(
            device, variant, weights_path, caution_thresh, stop_thresh,
        )
        self.device = self.detector.device
        self.params_m = self.detector.params_m
        self.name = "dronet-v3-tiny" if variant == "tiny-v3" else "dronet-v3"
        self.v_nominal, self.w_max = v_nominal, w_max
        self.cfg = {"len_traj_pred": len_traj_pred, "context_size": 0,
                    "image_size": self.img_size, "normalize": False}

    def predict_bgr(self, frame_bgr, goal=None):
        if goal is not None:
            raise ValueError("DroNet v3 is reactive and cannot navigate to an image goal")
        info = self.detector.evaluate(frame_bgr)
        # +w is leftward, as in the upstream Crazyflie labels. Tello mapping negates it.
        w = info["steer"] * self.w_max
        v = self.v_nominal * info["speed_scale"]
        times = np.linspace(0.2, 1.6, self.cfg["len_traj_pred"])
        if abs(w) < 1e-6:
            points = np.column_stack((v * times, np.zeros_like(times)))
        else:
            points = np.column_stack((v * np.sin(w * times) / w,
                                      v * (1 - np.cos(w * times)) / w))
        # The pipeline applies speed_scale after filtering, so an emergency
        # brake cannot be delayed by EMA and speed is not scaled twice.
        return PolicyOutput(points, command=(self.v_nominal, w), shield_info=info)

    def predict(self, frames, goal=None):
        return self.predict_bgr(frames, goal)

    def reset(self):
        self.detector.reset()
