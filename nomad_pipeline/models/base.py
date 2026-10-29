from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class PolicyOutput:
    waypoints: np.ndarray            # (K,2) або (S,K,2) у системі робота
    dist: Optional[float] = None
    # Reactive policies can supply commands directly; waypoints remain for HUD.
    command: Optional[Tuple[float, float]] = None
    shield_info: Optional[dict] = None


class NavPolicy:
    def predict(self, frames: np.ndarray, goal: Optional[np.ndarray] = None) -> PolicyOutput:
        raise NotImplementedError
