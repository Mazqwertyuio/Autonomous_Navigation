from collections import deque
import cv2
import numpy as np

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FrameBuffer:
    """Буфер останніх N кадрів I_{t-N+1..t}. Холодний старт: дублюємо перший кадр."""

    def __init__(self, size: int = 4, img_size=(96, 96)):
        self.size = size
        self.img_size = img_size
        self.frames = deque(maxlen=size)

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, self.img_size, interpolation=cv2.INTER_CUBIC)
        x = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
        return x.transpose(2, 0, 1)  # (3,H,W)

    def push(self, bgr: np.ndarray) -> None:
        x = self.preprocess(bgr)
        if not self.frames:
            self.frames.extend([x] * self.size)
        else:
            self.frames.append(x)

    def get(self) -> np.ndarray:
        """(size,3,H,W), порядок від старого до нового."""
        assert self.frames, "buffer empty"
        return np.stack(self.frames, axis=0)
