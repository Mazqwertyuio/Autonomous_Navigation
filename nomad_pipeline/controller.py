import numpy as np


def linear_tello_rc(v, w, v_max, w_max, max_fb_rc=22, max_yaw_rc=28):
    """Baseline unit conversion, without minimum boosts, strafe or turn slowdown."""
    if v_max <= 0 or w_max <= 0 or not np.isfinite([v, w]).all():
        raise ValueError("Require finite commands and positive velocity limits")
    fb = int(round(float(np.clip(v / v_max, -1, 1)) * max_fb_rc))
    yaw = int(round(float(np.clip(-w / w_max, -1, 1)) * max_yaw_rc))
    return 0, fb, 0, yaw


def pure_pursuit(waypoints_xy, lookahead=0.25, v_max=0.5, w_max=1.0, max_points=3):
    """Повертає (v, w) за найближчими 2-3 точками для миттєвої реакції на маневри."""
    p = np.asarray(waypoints_xy, dtype=np.float64).reshape(-1, 2)
    if max_points and len(p) > max_points:
        p = p[:max_points]
    p = p[p[:, 0] > 1e-6] if len(p) else p
    if len(p) == 0:
        return 0.0, 0.0
    d = np.hypot(p[:, 0], p[:, 1])
    idx = np.nonzero(d >= lookahead)[0]
    tx, ty = p[idx[0]] if len(idx) else p[-1]
    L = max(np.hypot(tx, ty), 1e-6)
    curvature = 2.0 * ty / (L * L)
    v = v_max
    w = float(np.clip(curvature * v, -w_max, w_max))
    return float(v), w


def select_best_trajectory(samples_xy, prev_wp=None):
    """Обирає найкращу траєкторію серед S семплів NoMaD (S, K, 2).

    Критерії:
      1. Просування вперед (x_end).
      2. Мінімальна кривизна та бічні коливання (плавність).
      3. Часова узгодженість із траєкторією попереднього кроку (temporal consistency),
         щоб дрон не перестрибував між лівою і правою гілкою щокадру.

    Повертає перевпорядкований масив (S, K, 2), де індекс 0 — найкраща траєкторія.
    """
    arr = np.asarray(samples_xy, dtype=np.float64)
    if arr.ndim == 2:
        return arr[None]
    S = len(arr)
    if S <= 1:
        return arr

    # 1. Просування вперед
    fwd_score = arr[:, -1, 0]

    # 2. Штраф за різку кривизну (друга похідна по y) та сильне бічне зміщення
    dy = np.diff(arr[:, :, 1], axis=1)
    curvature_penalty = np.mean(np.abs(dy), axis=1) + 0.25 * np.abs(arr[:, -1, 1])

    # 3. Часова узгодженість з попереднім кроком
    temp_penalty = np.zeros(S, dtype=np.float64)
    if prev_wp is not None and prev_wp.shape == arr[0].shape:
        temp_penalty = np.mean(np.linalg.norm(arr - prev_wp[None], axis=-1), axis=1)

    # Дозволяємо дрону маневрувати та обирати плавні бічні повороти
    scores = 0.8 * fwd_score - 0.20 * curvature_penalty - 1.0 * temp_penalty
    order = np.argsort(-scores)
    return arr[order]


class FlightCommandFilter:
    """Згладжування команд (v, w) за допомогою EMA-фільтра, зони нечутливості (deadband)
    та інтеграції з тактичним щитом безпеки (Collision Shield).
    """

    def __init__(self, alpha=0.65, w_deadband=0.03):
        self.alpha = alpha
        self.w_deadband = w_deadband
        self.v_smooth = None
        self.w_smooth = None

    def reset(self):
        self.v_smooth = None
        self.w_smooth = None

    def update(self, v_raw: float, w_raw: float, shield_scale: float = 1.0):
        if self.v_smooth is None:
            self.v_smooth = float(v_raw)
            self.w_smooth = float(w_raw)
        else:
            self.v_smooth = self.alpha * float(v_raw) + (1.0 - self.alpha) * self.v_smooth
            self.w_smooth = self.alpha * float(w_raw) + (1.0 - self.alpha) * self.w_smooth

        w_out = 0.0 if abs(self.w_smooth) < self.w_deadband else self.w_smooth
        v_out = self.v_smooth * float(np.clip(shield_scale, 0.0, 1.0))
        return float(v_out), float(w_out)
