from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    model: str = "nomad"          # gnm | vint | nomad | dronet-v3 | dronet-v3-tiny
    device: str = "auto"          # auto | cpu | cuda
    seed: int = 0
    pure_nomad: bool = False      # isolated goal-free baseline, without autonomy add-ons
    use_obstacle_shield: bool = True
    context: int = 4
    len_traj_pred: int = 8
    num_diffusion_iters: int = 2  # 2 кроки DDIM для високої швидкодії (~8.5+ FPS)
    num_samples: int = 1          # 1 семпл для миттєвого інференсу без затримок
    img_size: tuple = (96, 96)    # (W, H) для моделі
    # камера
    h_cam: float = 0.15           # відносна висота коридору відносно камери
    fx: float = 0.0               # 0 -> авто (0.8*W)
    fy: float = 0.0
    cx: float = 0.0               # 0 -> W/2
    cy: float = 0.0               # 0 -> H/2
    pitch_deg: float = 0.0        # нахил камери вниз, градуси
    hfov_deg: float = 0.0         # 0 -> fx=0.8*W; інакше fx з горизонтального FOV
    cam_dx: float = 0.0
    cam_dy: float = 0.0
    min_depth: float = 0.1
    # контролер та згладжування польоту
    lookahead: float = 0.25       # корекція за найближчими 1-2 точками (миттєва реакція)
    max_pursuit_points: int = 3   # тільки найближчі 2-3 точки
    stop_dist: float = 1.5        # зупинка, якщо dist до цілі (у кроках) менша
    v_max: float = 0.40
    w_max: float = 0.60
    ema_alpha: float = 0.85       # динамічна швидка реакція без запізнення
    w_deadband: float = 0.008     # точна реакція на маневри
    use_shield: bool = False      # тактичний аварійний щит Visual LiDAR / DA2
    shield_emergency_only: bool = True  # Тільки аварійний стоп (без втручання в рулювання та швидкість)
    shield_brake_dist_m: float = 0.45   # Критична дистанція аварійного стопу (метри)
    use_dronet_shield: bool = False     # Official DroNet v3 collision-probability shield
    dronet_variant: str = "v3"         # v3 | tiny-v3 (for the independent shield)
    dronet_weights: str = None         # optional path to the official checkpoint
    dronet_caution_thresh: float = 0.60
    dronet_stop_thresh: float = 0.80
    use_novelty_explorer: bool = False  # Дослідник тільки нових просторів (Topomap Anti-Looping)
    # VLFly / Перспективна ціль перед дроном
    use_vlfly: bool = False             # Режим VLFly з автоматичним семплуванням цілі перед дроном
    frontier_crop_ratio: float = 0.55   # Розмір перспективного вікна (55% кадру в точці сходу)
    frontier_jitter: float = 0.12       # Випадкове бокове відхилення для дослідження нових просторів
    frontier_auto_advance_dist: float = 2.2 # Дистанція перемикання на наступний перспективний кадр
    frontier_max_frames: int = 35       # Максимальний час утримання однієї підцілі (кроків)

    def __post_init__(self):
        if self.pure_nomad:
            if self.model != "nomad":
                raise ValueError("Pure NoMaD requires model='nomad'")
            if any((self.use_shield, self.use_dronet_shield,
                    self.use_novelty_explorer, self.use_vlfly)):
                raise ValueError("Pure NoMaD cannot be combined with shields, novelty or VLFly")
            if self.num_samples != 1 or self.num_diffusion_iters < 1:
                raise ValueError("Pure NoMaD requires one sample and positive diffusion iterations")
            self.use_obstacle_shield = False
            self.ema_alpha = 1.0
            self.w_deadband = 0.0
            self.max_pursuit_points = 0

    def intrinsics(self, w: int, h: int) -> np.ndarray:
        base = (w / 2) / np.tan(np.radians(self.hfov_deg) / 2) if self.hfov_deg else 0.8 * w
        fx = self.fx or base
        fy = self.fy or base
        cx = self.cx or w / 2
        cy = self.cy or h / 2
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
