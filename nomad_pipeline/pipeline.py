import time
import numpy as np
import cv2
import torch
from .config import Config
from .frame_buffer import FrameBuffer
from .controller import pure_pursuit, select_best_trajectory, FlightCommandFilter
from .visualize import draw_trajectory
from .obstacle_shield import ForbiddenZoneManager, MonocularObstacleShield


def build_policy(name: str, device=None, **kw):
    if device is None or str(device) == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if name in ("dronet-v3", "dronet-v3-tiny"):
        from .models.pulp_dronet import DroNetV3Policy
        return DroNetV3Policy(device, variant="tiny-v3" if name.endswith("-tiny") else "v3", **kw)
    if name == "nomad":
        from .models.nomad import NoMaDPolicy
        return NoMaDPolicy(device, **kw)
    from .models.gnm_vint import GNMPolicy, ViNTPolicy
    return {"gnm": GNMPolicy, "vint": ViNTPolicy}[name](device, **kw)


class NavPipeline:
    def __init__(self, cfg: Config, policy=None, shield=None):
        self.cfg = cfg
        if cfg.pure_nomad and shield is not None:
            raise ValueError("Pure NoMaD does not accept an injected shield")
        if policy is None:
            kw = {"num_samples": cfg.num_samples, "num_diffusion_iters": cfg.num_diffusion_iters} if cfg.model == "nomad" else {}
            if cfg.model in ("dronet-v3", "dronet-v3-tiny"):
                if cfg.use_vlfly:
                    raise ValueError("DroNet v3 is reactive; VLFly image goals are unsupported")
                kw = dict(weights_path=cfg.dronet_weights, len_traj_pred=cfg.len_traj_pred,
                          v_nominal=cfg.v_max, w_max=cfg.w_max,
                          caution_thresh=cfg.dronet_caution_thresh, stop_thresh=cfg.dronet_stop_thresh)
            policy = build_policy(cfg.model, device=cfg.device, **kw)
        self.policy = policy
        self.buffer = FrameBuffer(self.policy.context + 1, self.policy.img_size)
        self.cmd_filter = FlightCommandFilter(alpha=cfg.ema_alpha, w_deadband=cfg.w_deadband)
        self.shield = shield
        if self.shield is None and getattr(cfg, "use_dronet_shield", False):
            from .dronet_shield import DroNetShield
            self.shield = DroNetShield(
                device=self.policy.device, variant=cfg.dronet_variant,
                weights_path=cfg.dronet_weights, caution_thresh=cfg.dronet_caution_thresh,
                stop_thresh=cfg.dronet_stop_thresh,
            )
        elif self.shield is None and cfg.use_shield:
            from .shield import CollisionShield
            self.shield = CollisionShield(
                brake_dist_m=getattr(cfg, "shield_brake_dist_m", 0.45),
                emergency_only=getattr(cfg, "shield_emergency_only", True)
            )
        self.novelty_engine = None
        if getattr(cfg, "use_novelty_explorer", False):
            from .novelty_explorer import NoveltyMemoryEngine
            self.novelty_engine = NoveltyMemoryEngine(self.policy.device, min_node_dist=1.0)

        # Менеджер заборонених зон пілота (вікна, дзеркала, стіни)
        self.forbidden_mgr = None if cfg.pure_nomad else ForbiddenZoneManager()

        # Монокулярний щит розпізнавання стін та віконних відблисків
        self.obstacle_shield = MonocularObstacleShield() if cfg.use_obstacle_shield else None
        self.last_hazard_info = {"state": "DISABLED", "hazard": "DISABLED", "speed_scale": 1.0}

        # Менеджер перспективних візуальних цілей перед дроном (метод VLFly / ViNT)
        self.frontier_mgr = None
        if getattr(cfg, "use_vlfly", False) or cfg.model in ("vint", "gnm"):
            from .frontier_manager import FrontierSubgoalManager
            self.frontier_mgr = FrontierSubgoalManager(
                crop_ratio=getattr(cfg, "frontier_crop_ratio", 0.55),
                jitter=getattr(cfg, "frontier_jitter", 0.12),
                auto_advance_dist=getattr(cfg, "frontier_auto_advance_dist", 2.2),
                max_subgoal_frames=getattr(cfg, "frontier_max_frames", 35),
            )

        self.goal = None
        self._prev_wp = None
        self._t = time.time()
        self.fps = 0.0
        self.reached = False
        self.last_shield_info = None
        self.last_novelty_info = None
        self._last_step_dist = 0.0
        self._last_pred_dist = None

    def on_mouse(self, event, x, y, flags, param):
        """Обробник подій миші для малювання заборонених зон вікон/стін."""
        if self.forbidden_mgr is not None:
            self.forbidden_mgr.on_mouse(event, x, y, flags, param)

    def add_center_forbidden_zone(self, w_img: int, h_img: int):
        """Гаряча клавіша 'w': швидко позначити центр як заборонене вікно."""
        if self.forbidden_mgr is not None:
            self.forbidden_mgr.add_center_zone(w_img, h_img, span_pct=0.42, label="WINDOW / OBSTACLE")

    def clear_forbidden_zones(self):
        """Гаряча клавіша 'x': очистити всі мітки заборонених зон."""
        if self.forbidden_mgr is not None:
            self.forbidden_mgr.clear()

    def set_goal(self, bgr):
        if self.cfg.pure_nomad and bgr is not None:
            raise ValueError("Pure NoMaD uses goal=None (goal mask=1); image goals are disabled")
        self.goal = None if bgr is None else self.buffer.preprocess(bgr)

    def force_resample_frontier(self, bgr, reason: str = "MANUAL"):
        """Миттєво бере новий перспективний кадр перед дроном з урахуванням заборонених зон."""
        if self.frontier_mgr is not None:
            return self.frontier_mgr.force_resample(bgr, self.buffer, reason=reason, fzm=self.forbidden_mgr)
        return None

    def reset(self):
        self.buffer.frames.clear()
        self.cmd_filter.reset()
        self._prev_wp = None
        self._last_step_dist = 0.0
        self._last_pred_dist = None
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.last_shield_info = None
        if self.shield is not None:
            self.shield.reset()
        if self.novelty_engine is not None:
            self.novelty_engine.memory_embeddings.clear()
            self.novelty_engine.memory_thumbnails.clear()
            self.novelty_engine.memory_coords.clear()
        if self.frontier_mgr is not None:
            self.frontier_mgr.current_goal_tensor = None
            self.frontier_mgr.current_goal_bgr = None

    def step(self, bgr):
        """Повертає (overlay, waypoints[K,2] обраної траєкторії, (v,w), PolicyOutput)."""
        # DroNet consumes raw grayscale pixels with a center crop, whereas
        # ViNT/NoMaD consume the normalized RGB context buffer.
        predict_bgr = getattr(self.policy, "predict_bgr", None)
        if predict_bgr is None:
            self.buffer.push(bgr)

        # Отримуємо ціль: фіксовану вручну або динамічно перед дроном (VLFly Frontier з фільтрацією зон)
        active_goal = self.goal
        if active_goal is None and self.frontier_mgr is not None:
            active_goal = self.frontier_mgr.step(bgr, self.buffer, current_dist=self._last_pred_dist, fzm=self.forbidden_mgr)

        out = (predict_bgr(bgr, active_goal) if predict_bgr is not None
               else self.policy.predict(self.buffer.get(), active_goal))
        self.last_shield_info = out.shield_info
        wps = out.waypoints
        self._last_pred_dist = out.dist

        # Обираємо найкращу траєкторію за плавністю та узгодженістю з попереднім кроком
        if self.cfg.pure_nomad:
            samples = np.asarray(wps)
            if samples.ndim == 2:
                samples = samples[None]
            if len(samples) != 1:
                raise ValueError("Pure NoMaD expects exactly one trajectory")
        else:
            samples = select_best_trajectory(wps, self._prev_wp)
        best_wp = samples[0]
        self._prev_wp = best_wp.copy()

        K = self.cfg.intrinsics(bgr.shape[1], bgr.shape[0])

        # Перевірка монокулярного щита: чи є перед дроном стіна, закрите вікно або заборонена зона
        if self.obstacle_shield is not None:
            self.last_hazard_info = self.obstacle_shield.evaluate(
                bgr, fzm=self.forbidden_mgr, waypoints_xy=best_wp, K=K, cfg=self.cfg
            )

        v_raw, w_raw = pure_pursuit(
            best_wp, self.cfg.lookahead, self.cfg.v_max, self.cfg.w_max,
            max_points=getattr(self.cfg, "max_pursuit_points", 3)
        )
        if out.command is not None:
            v_raw, w_raw = out.command
            v_raw = float(np.clip(v_raw, 0.0, self.cfg.v_max))
            w_raw = float(np.clip(w_raw, -self.cfg.w_max, self.cfg.w_max))

        # Якщо виявлено небезпеку (вікно, стіна, заборонена мітка): ГЛУШИМО ТЯГУ ВПЕРЕД
        hazard_scale = self.last_hazard_info.get("speed_scale", 1.0)
        if self.frontier_mgr is not None and self.frontier_mgr.is_trapped and self.goal is None:
            hazard_scale = 0.0
            self.last_hazard_info = dict(
                state="EMERGENCY_BRAKE", hazard="TRAPPED", speed_scale=0.0,
                steer_bias=0.0, label="NO AVAILABLE FRONTIER",
            )
        if hazard_scale <= 0.0:
            v_raw = 0.0
            # Додаємо легкий відворот убік від небезпеки
            steer_bias = self.last_hazard_info.get("steer_bias", 0.0)
            w_raw = float(np.clip(steer_bias, -self.cfg.w_max, self.cfg.w_max))

        # Перевірка тактичного щита безпеки (якщо увімкнено)
        shield_scale = hazard_scale
        if out.shield_info is not None:
            shield_scale = min(shield_scale, out.shield_info["speed_scale"])
        if self.shield is not None:
            self.last_shield_info = self.shield.evaluate(bgr, forward_speed_mps=v_raw)
            shield_scale = min(shield_scale, self.last_shield_info["speed_scale"])
            if self.last_shield_info.get("state") == "EMERGENCY_BRAKE" and "steer" in self.last_shield_info:
                if hazard_scale > 0:
                    w_raw = float(np.clip(self.cfg.w_max * self.last_shield_info["steer"], -self.cfg.w_max, self.cfg.w_max))

        # EMA-згладжування та deadband
        vw = ((float(v_raw), float(w_raw)) if self.cfg.pure_nomad
              else self.cmd_filter.update(v_raw, w_raw, shield_scale=shield_scale))
        self._last_step_dist = float(vw[0] * 0.06)

        reached = out.dist is not None and out.dist < self.cfg.stop_dist
        if reached and self.frontier_mgr is None:
            vw = (0.0, 0.0)
        self.reached = reached

        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self._t, 1e-6) if self.fps else 1 / max(now - self._t, 1e-6)
        self._t = now
        img = bgr

        # Визначаємо мітку режиму навігації
        if self.cfg.pure_nomad:
            mode_label = "PURE NOMAD / GOAL MASK=1"
        elif self.goal is not None:
            mode_label = "GOAL"
        elif self.frontier_mgr is not None:
            mode_label = f"VLFLY-{self.frontier_mgr.chosen_sector} #{self.frontier_mgr.subgoal_id}"
        else:
            mode_label = "EXPLORE"

        # 1. Малюємо траєкторію польоту
        traj_color = (0, 255, 0) if hazard_scale > 0.0 else (0, 60, 255)
        for i, s in enumerate(samples):
            img = draw_trajectory(img, s, K, self.cfg,
                                  color=traj_color if i == 0 else (255, 160, 0),
                                  is_primary=(i == 0),
                                  vw=vw if i == 0 else None,
                                  fps=self.fps if i == 0 else None,
                                  mode_str=mode_label,
                                  dist=out.dist if i == 0 else None,
                                  shield_info=self.last_shield_info if i == 0 else None)

        # 2. Накладаємо користувацькі заборонені мітки (вікна, стіни)
        if self.forbidden_mgr is not None:
            img = self.forbidden_mgr.draw_hud_zones(img)

        # 3. Якщо попереду небезпека (стіна / вікно) — виводимо червоний аварійний банер
        if self.last_hazard_info.get("state") == "EMERGENCY_BRAKE":
            h_img, w_img = img.shape[:2]
            cv2.rectangle(img, (0, 70), (w_img, 100), (10, 10, 210), -1)
            hazard_txt = f"[{self.last_hazard_info.get('label', 'OBSTACLE DETECTED')}] -> FORWARD BLOCKED"
            cv2.putText(img, hazard_txt, (15, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        # 4. Накладаємо мініатюру цілі у кут кадру, якщо працює Frontier Manager
        if self.frontier_mgr is not None and self.goal is None:
            img = self.frontier_mgr.render_hud_thumbnail(img, dist=out.dist, fps=self.fps)

        return img, best_wp, vw, out
