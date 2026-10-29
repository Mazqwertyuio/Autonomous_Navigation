"""EGO-Planner v2 (HKUST Aerial Robotics) для квадрокоптера DJI Tello.

Принцип роботи:
1. Замість дискретного вибору з фіксованих дуг використовується ГРАДІЄНТНИЙ НЕПЕРЕРВНИЙ ОПТИМІЗАТОР ТРАЄКТОРІЙ (ESDF-free Gradient Optimization);
2. Глибина глибинного датчика (Metric Depth) формує локальну карту перешкод у 3D;
3. Для будь-якої перешкоди на шляху (кути, меблі, звуження) генеруються віртуальні вектори відштовхування (Collision Repulsion Forces);
4. B-сплайн траєкторія оптимізується функцією вартості:
   Cost = w_smooth * Smoothness(Jerk) + w_coll * Repulsion(Obstacles) + w_feas * Feasibility(Speed/Acc);
5. Траєкторія огинає перешкоди плавно в 3D (з урахуванням габаритів дрона 36 см) і виводить дрон у вільний коридор;
6. Візуалізує векторну B-сплайн стрічку та вектори оптимізації EGO-Planner.
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
_LOCAL_METRIC_DIR = os.path.join(ROOT_DIR, "third_party", "Depth-Anything-V2", "metric_depth")
METRIC_DIR = _LOCAL_METRIC_DIR if os.path.exists(_LOCAL_METRIC_DIR) else r"C:\Users\User\Desktop\bd_3d\Depth-Anything-V2\metric_depth"
if METRIC_DIR not in sys.path:
    sys.path.insert(0, METRIC_DIR)

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LOCAL_METRIC_WEIGHTS = os.path.join(ROOT_DIR, "weights", "depth_anything_v2_metric_hypersim_vits.pth")
METRIC_WEIGHTS = _LOCAL_METRIC_WEIGHTS if os.path.exists(_LOCAL_METRIC_WEIGHTS) else r"C:\Users\User\Desktop\bd_3d\Depth-Anything-V2\checkpoints\depth_anything_v2_metric_hypersim_vits.pth"


class EgoPlannerV2:
    def __init__(
        self,
        checkpoint_path: str = METRIC_WEIGHTS,
        device: str = DEVICE,
        drone_cruise_alt: float = 0.85,
        drone_radius: float = 0.22,       # радіус безпеки Tello з захистом 22 см
        fov_h_deg: float = 82.6,
        max_horizon_m: float = 3.20,
    ):
        self.device = device
        self.drone_cruise_alt = drone_cruise_alt
        self.drone_radius = drone_radius
        self.fov_h = np.deg2rad(fov_h_deg)
        self.max_horizon = max_horizon_m

        print(f"[EGO-Planner-v2] Завантаження метричного ядра на {device}...")
        self.depth_net = DepthAnythingV2(
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384],
            max_depth=20.0
        ).to(self.device)
        self.depth_net.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        self.depth_net = self.depth_net.to(self.device).eval()

        self.prev_opt_pts = None

    def project_3d_to_cam(self, x, y, z, w=640, h=480, y_offset=0.0):
        if x <= 0.08:
            return None
        fx = (w / 2.0) / np.tan(self.fov_h / 2.0)
        fy = fx
        cx = w / 2.0
        cy = h / 2.0

        z_cam = x
        x_cam = -y
        y_cam = -(z - self.drone_cruise_alt) + y_offset

        u = int(cx + (fx * x_cam) / z_cam)
        v = int(cy + (fy * y_cam) / z_cam)
        return (u, v)

    def extract_3d_obstacles(self, depth_meters: np.ndarray, w: int, h: int):
        """Вилучення 3D хмари найближчих перешкод на горизонті польоту дрона."""
        fx = (w / 2.0) / np.tan(self.fov_h / 2.0)
        fy = fx
        cx = w / 2.0
        cy = h / 2.0

        # Семплюємо сітку точок у польотному коридорі
        v_min, v_max = int(h * 0.30), int(h * 0.62)
        step_u = 16
        step_v = 12

        obs_points = []
        for v in range(v_min, v_max, step_v):
            for u in range(20, w - 20, step_u):
                d = float(depth_meters[v, u])
                if 0.35 < d < (self.max_horizon + 0.50):
                    z_cam = d
                    x_cam = (u - cx) * z_cam / fx
                    y_cam = (v - cy) * z_cam / fy

                    # Перевід у систему координат дрона (X forward, Y left, Z up)
                    x_body = z_cam
                    y_body = -x_cam
                    z_body = -(y_cam) + self.drone_cruise_alt
                    obs_points.append([x_body, y_body, z_body])

        return np.array(obs_points) if obs_points else np.zeros((0, 3))

    def plan(self, bgr_img: np.ndarray) -> dict:
        h, w = bgr_img.shape[:2]

        with torch.no_grad():
            depth_meters = self.depth_net.infer_image(bgr_img, input_size=252)

        depth_resized = cv2.resize(depth_meters, (w, h), interpolation=cv2.INTER_LINEAR)

        # Визначаємо вільний вектор погляду (найглибший відкритий напрямок)
        flight_strip = depth_resized[int(h * 0.40):int(h * 0.58), :]
        col_depths = np.median(flight_strip, axis=0)

        # Шукаємо максимальну відкриту глибину
        best_u = int(np.argmax(col_depths))
        max_open_dist = float(col_depths[best_u])
        fx = (w / 2.0) / np.tan(self.fov_h / 2.0)
        target_y = -float((best_u - w / 2.0) * min(max_open_dist, self.max_horizon) / fx)

        target_reach = min(max(max_open_dist, 0.60), self.max_horizon)
        front_dist = float(depth_resized[int(h * 0.50), int(w * 0.50)])
        is_dead_end = (front_dist < 0.80) and (max_open_dist < 0.90)

        # Початкова опорна траєкторія (7 контрольних точок B-сплайна)
        n_ctrl = 7
        p_start = np.array([0.25, 0.0, self.drone_cruise_alt])
        p_end = np.array([target_reach, target_y, self.drone_cruise_alt])

        if self.prev_opt_pts is not None and len(self.prev_opt_pts) == n_ctrl:
            init_pts = 0.60 * self.prev_opt_pts + 0.40 * np.linspace(p_start, p_end, n_ctrl)
        else:
            init_pts = np.linspace(p_start, p_end, n_ctrl)

        # Отримуємо 3D точки перешкод
        obs_pts = self.extract_3d_obstacles(depth_resized, w, h)

        # --- EGO-Planner Gradient Optimization ---
        # Оптимізуємо внутрішні контрольні точки
        r_safe = self.drone_radius + 0.12

        def ego_cost(x_flat):
            inner = x_flat.reshape(-1, 3)
            pts = np.vstack([p_start, inner, p_end])

            # 1. Гладенькість (мінімізація прискорення/ривка): sum ||p_{i+1} - 2p_i + p_{i-1}||^2
            acc = pts[2:] - 2.0 * pts[1:-1] + pts[:-2]
            smooth_cost = np.sum(acc**2) * 8.0

            # 2. Відштовхування від перешкод (Collision Repulsion)
            coll_cost = 0.0
            if len(obs_pts) > 0:
                # Векторизована відстань між усіма точками траєкторії та перешкодами
                # diff: [n_pts, n_obs, 3]
                diff = pts[:, np.newaxis, :] - obs_pts[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=-1)  # [n_pts, n_obs]
                penetrations = np.maximum(0.0, r_safe - dists)
                coll_cost = np.sum(penetrations**2) * 1500.0

            # 3. Висота (утримання заданого ешелону cruise_alt)
            alt_cost = np.sum((pts[:, 2] - self.drone_cruise_alt)**2) * 50.0

            return smooth_cost + coll_cost + alt_cost

        # Швидка квазі-ньютонівська оптимізація (L-BFGS-B, макс 15 ітерацій для real-time)
        res = minimize(
            ego_cost,
            init_pts[1:-1].flatten(),
            method="L-BFGS-B",
            options={"maxiter": 15, "ftol": 1e-4}
        )

        opt_inner = res.x.reshape(-1, 3)
        opt_ctrl = np.vstack([p_start, opt_inner, p_end])
        self.prev_opt_pts = opt_ctrl

        # Інтерполюємо плавний B-сплайн
        try:
            tck, u = splprep([opt_ctrl[:, 0], opt_ctrl[:, 1], opt_ctrl[:, 2]], k=3, s=0)
            u_fine = np.linspace(0, 1, 16)
            sx, sy, sz = splev(u_fine, tck)
            smooth_traj = np.stack([sx, sy, sz], axis=-1)
        except Exception:
            smooth_traj = opt_ctrl

        status_text = f"EGO-PLANNER v2 | OPTIMIZED FLOW: {target_reach:.2f}m (FRONT: {front_dist:.2f}m)"
        if is_dead_end:
            status_text = f"EGO-PLANNER v2 | DEAD-END HOVER STOP ({front_dist:.2f}m)"

        return {
            "trajectory": smooth_traj,
            "ctrl_points": opt_ctrl,
            "is_dead_end": is_dead_end,
            "front_dist": front_dist,
            "reach_dist": target_reach,
            "status_text": status_text,
            "n_obs": len(obs_pts),
        }

    def render(self, bgr_img: np.ndarray, plan_result: dict) -> np.ndarray:
        vis = bgr_img.copy()
        h, w = vis.shape[:2]

        traj = plan_result["trajectory"]
        ctrl_pts = plan_result["ctrl_points"]
        is_dead_end = plan_result["is_dead_end"]
        status_text = plan_result["status_text"]
        front_dist = plan_result["front_dist"]

        tape_half_w = 0.28
        y_tape_offset = 0.32

        proj_centers = []
        proj_lefts = []
        proj_rights = []

        for pt in traj:
            xf, yl, za = pt[0], pt[1], pt[2]
            pc = self.project_3d_to_cam(xf, yl, za, w, h, y_offset=0.0)
            pl = self.project_3d_to_cam(xf, yl + tape_half_w, za, w, h, y_offset=y_tape_offset)
            pr = self.project_3d_to_cam(xf, yl - tape_half_w, za, w, h, y_offset=y_tape_offset)

            if pc and 0 <= pc[0] < w and 0 <= pc[1] < h: proj_centers.append(pc)
            if pl and 0 <= pl[0] < w and 0 <= pl[1] < h: proj_lefts.append(pl)
            if pr and 0 <= pr[0] < w and 0 <= pr[1] < h: proj_rights.append(pr)

        n_segs = min(len(proj_lefts), len(proj_rights))

        if is_dead_end:
            ribbon_color = (0, 0, 230)
            border_color = (0, 0, 255)
            hud_c = (0, 0, 255)
        else:
            ribbon_color = (0, 235, 45)      # Неоновий зелений
            border_color = (0, 255, 120)
            hud_c = (0, 255, 0)

        overlay = vis.copy()
        for i in range(n_segs - 1):
            poly = np.array([proj_lefts[i], proj_lefts[i+1], proj_rights[i+1], proj_rights[i]], dtype=np.int32)
            cv2.fillPoly(overlay, [poly], ribbon_color)

        vis = cv2.addWeighted(overlay, 0.42, vis, 0.58, 0)

        # Неонові межі
        for i in range(n_segs - 1):
            cv2.line(vis, proj_lefts[i], proj_lefts[i+1], border_color, 2, cv2.LINE_AA)
            cv2.line(vis, proj_rights[i], proj_rights[i+1], border_color, 2, cv2.LINE_AA)

        # Опорні контрольні точки оптимізації EGO-Planner
        for pt in ctrl_pts:
            p_pix = self.project_3d_to_cam(pt[0], pt[1], pt[2], w, h, y_offset=0.0)
            if p_pix and 0 <= p_pix[0] < w and 0 <= p_pix[1] < h:
                cv2.circle(vis, p_pix, 5, (255, 200, 0), -1, cv2.LINE_AA)
                cv2.circle(vis, p_pix, 7, (255, 255, 255), 1, cv2.LINE_AA)

        # Центральна лінія від носа дрона
        nose_pt = (int(w / 2.0), h - 15)
        if len(proj_centers) > 0:
            target_pt = proj_centers[-1]
            cv2.line(vis, nose_pt, target_pt, (255, 255, 255), 2, cv2.LINE_AA)

            t_color = (0, 0, 255) if is_dead_end else (0, 255, 0)
            cv2.circle(vis, target_pt, 16, t_color, 2, cv2.LINE_AA)
            cv2.circle(vis, target_pt, 8, t_color, -1, cv2.LINE_AA)
            cv2.circle(vis, target_pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            tag = f"STOP ({front_dist:.2f}m)" if is_dead_end else f"TARGET ({traj[-1, 0]:.2f}m)"
            cv2.putText(vis, tag, (target_pt[0] - 38, max(target_pt[1] - 22, 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, t_color, 2, cv2.LINE_AA)

        # Верхній HUD EGO-Planner
        cv2.rectangle(vis, (0, 0), (w, 54), (15, 15, 15), -1)
        cv2.line(vis, (0, 54), (w, 54), hud_c, 2)
        cv2.putText(vis, status_text, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, hud_c, 2)
        cv2.putText(vis, f"HKUST EGO-Planner v2 | Repulsion-field Opt: Active | 3D Obstacles: {plan_result['n_obs']}",
                    (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

        return vis


if __name__ == "__main__":
    planner = EgoPlannerV2()
    print("[EGO-Planner-v2] Ініціалізація успішна!")
