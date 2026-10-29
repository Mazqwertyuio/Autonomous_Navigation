"""VLFly Prospective Visual Frontier Sub-goal Manager:
Генерує динамічні візуальні цілі перед дроном для навігації моделлю ViNT (Visual Navigation Transformer).
Включає інтелектуальний відбір секторів за геометричними орієнтирами:
  - Оцінює 3 сектори (Центр, Ліворуч, Праворуч).
  - Надає пріоритет секторам із лініями підлоги, кутами стін, столами та дверима.
  - Ігнорує закриті вікна з сонячним відблиском (Window Glare).
  - Ігнорує сектори з втратою орієнтирів (однотонні стіни, score < 35).
  - Суворо оминає заборонені користувацькі мітки (Forbidden Zones).
"""

import cv2
import numpy as np
from .obstacle_shield import compute_geometric_anchors


class FrontierSubgoalManager:
    def __init__(self, crop_ratio: float = 0.55, jitter: float = 0.12,
                 auto_advance_dist: float = 2.2, max_subgoal_frames: int = 35):
        self.crop_ratio = crop_ratio
        self.jitter = jitter
        self.auto_advance_dist = auto_advance_dist
        self.max_subgoal_frames = max_subgoal_frames

        self.current_goal_bgr = None
        self.current_goal_tensor = None
        self.frames_on_subgoal = 0
        self.subgoal_id = 0
        self.last_advance_reason = "INIT"
        self.chosen_sector = "CENTER"
        self.is_trapped = False
        self.last_anchor_score = 0.0

    def sample_new_goal(self, frame_bgr: np.ndarray, buffer, random_jitter: bool = True,
                        custom_crop: np.ndarray = None, reason: str = "AUTO", fzm = None):
        """Оцінює сектори та фіксує безпечний перспективний кадр перед дроном."""
        if custom_crop is not None:
            self.current_goal_bgr = custom_crop.copy()
            self.chosen_sector = "CUSTOM"
            self.is_trapped = False
        else:
            H, W = frame_bgr.shape[:2]
            ch = int(H * self.crop_ratio)
            cw = int(W * self.crop_ratio)
            cy = H // 2

            # Кандидати для вибору напрямку руху: Центр, Ліворуч, Праворуч
            candidates = [
                ("CENTER", W // 2),
                ("LEFT", int(W * 0.30)),
                ("RIGHT", int(W * 0.70)),
            ]

            scored_candidates = []
            for name, base_cx in candidates:
                cx = base_cx
                if random_jitter and self.jitter > 0 and name == "CENTER":
                    ox = int(np.random.uniform(-self.jitter, self.jitter) * W)
                    cx = int(np.clip(cx + ox, cw // 2, W - cw // 2))

                x1 = int(np.clip(cx - cw // 2, 0, W - cw))
                x2 = x1 + cw
                y1 = int(np.clip(cy - ch // 2, 0, H - ch))
                y2 = y1 + ch

                # 1. Перевірка на перетин із користувацькими мітками (вікна / стіни)
                is_forbidden = False
                if fzm is not None and fzm.overlaps_box(x1, y1, x2, y2):
                    is_forbidden = True

                crop = frame_bgr[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                # Оцінка геометричних орієнтирів (лінія підлоги, кути, столи)
                score_anchors, n_lines, canny_pct, floor_pct = compute_geometric_anchors(crop)

                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                glare_pct = float(np.mean(hsv[:, :, 2] > 230)) * 100.0

                if is_forbidden:
                    score = -9999.0
                    status = "FORBIDDEN_ZONE"
                elif score_anchors < 35.0 or lap_var < 18.0:
                    score = -5000.0 + score_anchors
                    status = "NO_ANCHORS_WALL"
                elif glare_pct > 8.0:
                    score = -4000.0
                    status = "WINDOW_GLARE"
                else:
                    bias = 6.0 if name == "CENTER" else 0.0
                    score = (score_anchors * 0.35) + np.log(max(lap_var, 1.0)) * 8.0 + bias
                    status = "CLEAR"

                scored_candidates.append((name, score, status, crop, (x1, y1, x2, y2), score_anchors))

            scored_candidates.sort(key=lambda item: item[1], reverse=True)
            best_cand = scored_candidates[0]
            self.chosen_sector = best_cand[0]
            self.current_goal_bgr = best_cand[3].copy()
            self.last_anchor_score = best_cand[5]
            self.is_trapped = (best_cand[1] < 0.0)

        self.current_goal_tensor = buffer.preprocess(self.current_goal_bgr)
        self.frames_on_subgoal = 0
        self.subgoal_id += 1
        self.last_advance_reason = reason
        return self.current_goal_tensor

    def step(self, frame_bgr: np.ndarray, buffer, current_dist: float = None, fzm = None):
        """Оновлює стан підцілі: просуває горизонт при досягненні дистанції або таймауті."""
        if self.current_goal_tensor is None:
            return self.sample_new_goal(frame_bgr, buffer, reason="START", fzm=fzm)

        self.frames_on_subgoal += 1
        need_advance = False
        reason = ""

        if current_dist is not None and current_dist <= self.auto_advance_dist:
            need_advance = True
            reason = f"REACHED (d={current_dist:.1f}m)"
        elif self.frames_on_subgoal >= self.max_subgoal_frames:
            need_advance = True
            reason = f"TTL_EXPIRED ({self.max_subgoal_frames}f)"

        if need_advance:
            self.sample_new_goal(frame_bgr, buffer, random_jitter=True, reason=reason, fzm=fzm)

        return self.current_goal_tensor

    def force_resample(self, frame_bgr: np.ndarray, buffer, reason: str = "UNSTICK_RESAMPLE", fzm = None):
        """Примусово оновлює ціль (наприклад, після повороту від стіни або за командою пілота)."""
        return self.sample_new_goal(frame_bgr, buffer, random_jitter=True, reason=reason, fzm=fzm)

    def set_manual_frame(self, frame_bgr: np.ndarray, buffer):
        """Встановлює повний поточний кадр як фіксовану ціль."""
        return self.sample_new_goal(frame_bgr, buffer, random_jitter=False, custom_crop=frame_bgr, reason="MANUAL_LOCK")

    def render_hud_thumbnail(self, vis: np.ndarray, dist: float = None, fps: float = 10.0) -> np.ndarray:
        """Накладає мініатюру поточної цілі перед дроном у кут екрана для польотного запису."""
        if self.current_goal_bgr is None:
            return vis

        h_img, w_img = vis.shape[:2]
        thumb_w, thumb_h = 140, 105
        pad = 12
        x1 = w_img - thumb_w - pad
        y1 = 76
        x2 = x1 + thumb_w
        y2 = y1 + thumb_h

        if x1 >= 0 and y2 < h_img:
            thumb = cv2.resize(self.current_goal_bgr, (thumb_w, thumb_h))
            vis[y1:y2, x1:x2] = thumb
            border_col = (0, 230, 255) if self.chosen_sector == "CENTER" else (0, 165, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), border_col, 2)

            cv2.rectangle(vis, (x1, y2 - 20), (x2, y2), (20, 24, 28), -1)
            dist_str = f"d={dist:.1f}m" if dist is not None else "--"
            label = f"#{self.subgoal_id} [{self.chosen_sector}] {dist_str}"
            cv2.putText(vis, label, (x1 + 4, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 230, 255), 1, cv2.LINE_AA)

            cv2.rectangle(vis, (x1, y1 - 18), (x2, y1), (20, 24, 28), -1)
            header = "FRONTIER: TRAPPED" if self.is_trapped else f"VLFly {self.chosen_sector}"
            head_col = (40, 40, 255) if self.is_trapped else (0, 255, 120)
            cv2.putText(vis, header, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, head_col, 1, cv2.LINE_AA)

        return vis
