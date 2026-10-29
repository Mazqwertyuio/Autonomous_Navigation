"""Модуль дослідження тільки нових просторів (Novelty-Driven Frontier Explorer).
Запобігає зацикленню, блуканню старими кімнатами та змушує дрон шукати нові території:
1. Записує топологічну пам'ять відвіданих місць (Visited Keyframes Buffer).
2. Обчислює коефіцієнт новизни сцени (Visual Novelty Score) через косинусну відстань ембедінгів.
3. Накладає штраф на повернення назад (Anti-Backtracking / Revisit Penalty).
4. Обирає дифузійні траєкторії NoMaD, які ведуть у раніше небачені коридори та кімнати.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF
import cv2
from PIL import Image as PILImage


class NoveltyMemoryEngine:
    def __init__(self, device: torch.device, min_node_dist: float = 1.2, max_memory_nodes: int = 200):
        self.device = device
        self.min_node_dist = min_node_dist
        self.max_memory_nodes = max_memory_nodes

        # База збережених вузлів пам'яті: список ембедінгів та мініатюр
        self.memory_embeddings = []
        self.memory_thumbnails = []
        self.memory_coords = []  # Приблизні відносні координати (dead-reckoning)

        self.current_pos = np.array([0.0, 0.0])
        self.total_distance = 0.0

    def compute_compact_feature(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Швидкий просторово-кольоровий вектор сцени (0.2 мс на CPU/GPU)."""
        small = cv2.resize(frame_bgr, (64, 48))
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        # Гістограми по каналах L, A, B для стійкості до мікротремтіння
        hist_l = cv2.calcHist([lab], [0], None, [16], [0, 256]).flatten()
        hist_a = cv2.calcHist([lab], [1], None, [16], [0, 256]).flatten()
        hist_b = cv2.calcHist([lab], [2], None, [16], [0, 256]).flatten()
        feat = np.concatenate([hist_l, hist_a, hist_b])
        norm = np.linalg.norm(feat)
        return feat / max(norm, 1e-6)

    def evaluate_novelty(self, frame_bgr: np.ndarray, current_step_dist: float = 0.0) -> dict:
        """Оцінює, наскільки поточна сцена нова порівняно з усією історією польоту."""
        feat = self.compute_compact_feature(frame_bgr)
        self.total_distance += current_step_dist

        if not self.memory_embeddings:
            # Перший кадр — точка відліку
            self.memory_embeddings.append(feat)
            thumb = cv2.resize(frame_bgr, (80, 60))
            self.memory_thumbnails.append(thumb)
            self.memory_coords.append(self.current_pos.copy())
            return {
                "novelty_score": 1.0,
                "is_new_area": True,
                "nodes_count": 1,
                "closest_node_idx": 0,
                "revisit_risk": 0.0,
            }

        # Порівнюємо поточний кадр з усіма збереженими вузлами в пам'яті
        mat = np.array(self.memory_embeddings)
        sims = np.dot(mat, feat)  # Косинусна схожість [-1, 1]
        max_sim = float(np.max(sims))
        closest_idx = int(np.argmax(sims))

        # Новизна: 1.0 - максимальна схожість
        novelty_score = float(np.clip(1.0 - max_sim, 0.0, 1.0))
        # Якщо схожість дуже висока (> 0.88), це вже знайоме старе місце!
        revisit_risk = float(np.clip((max_sim - 0.70) / 0.25, 0.0, 1.0))

        # Чи час зберегти новий вузол пам'яті (нова кімната або пройдено відстань)
        is_far_enough = current_step_dist > self.min_node_dist or len(self.memory_embeddings) < 3
        if novelty_score > 0.28 or is_far_enough:
            if len(self.memory_embeddings) < self.max_memory_nodes:
                self.memory_embeddings.append(feat)
                thumb = cv2.resize(frame_bgr, (80, 60))
                self.memory_thumbnails.append(thumb)
                self.memory_coords.append(self.current_pos.copy())

        return {
            "novelty_score": novelty_score,
            "is_new_area": novelty_score > 0.25,
            "nodes_count": len(self.memory_embeddings),
            "closest_node_idx": closest_idx,
            "revisit_risk": revisit_risk,
        }

    def bias_trajectories_for_novelty(self, samples_xy: np.ndarray, revisit_risk: float) -> np.ndarray:
        """Перевпорядковує дифузійні траєкторії NoMaD:
        якщо дрон починає розвертатися назад у вже відвідані місця (revisit_risk > 0.5),
        алгоритм штрафує повернення назад і віддає пріоритет траєкторіям у глибину вперед!
        """
        arr = np.asarray(samples_xy, dtype=np.float64)
        if len(arr) <= 1:
            return arr

        # 1. Просування вглиб (новий простір завжди попереду)
        fwd_scores = arr[:, -1, 0]

        # 2. Штраф за різкий розворот назад у старий простір (якщо x_end малий або y_end сильний)
        turnback_penalty = np.abs(arr[:, -1, 1]) * revisit_risk * 2.0

        total_scores = fwd_scores - turnback_penalty
        order = np.argsort(-total_scores)
        return arr[order]
