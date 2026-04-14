"""Online learning hooks for incremental user embedding updates."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict

import joblib
import numpy as np


class OnlineLearner:
    def __init__(self, vector_dim: int = 32) -> None:
        self.vector_dim = vector_dim
        self.user_vectors: Dict[str, np.ndarray] = {}
        self.user_history: Dict[str, list[str]] = defaultdict(list)

    def update_user_vector(self, user_id: str, item_vector: np.ndarray, reward: float, lr: float = 0.05) -> np.ndarray:
        if user_id not in self.user_vectors:
            self.user_vectors[user_id] = np.zeros(self.vector_dim, dtype=float)
        current = self.user_vectors[user_id]
        delta = lr * float(reward) * item_vector
        updated = current + delta
        norm = np.linalg.norm(updated)
        if norm > 0:
            updated = updated / norm
        self.user_vectors[user_id] = updated
        return updated

    def observe_feedback(self, user_id: str, item_id: str | None, reward: float, item_vector: np.ndarray | None = None) -> np.ndarray:
        if item_id:
            history = self.user_history[user_id]
            if not history or history[-1] != item_id:
                history.append(item_id)
                self.user_history[user_id] = history[-50:]
        if item_vector is None:
            item_vector = np.zeros(self.vector_dim, dtype=float)
            if item_id is not None:
                item_vector[0] = 1.0
        if item_vector.size != self.vector_dim:
            item_vector = np.resize(item_vector, self.vector_dim)
        return self.update_user_vector(user_id, item_vector, reward)

    def score(self, user_id: str, item_vector: np.ndarray) -> float:
        user_vector = self.user_vectors.get(user_id)
        if user_vector is None:
            return 0.0
        if item_vector.size != self.vector_dim:
            item_vector = np.resize(item_vector, self.vector_dim)
        return float(np.dot(user_vector, item_vector))

    def checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"vector_dim": self.vector_dim, "user_vectors": self.user_vectors, "user_history": dict(self.user_history)}, path)

    def load_checkpoint(self, path: str) -> None:
        if not os.path.exists(path):
            return
        payload = joblib.load(path)
        self.vector_dim = int(payload.get("vector_dim", self.vector_dim))
        self.user_vectors = payload.get("user_vectors", {})
        self.user_history = defaultdict(list, payload.get("user_history", {}))
