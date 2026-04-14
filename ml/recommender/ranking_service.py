"""Learning-to-rank service with LightGBM optional backend and fallback scoring."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

try:
    from lightgbm import LGBMRanker
except ImportError:  # pragma: no cover
    LGBMRanker = None


class RankingService:
    def __init__(self) -> None:
        self.model = None

    def fit(self, x: np.ndarray, y: np.ndarray, group: List[int]) -> None:
        if LGBMRanker is None:
            return
        ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
        )
        ranker.fit(x, y, group=group)
        self.model = ranker

    @staticmethod
    def _feature_vector(candidate: str, rank: int, user_feats: Dict[str, float], item_feats: Dict[str, float]) -> List[float]:
        similarity = float(item_feats.get("embedding_similarity", 0.5))
        recency = float(item_feats.get("recency_hours", 72.0))
        popularity = float(item_feats.get("popularity", 0.5))
        completion_propensity = float(user_feats.get("completion_rate", 0.3))
        ctr = float(user_feats.get("ctr", 0.1))
        freshness = float(item_feats.get("freshness", 0.5))
        difficulty_match = float(item_feats.get("difficulty_match", 0.5))
        diversity = 1.0 / max(len(set(candidate.split(" "))), 1)
        return [
            similarity,
            1.0 / (1.0 + math.log1p(recency)),
            popularity,
            completion_propensity,
            ctr,
            freshness,
            difficulty_match,
            diversity,
            1.0 / max(rank, 1),
        ]

    @staticmethod
    def _fallback_score(vec: List[float]) -> float:
        return (
            0.28 * vec[0]
            + 0.18 * vec[1]
            + 0.14 * vec[2]
            + 0.14 * vec[3]
            + 0.08 * vec[4]
            + 0.08 * vec[5]
            + 0.06 * vec[6]
            + 0.04 * vec[7]
            + 0.02 * vec[8]
        )

    def rank_candidates(
        self,
        candidates: List[str],
        user_features: Dict[str, float],
        item_feature_map: Dict[str, Dict[str, float]],
        top_k: int,
    ) -> List[str]:
        if not candidates:
            return []

        features = [
            self._feature_vector(c, i + 1, user_features, item_feature_map.get(c, {}))
            for i, c in enumerate(candidates)
        ]
        if self.model is not None:
            scores = self.model.predict(np.asarray(features, dtype=float))
            order = np.argsort(-scores)
        else:
            scores = np.asarray([self._fallback_score(v) for v in features])
            order = np.argsort(-scores)

        ranked = [candidates[int(i)] for i in order]
        return ranked[:top_k]
