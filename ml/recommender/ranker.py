"""Candidate ranking with optional LightGBM ranker and fallback scorer."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    from lightgbm import LGBMRanker
except ImportError:  # pragma: no cover
    LGBMRanker = None


class RecommendationRanker:
    def __init__(self) -> None:
        self.model = None

    def fit(self, features: np.ndarray, labels: np.ndarray, group: List[int]) -> None:
        if LGBMRanker is None:
            return
        ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            learning_rate=0.05,
            n_estimators=120,
            num_leaves=31,
            min_data_in_leaf=10,
            random_state=42,
        )
        ranker.fit(features, labels, group=group)
        self.model = ranker

    @staticmethod
    def _fallback_score(topic: str, content_rank: int, collab_rank: int, quiz_score: float) -> float:
        content_component = 1.0 / np.log2(max(content_rank, 1) + 1.0)
        collab_component = 1.0 / np.log2(max(collab_rank, 1) + 1.0)
        is_foundation = "Basics" in topic or "Intro" in topic
        is_extension = any(token in topic for token in ["Advanced", "Applications", "Analysis", "Multivariable", "Regression"])
        readiness = min(max(quiz_score / 100.0, 0.0), 1.0)
        pedagogy_bonus = 0.08 if is_foundation and readiness < 0.7 else 0.08 if is_extension and readiness >= 0.7 else 0.0
        return 0.52 * content_component + 0.38 * collab_component + 0.10 * readiness + pedagogy_bonus

    def rank(
        self,
        candidates: Dict[str, List[str]],
        quiz_score: float,
        top_k: int = 5,
    ) -> List[str]:
        content = candidates.get("content", [])
        collaborative = candidates.get("collaborative", [])
        merged = list(dict.fromkeys(content + collaborative))
        if not merged:
            return []

        if self.model is not None:
            features = []
            for topic in merged:
                c_rank = content.index(topic) + 1 if topic in content else len(content) + 5
                u_rank = collaborative.index(topic) + 1 if topic in collaborative else len(collaborative) + 5
                features.append([c_rank, u_rank, quiz_score])
            preds = self.model.predict(np.array(features))
            order = np.argsort(-preds)
            ranked = [merged[i] for i in order]
        else:
            scored = []
            for topic in merged:
                c_rank = content.index(topic) + 1 if topic in content else len(content) + 5
                u_rank = collaborative.index(topic) + 1 if topic in collaborative else len(collaborative) + 5
                scored.append((topic, self._fallback_score(topic, c_rank, u_rank, quiz_score)))
            scored.sort(key=lambda x: x[1], reverse=True)
            ranked = [t for t, _ in scored]

        return rerank_diversity_novelty(ranked, content, top_k)


def rerank_diversity_novelty(ranked_topics: List[str], content_topics: List[str], top_k: int) -> List[str]:
    if not ranked_topics:
        return []
    selected: List[str] = []
    seen_prefixes = set()

    for topic in ranked_topics:
        prefix = topic.split(" ")[0]
        if prefix not in seen_prefixes or len(selected) < max(2, top_k // 3):
            selected.append(topic)
            seen_prefixes.add(prefix)
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for t in ranked_topics:
            if t not in selected:
                selected.append(t)
            if len(selected) >= top_k:
                break

    if content_topics:
        anchor = content_topics[0]
        if anchor not in selected:
            selected.insert(0, anchor)
            selected = selected[:top_k]

    return selected[:top_k]
