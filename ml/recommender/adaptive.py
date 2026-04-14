"""Real-time adaptive recommendation orchestration."""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Dict, List, Sequence

import numpy as np

from api.core.feature_store import FeatureStore
from ml.inference.adaptive_learning import ContextualBanditPolicy, SequenceTransformerRecommender, event_reward, load_adaptive_artifacts
from ml.inference.online_learning import OnlineLearner


def _hash_vector(value: str, dim: int = 9) -> np.ndarray:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    raw = np.frombuffer(digest, dtype=np.uint8).astype(float)
    if raw.size < dim:
        raw = np.pad(raw, (0, dim - raw.size), constant_values=0.0)
    raw = raw[:dim]
    return (raw / 255.0).astype(float)


class AdaptiveRecommender:
    def __init__(self, model_dir: str, feature_store: FeatureStore | None = None) -> None:
        self.model_dir = model_dir
        self.feature_store = feature_store or FeatureStore()
        self.bandit, self.sequence_model = self._load_models()
        self.online_learner = OnlineLearner(vector_dim=9)
        self._lock = threading.Lock()

    def _load_models(self) -> tuple[ContextualBanditPolicy, SequenceTransformerRecommender]:
        try:
            return load_adaptive_artifacts(self.model_dir)
        except Exception:
            return ContextualBanditPolicy(), SequenceTransformerRecommender()

    def _item_context(self, candidate: str, user_features: Dict[str, float], item_feature_map: Dict[str, Dict[str, float]], sequence_score: float = 0.0, online_score: float = 0.0) -> Dict[str, float]:
        item_features = item_feature_map.get(candidate, {})
        return {
            "quiz_score": float(user_features.get("quiz_score", 50.0)),
            "engagement_score": float(user_features.get("engagement_score", 50.0)),
            "consistency_score": float(user_features.get("consistency_score", 50.0)),
            "freshness": float(item_features.get("freshness", 0.5)),
            "difficulty_match": float(item_features.get("difficulty_match", 0.5)),
            "popularity": float(item_features.get("popularity", 0.3)),
            "sequence_score": float(sequence_score),
            "online_score": float(online_score),
        }

    def _online_score(self, user_id: str, candidate: str, item_feature_map: Dict[str, Dict[str, float]]) -> float:
        user_vec = self.online_learner.user_vectors.get(user_id)
        if user_vec is None:
            return 0.0
        item_vec = _hash_vector(candidate, dim=self.online_learner.vector_dim)
        features = item_feature_map.get(candidate, {})
        item_vec = 0.7 * item_vec + 0.3 * np.array(
            [
                float(features.get("embedding_similarity", 0.5)),
                float(features.get("freshness", 0.5)),
                float(features.get("popularity", 0.5)),
                float(features.get("difficulty_match", 0.5)),
                float(features.get("recency_hours", 72.0)) / 100.0,
                float(features.get("item_ctr", 0.3)),
                1.0,
                0.5,
                0.5,
            ],
            dtype=float,
        )
        if item_vec.size != user_vec.size:
            item_vec = np.resize(item_vec, user_vec.size)
        return float(np.dot(user_vec, item_vec))

    def rank(
        self,
        user_id: str,
        candidates: Sequence[str],
        user_features: Dict[str, float],
        item_feature_map: Dict[str, Dict[str, float]],
        recent_history: Sequence[str] | None = None,
        top_k: int = 5,
    ) -> Dict[str, object]:
        history = list(recent_history or [])
        sequence_scores = self.sequence_model.score_candidates(history, candidates)

        candidate_contexts = {}
        online_scores = {}
        for candidate in candidates:
            online_score = self._online_score(user_id, candidate, item_feature_map)
            online_scores[candidate] = online_score
            candidate_contexts[candidate] = self._item_context(candidate, user_features, item_feature_map, sequence_scores.get(candidate, 0.0), online_score)

        bandit_scores = self.bandit.score_candidates(candidate_contexts)
        blended: Dict[str, float] = {}
        for index, candidate in enumerate(candidates):
            base_rank = 1.0 / float(index + 1)
            blended[candidate] = (
                0.25 * base_rank
                + 0.30 * float(bandit_scores.get(candidate, 0.0))
                + 0.25 * float(sequence_scores.get(candidate, 0.0))
                + 0.20 * float(online_scores.get(candidate, 0.0))
            )

        ordered = sorted(blended, key=blended.get, reverse=True)
        return {
            "recommended_topics": ordered[:top_k],
            "bandit_scores": bandit_scores,
            "sequence_scores": sequence_scores,
            "online_scores": online_scores,
            "source": "adaptive_bandit_transformer",
        }

    def observe_event(self, event) -> None:
        reward = event_reward(str(event.event_type), event.payload)
        with self._lock:
            if event.item_id:
                item_id = str(event.item_id)
                user_features = self.feature_store.get_online_user_features(event.user_id)
                item_features = self.feature_store.get_online_item_features(item_id)
                context = {
                    "quiz_score": float(user_features.get("latest_quiz_score", user_features.get("quiz_score", 50.0))),
                    "engagement_score": float(user_features.get("engagement_score", user_features.get("ctr", 50.0))),
                    "consistency_score": float(user_features.get("completion_rate", 0.5) * 100.0),
                    "freshness": float(item_features.get("freshness", 0.5)),
                    "difficulty_match": float(item_features.get("difficulty_match", 0.5)),
                    "popularity": float(item_features.get("item_ctr", item_features.get("popularity", 0.3))),
                    "sequence_score": 0.0,
                    "online_score": 0.0,
                }
                self.bandit.update(item_id, context, reward)

                item_vector = _hash_vector(item_id, dim=self.online_learner.vector_dim)
                self.online_learner.observe_feedback(event.user_id, item_id, reward, item_vector=item_vector)

            history = self.feature_store.get_recent_user_sequence(event.user_id, limit=24)
            if event.item_id and history and history[-1] == str(event.item_id):
                prior_history = history[:-1]
            else:
                prior_history = list(history)
            if prior_history and event.item_id:
                self.sequence_model.bigram_counts[prior_history[-1]][str(event.item_id)] += 1
            self.feature_store.write_online_features(
                entity=f"user:{event.user_id}",
                values={
                    "recent_sequence": history[-24:],
                    "last_reward": reward,
                    "last_event_type": event.event_type,
                },
                ttl=7200,
            )

    def save(self) -> None:
        with self._lock:
            bandit_path = os.path.join(self.model_dir, "adaptive_bandit.joblib")
            sequence_path = os.path.join(self.model_dir, "adaptive_sequence.pt")
            self.bandit.save(bandit_path)
            self.sequence_model.save(sequence_path)
            self.online_learner.checkpoint(os.path.join(self.model_dir, "online_learner.joblib"))

