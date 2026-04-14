"""Embedding model with two-tower optional path and MF fallback."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from ml.recommender.candidate_generation import TOPIC_LIBRARY


@dataclass
class EmbeddingArtifacts:
    user_vectors: Dict[str, np.ndarray]
    item_vectors: Dict[str, np.ndarray]
    vector_dim: int
    method: str
    item_popularity: Dict[str, float] | None = None
    user_history: Dict[str, List[str]] | None = None
    item_metadata: Dict[str, Dict[str, float]] | None = None


def _subject_topic_pool(subject: str, max_topics: int = 5) -> List[str]:
    topics = TOPIC_LIBRARY.get(subject, [])
    if topics:
        return topics[:max_topics]
    fallback = [topic for values in TOPIC_LIBRARY.values() for topic in values]
    return fallback[:max_topics]


def build_implicit_events_from_performance(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"student_id", "subject_strength", "subject_weakness", "quiz_score", "attendance", "assignment_score", "previous_score", "time_spent_hours", "attempts"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=["user_id", "item_id", "interaction_weight", "source"])

    rows: List[Dict[str, object]] = []
    for _, row in frame.iterrows():
        user_id = str(row.get("student_id"))
        quiz_score = float(row.get("quiz_score", 0.0))
        attendance = float(row.get("attendance", 0.0))
        assignment_score = float(row.get("assignment_score", 0.0))
        previous_score = float(row.get("previous_score", 0.0))
        time_spent = float(row.get("time_spent_hours", 1.0))
        attempts = float(row.get("attempts", 1.0))

        mastery = np.clip((0.35 * quiz_score + 0.25 * assignment_score + 0.20 * attendance + 0.20 * previous_score) / 100.0, 0.0, 1.0)
        effort = np.clip((time_spent / 8.0) + (attempts / 10.0), 0.2, 2.0)
        remediation_need = np.clip((70.0 - quiz_score) / 20.0 + (65.0 - attendance) / 30.0 + (5.0 - min(attempts, 5.0)) * 0.1, 0.2, 2.2)

        strength_topics = _subject_topic_pool(str(row.get("subject_strength", "")), max_topics=4)
        weakness_topics = _subject_topic_pool(str(row.get("subject_weakness", "")), max_topics=4)

        for rank, topic in enumerate(strength_topics, start=1):
            weight = (0.75 + mastery + effort * 0.15) / rank
            rows.append({"user_id": user_id, "item_id": topic, "interaction_weight": float(weight), "source": "strength"})

        for rank, topic in enumerate(weakness_topics, start=1):
            weight = (0.90 + remediation_need) / rank
            rows.append({"user_id": user_id, "item_id": topic, "interaction_weight": float(weight), "source": "weakness"})

        if quiz_score >= 75:
            for topic in _subject_topic_pool(str(row.get("subject_strength", "")), max_topics=5)[2:5]:
                rows.append({"user_id": user_id, "item_id": topic, "interaction_weight": 0.35, "source": "extension"})

    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events["interaction_weight"] = events["interaction_weight"].astype(float).clip(lower=0.05)
    return events


def build_content_item_vectors(topics: Dict[str, List[str]] | None = None, vector_dim: int = 32) -> EmbeddingArtifacts:
    topic_map = topics or TOPIC_LIBRARY
    items: List[str] = [topic for values in topic_map.values() for topic in values]
    if not items:
        return EmbeddingArtifacts(user_vectors={}, item_vectors={}, vector_dim=vector_dim, method="content")

    documents = []
    for subject, subject_topics in topic_map.items():
        for idx, topic in enumerate(subject_topics):
            documents.append(f"{subject} {topic} {idx} prerequisite mastery practice")

    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = tfidf.fit_transform(documents)
    k = max(2, min(vector_dim, min(matrix.shape) - 1))
    svd = TruncatedSVD(n_components=k, random_state=42)
    reduced = svd.fit_transform(matrix)

    item_vectors = {item: reduced[i].astype(np.float32) for i, item in enumerate(items)}
    item_popularity = {item: float(1.0 / (1.0 + idx)) for idx, item in enumerate(items)}
    metadata = {item: {"subject_rank": float(idx), "subject_weight": float(1.0 / (idx + 1))} for idx, item in enumerate(items)}
    return EmbeddingArtifacts(user_vectors={}, item_vectors=item_vectors, vector_dim=k, method="content", item_popularity=item_popularity, item_metadata=metadata)


class TwoTowerModel(nn.Module if nn else object):
    def __init__(self, user_dim: int, item_dim: int, hidden_dim: int = 64):
        if not nn:
            return
        super().__init__()
        self.user_tower = nn.Sequential(nn.Linear(user_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.item_tower = nn.Sequential(nn.Linear(item_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, user_x, item_x):
        if not nn:
            return None
        u = self.user_tower(user_x)
        i = self.item_tower(item_x)
        return (u * i).sum(dim=1)


def train_matrix_factorization(events: pd.DataFrame, vector_dim: int = 32) -> EmbeddingArtifacts:
    if events.empty:
        return EmbeddingArtifacts(user_vectors={}, item_vectors={}, vector_dim=vector_dim, method="mf")

    clean = events.copy()
    if "interaction_weight" not in clean.columns:
        clean["interaction_weight"] = 1.0
    clean = clean[["user_id", "item_id", "interaction_weight"]].dropna()
    clean["user_id"] = clean["user_id"].astype(str)
    clean["item_id"] = clean["item_id"].astype(str)
    clean["interaction_weight"] = clean["interaction_weight"].astype(float).clip(lower=0.05)

    if clean.empty:
        return EmbeddingArtifacts(user_vectors={}, item_vectors={}, vector_dim=vector_dim, method="mf")

    users = sorted(clean["user_id"].unique())
    items = sorted(clean["item_id"].unique())
    if len(users) < 2 or len(items) < 2:
        return build_content_item_vectors(vector_dim=vector_dim)

    user_index = {user_id: idx for idx, user_id in enumerate(users)}
    item_index = {item_id: idx for idx, item_id in enumerate(items)}
    user_history = clean.groupby("user_id")["item_id"].apply(lambda s: list(dict.fromkeys(s.tolist()))).to_dict()
    item_popularity = clean.groupby("item_id")["interaction_weight"].sum().to_dict()

    rng = np.random.default_rng(42)
    k = max(2, min(vector_dim, min(len(users), len(items), 32)))
    user_factors = rng.normal(0.0, 0.1, size=(len(users), k)).astype(np.float32)
    item_factors = rng.normal(0.0, 0.1, size=(len(items), k)).astype(np.float32)
    user_bias = np.zeros(len(users), dtype=np.float32)
    item_bias = np.zeros(len(items), dtype=np.float32)

    positive_pairs: List[Tuple[int, int, float]] = [
        (user_index[row.user_id], item_index[row.item_id], float(row.interaction_weight))
        for row in clean.itertuples(index=False)
    ]
    user_positive_items = defaultdict(set)
    for u_idx, i_idx, _ in positive_pairs:
        user_positive_items[u_idx].add(i_idx)

    if not positive_pairs:
        return EmbeddingArtifacts(user_vectors={}, item_vectors={}, vector_dim=k, method="mf")

    n_epochs = 25
    reg = 0.02
    learning_rate = 0.04
    for _ in range(n_epochs):
        rng.shuffle(positive_pairs)
        for u_idx, i_idx, weight in positive_pairs:
            neg_idx = int(rng.integers(0, len(items)))
            attempts = 0
            while neg_idx in user_positive_items[u_idx] and attempts < 10:
                neg_idx = int(rng.integers(0, len(items)))
                attempts += 1
            if neg_idx in user_positive_items[u_idx]:
                continue

            u_vec = user_factors[u_idx]
            i_vec = item_factors[i_idx]
            j_vec = item_factors[neg_idx]

            pos_score = float(np.dot(u_vec, i_vec) + user_bias[u_idx] + item_bias[i_idx])
            neg_score = float(np.dot(u_vec, j_vec) + user_bias[u_idx] + item_bias[neg_idx])
            x_uij = pos_score - neg_score
            sigmoid = 1.0 / (1.0 + math.exp(np.clip(x_uij, -35.0, 35.0)))
            scaled_lr = learning_rate * weight

            user_factors[u_idx] += scaled_lr * ((i_vec - j_vec) * sigmoid - reg * u_vec)
            item_factors[i_idx] += scaled_lr * (u_vec * sigmoid - reg * i_vec)
            item_factors[neg_idx] += scaled_lr * (-u_vec * sigmoid - reg * j_vec)
            user_bias[u_idx] += scaled_lr * (sigmoid - reg * user_bias[u_idx])
            item_bias[i_idx] += scaled_lr * (sigmoid - reg * item_bias[i_idx])
            item_bias[neg_idx] += scaled_lr * (-sigmoid - reg * item_bias[neg_idx])

    user_vectors = {user_id: user_factors[idx] for user_id, idx in user_index.items()}
    item_vectors = {item_id: item_factors[idx] for item_id, idx in item_index.items()}
    return EmbeddingArtifacts(
        user_vectors=user_vectors,
        item_vectors=item_vectors,
        vector_dim=k,
        method="bpr",
        item_popularity=item_popularity,
        user_history=user_history,
    )


def train_two_tower_if_available(events: pd.DataFrame, vector_dim: int = 32) -> EmbeddingArtifacts:
    if events.empty:
        return EmbeddingArtifacts(user_vectors={}, item_vectors={}, vector_dim=vector_dim, method="mf")

    if {"user_id", "item_id"}.issubset(events.columns):
        return train_matrix_factorization(events, vector_dim=vector_dim)

    return train_matrix_factorization(events, vector_dim=vector_dim)


def save_embedding_artifacts(path: str, artifacts: EmbeddingArtifacts) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(artifacts, path)


def load_embedding_artifacts(path: str) -> EmbeddingArtifacts | None:
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def build_ann_index(item_vectors: Dict[str, np.ndarray]):
    ids = list(item_vectors.keys())
    if not ids:
        return {"ids": [], "index": None, "method": "none"}
    mat = np.vstack([item_vectors[i] for i in ids]).astype("float32")

    if faiss is not None:
        index = faiss.IndexFlatIP(mat.shape[1])
        faiss.normalize_L2(mat)
        index.add(mat)
        return {"ids": ids, "index": index, "method": "faiss"}

    nn_index = NearestNeighbors(metric="cosine", n_neighbors=min(50, len(ids)))
    nn_index.fit(mat)
    return {"ids": ids, "index": nn_index, "method": "sklearn"}


def ann_query(index_pack, query_vec: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
    ids = index_pack.get("ids", [])
    index = index_pack.get("index")
    method = index_pack.get("method")
    if index is None or not ids:
        return []

    vec = query_vec.astype("float32").reshape(1, -1)
    if method == "faiss":
        import faiss as _faiss  # type: ignore

        _faiss.normalize_L2(vec)
        scores, idxs = index.search(vec, min(top_k, len(ids)))
        return [(ids[int(i)], float(scores[0][pos])) for pos, i in enumerate(idxs[0]) if int(i) >= 0]

    distances, idxs = index.kneighbors(vec, n_neighbors=min(top_k, len(ids)))
    scored = []
    for pos, i in enumerate(idxs[0]):
        score = 1.0 - float(distances[0][pos])
        scored.append((ids[int(i)], score))
    return scored


def score_user_items(user_vec: np.ndarray, item_vectors: Dict[str, np.ndarray], item_popularity: Dict[str, float] | None = None) -> List[Tuple[str, float]]:
    popularity = item_popularity or {}
    scored: List[Tuple[str, float]] = []
    for item_id, vec in item_vectors.items():
        similarity = float(np.dot(user_vec.astype(np.float32), vec.astype(np.float32)))
        scored.append((item_id, similarity + 0.05 * float(popularity.get(item_id, 0.0))))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored

