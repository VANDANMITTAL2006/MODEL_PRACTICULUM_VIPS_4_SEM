"""
Offline evaluation for recommendation quality.

Metrics reported:
- Content-based weak-subject alignment
- Collaborative and hybrid Hit@K, Precision@K, Recall@K (using subject_strength topics as proxy relevance)
- Recommendation coverage

Run:
  python models/evaluate_recommender.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Set

import numpy as np
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, BASE_DIR)

from backend.recommender import (
    TOPIC_LIBRARY,
    content_based_recommend,
    collaborative_recommend,
    hybrid_recommend,
    DEFAULT_KNN_NEIGHBORS,
    DEFAULT_KNN_METRIC,
)


def _safe_topics(subject: str) -> List[str]:
    return TOPIC_LIBRARY.get(subject, [])


def _hit_at_k(recommended: List[str], relevant: Set[str]) -> float:
    if not recommended:
        return 0.0
    return 1.0 if any(t in relevant for t in recommended) else 0.0


def _precision_at_k(recommended: List[str], relevant: Set[str]) -> float:
    if not recommended:
        return 0.0
    return sum(1 for t in recommended if t in relevant) / len(recommended)


def _recall_at_k(recommended: List[str], relevant: Set[str]) -> float:
    if not relevant:
        return 0.0
    return sum(1 for t in recommended if t in relevant) / len(relevant)


def _expected_topic_index_band(num_topics: int, quiz_score: float) -> Set[int]:
    """Expected difficulty band in weak-subject topic list based on quiz score."""
    if num_topics <= 0:
        return set()

    if quiz_score < 60:
        # Foundational range
        stop = max(1, int(np.ceil(num_topics * 0.5)))
        return set(range(0, stop))

    if quiz_score < 80:
        # Mid-level range
        start = max(0, num_topics // 3)
        stop = min(num_topics, start + max(2, num_topics // 2))
        return set(range(start, stop))

    # Advanced range
    start = max(0, num_topics // 2)
    return set(range(start, num_topics))


def _content_difficulty_fit(weak_subject: str, quiz_score: float, recommended: List[str]) -> float:
    """How well content topics match expected difficulty for this learner."""
    if not recommended:
        return 0.0

    subject_topics = TOPIC_LIBRARY.get(weak_subject, [])
    if not subject_topics:
        return 0.0

    expected_band = _expected_topic_index_band(len(subject_topics), quiz_score)
    if not expected_band:
        return 0.0

    hits = 0
    valid = 0
    for topic in recommended:
        if topic not in subject_topics:
            continue
        valid += 1
        if subject_topics.index(topic) in expected_band:
            hits += 1

    if valid == 0:
        return 0.0
    return hits / valid


def evaluate(
    df: pd.DataFrame,
    k: int = 5,
    num_neighbors: int = DEFAULT_KNN_NEIGHBORS,
    metric: str = DEFAULT_KNN_METRIC,
) -> Dict[str, float]:
    all_topics = {t for topics in TOPIC_LIBRARY.values() for t in topics}

    cb_alignment_scores: List[float] = []
    cb_difficulty_fit_scores: List[float] = []

    collab_hits: List[float] = []
    collab_precision: List[float] = []
    collab_recall: List[float] = []

    hybrid_hits: List[float] = []
    hybrid_precision: List[float] = []
    hybrid_recall: List[float] = []

    collab_unique = set()
    hybrid_unique = set()

    required_cols = [
        "quiz_score",
        "engagement_score",
        "consistency_score",
        "attempts",
        "subject_strength",
        "subject_weakness",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    for idx, row in df.iterrows():
        df_loo = df.drop(index=idx)

        student_features = {
            "quiz_score": float(row.get("quiz_score", 0)),
            "engagement_score": float(row.get("engagement_score", 0)),
            "consistency_score": float(row.get("consistency_score", 0)),
            "attempts": int(row.get("attempts", 0)),
        }

        weak_subject = str(row.get("subject_weakness", ""))
        strength_subject = str(row.get("subject_strength", ""))
        quiz_score = float(row.get("quiz_score", 0))

        cb = content_based_recommend(weak_subject, quiz_score, num_topics=k)
        collab = collaborative_recommend(
            student_features,
            df_loo,
            num_neighbors=num_neighbors,
            num_topics=k,
            metric=metric,
        )
        hyb = hybrid_recommend(
            subject_weakness=weak_subject,
            quiz_score=quiz_score,
            student_features=student_features,
            df=df_loo,
            num_topics=k,
            num_neighbors=num_neighbors,
            metric=metric,
        )

        cb_topics = cb.get("topics", [])[:k]
        collab_topics = collab.get("topics", [])[:k]
        hybrid_topics = hyb.get("recommended_topics", [])[:k]

        weak_topics = set(_safe_topics(weak_subject))
        relevant_strength_topics = set(_safe_topics(strength_subject))

        # Content-based objective proxy: are recommended topics from weak-subject catalog?
        if cb_topics:
            cb_alignment = sum(1 for t in cb_topics if t in weak_topics) / len(cb_topics)
        else:
            cb_alignment = 0.0
        cb_alignment_scores.append(cb_alignment)
        cb_difficulty_fit_scores.append(_content_difficulty_fit(weak_subject, quiz_score, cb_topics))

        # Collaborative/hybrid objective proxy: do recommendations match strength-topic relevance?
        collab_hits.append(_hit_at_k(collab_topics, relevant_strength_topics))
        collab_precision.append(_precision_at_k(collab_topics, relevant_strength_topics))
        collab_recall.append(_recall_at_k(collab_topics, relevant_strength_topics))

        hybrid_hits.append(_hit_at_k(hybrid_topics, relevant_strength_topics))
        hybrid_precision.append(_precision_at_k(hybrid_topics, relevant_strength_topics))
        hybrid_recall.append(_recall_at_k(hybrid_topics, relevant_strength_topics))

        collab_unique.update(collab_topics)
        hybrid_unique.update(hybrid_topics)

    metrics = {
        "samples": float(len(df)),
        "k": float(k),
        "content_weak_subject_alignment": float(np.mean(cb_alignment_scores)),
        "content_difficulty_fit": float(np.mean(cb_difficulty_fit_scores)),
        "collab_hit_at_k": float(np.mean(collab_hits)),
        "collab_precision_at_k": float(np.mean(collab_precision)),
        "collab_recall_at_k": float(np.mean(collab_recall)),
        "hybrid_hit_at_k": float(np.mean(hybrid_hits)),
        "hybrid_precision_at_k": float(np.mean(hybrid_precision)),
        "hybrid_recall_at_k": float(np.mean(hybrid_recall)),
        "collab_coverage": float(len(collab_unique) / len(all_topics)) if all_topics else 0.0,
        "hybrid_coverage": float(len(hybrid_unique) / len(all_topics)) if all_topics else 0.0,
    }
    return metrics


def main() -> None:
    data_path = os.path.join(BASE_DIR, "data", "Student_Performance.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    metrics = evaluate(df, k=5)

    print("=" * 60)
    print("Recommendation Evaluation (Offline)")
    print("=" * 60)
    print(f"Samples: {int(metrics['samples'])} | K: {int(metrics['k'])}")
    print()
    print("Content-based")
    print(f"  Weak-subject alignment (sanity check): {metrics['content_weak_subject_alignment']:.4f}")
    print(f"  Difficulty-fit score (strict):        {metrics['content_difficulty_fit']:.4f}")
    print()
    print("Collaborative (KNN)")
    print(f"  Hit@5:       {metrics['collab_hit_at_k']:.4f}")
    print(f"  Precision@5: {metrics['collab_precision_at_k']:.4f}")
    print(f"  Recall@5:    {metrics['collab_recall_at_k']:.4f}")
    print(f"  Coverage:    {metrics['collab_coverage']:.4f}")
    print()
    print("Hybrid")
    print(f"  Hit@5:       {metrics['hybrid_hit_at_k']:.4f}")
    print(f"  Precision@5: {metrics['hybrid_precision_at_k']:.4f}")
    print(f"  Recall@5:    {metrics['hybrid_recall_at_k']:.4f}")
    print(f"  Coverage:    {metrics['hybrid_coverage']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
