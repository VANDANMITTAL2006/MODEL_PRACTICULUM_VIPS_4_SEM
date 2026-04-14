"""Offline recommender evaluation with ranking and slice metrics."""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Set, cast

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from api.services.recommendation_engine import DEFAULT_KNN_METRIC, DEFAULT_KNN_NEIGHBORS, TOPIC_LIBRARY, hybrid_recommend


def _safe_topics(subject: str) -> List[str]:
    return TOPIC_LIBRARY.get(subject, [])


def _difficulty_relevance(subject: str, quiz_score: float) -> List[str]:
    topics = _safe_topics(subject)
    if not topics:
        return [topic for values in TOPIC_LIBRARY.values() for topic in values[:3]]
    if quiz_score < 50:
        return topics[:3]
    if quiz_score < 70:
        return topics[1:4] if len(topics) > 3 else topics[:]
    return topics[-3:] if len(topics) >= 3 else topics[:]


def _dcg_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    gains = [1.0 if t in relevant else 0.0 for t in recommended[:k]]
    return float(sum(g / np.log2(i + 2) for i, g in enumerate(gains)))


def _ndcg_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = _dcg_at_k(recommended, relevant, k)
    ideal = _dcg_at_k(list(relevant), relevant, min(k, len(relevant)))
    return float(dcg / ideal) if ideal > 0 else 0.0


def _map_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    hits = 0
    precisions = []
    for idx, topic in enumerate(recommended[:k], start=1):
        if topic in relevant:
            hits += 1
            precisions.append(hits / idx)
    if not precisions:
        return 0.0
    return float(np.mean(precisions))


def _precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    top = recommended[:k]
    if not top:
        return 0.0
    return float(sum(1 for t in top if t in relevant) / len(top))


def _recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return float(sum(1 for t in recommended[:k] if t in relevant) / len(relevant))


def _hit_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    return 1.0 if any(t in relevant for t in recommended[:k]) else 0.0


def _diversity(topics: List[str]) -> float:
    if len(topics) <= 1:
        return 0.0
    prefixes = [t.split(" ")[0] for t in topics]
    return float(len(set(prefixes)) / len(prefixes))


def evaluate(
    df: pd.DataFrame,
    k: int = 5,
    num_neighbors: int = DEFAULT_KNN_NEIGHBORS,
    metric: str = DEFAULT_KNN_METRIC,
) -> Dict[str, float]:
    all_topics = {t for values in TOPIC_LIBRARY.values() for t in values}
    rows = []
    recommended_unique = set()

    required = ["quiz_score", "engagement_score", "consistency_score", "attempts", "subject_strength", "subject_weakness"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    for idx, row in df.iterrows():
        df_loo = df.drop(index=idx)
        student_features = {
            "quiz_score": float(row.get("quiz_score", 0.0)),
            "engagement_score": float(row.get("engagement_score", 0.0)),
            "consistency_score": float(row.get("consistency_score", 0.0)),
            "attempts": int(row.get("attempts", 1)),
        }

        weak_subject = str(row.get("subject_weakness", ""))
        strength_subject = str(row.get("subject_strength", ""))
        result = hybrid_recommend(
            subject_weakness=weak_subject,
            quiz_score=float(row.get("quiz_score", 0.0)),
            student_features=student_features,
            df=df_loo,
            num_topics=k,
            num_neighbors=num_neighbors,
            metric=metric,
        )

        recs = result.get("recommended_topics", [])[:k]
        recommended_unique.update(recs)
        relevant = set(_difficulty_relevance(weak_subject, float(row.get("quiz_score", 0.0))))

        segment = "cold_start" if len(df_loo) < 2 else ("struggling" if row.get("quiz_score", 0) < 55 else "regular")
        rows.append(
            {
                "segment": segment,
                "hit_at_k": _hit_at_k(recs, relevant, k),
                "precision_at_k": _precision_at_k(recs, relevant, k),
                "recall_at_k": _recall_at_k(recs, relevant, k),
                "ndcg_at_k": _ndcg_at_k(recs, relevant, k),
                "map_at_k": _map_at_k(recs, relevant, k),
                "diversity": _diversity(recs),
                "novelty": float(sum(1 for t in recs if "Basics" not in t) / max(len(recs), 1)),
            }
        )

    frame = pd.DataFrame(rows)
    metrics = {
        "samples": float(len(df)),
        "k": float(k),
        "hit_at_k": float(frame["hit_at_k"].mean()),
        "precision_at_k": float(frame["precision_at_k"].mean()),
        "recall_at_k": float(frame["recall_at_k"].mean()),
        "ndcg_at_k": float(frame["ndcg_at_k"].mean()),
        "map_at_k": float(frame["map_at_k"].mean()),
        "coverage": float(len(recommended_unique) / max(len(all_topics), 1)),
        "diversity": float(frame["diversity"].mean()),
        "novelty": float(frame["novelty"].mean()),
    }

    segment_metrics = frame.groupby("segment").mean(numeric_only=True).to_dict(orient="index")
    cold_start = segment_metrics.get("cold_start", {})
    struggling = segment_metrics.get("struggling", {})
    metrics["cold_start_hit_at_k"] = float(cold_start.get("hit_at_k", 0.0))
    metrics["struggling_hit_at_k"] = float(struggling.get("hit_at_k", 0.0))
    metrics["cold_start_precision_at_k"] = float(cold_start.get("precision_at_k", 0.0))
    metrics["cold_start_recall_at_k"] = float(cold_start.get("recall_at_k", 0.0))

    out_dir = os.path.join(PROJECT_ROOT, "ml", "artifacts")
    frame.to_csv(os.path.join(out_dir, "recommender_offline_metrics_rows.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "recommender_offline_metrics.csv"), index=False)
    with open(os.path.join(out_dir, "recommender_offline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"global": metrics, "segment": segment_metrics}, f, indent=2)
    return metrics


def evaluate_cross_validation(df: pd.DataFrame, folds: int = 5, k: int = 5) -> Dict[str, float]:
    if df.empty:
        return {"folds": 0.0, "cv_hit_at_k": 0.0, "cv_precision_at_k": 0.0, "cv_recall_at_k": 0.0, "cv_ndcg_at_k": 0.0}

    rng = np.random.default_rng(42)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    parts = np.array_split(indices, min(folds, len(df)))
    rows = []
    for part in parts:
        test_df = df.iloc[part]
        train_df = df.drop(df.index[part])
        fold_recs = []
        for _, row in test_df.iterrows():
            student_features = {
                "quiz_score": float(row.get("quiz_score", 0.0)),
                "engagement_score": float(row.get("engagement_score", 0.0)),
                "consistency_score": float(row.get("consistency_score", 0.0)),
                "attempts": int(row.get("attempts", 1)),
            }
            result = hybrid_recommend(
                subject_weakness=str(row.get("subject_weakness", "Algebra")),
                quiz_score=float(row.get("quiz_score", 0.0)),
                student_features=student_features,
                df=train_df,
                num_topics=k,
            )
            recs = result.get("recommended_topics", [])[:k]
            relevant = set(_difficulty_relevance(str(row.get("subject_weakness", "Algebra")), float(row.get("quiz_score", 0.0))))
            fold_recs.append({
                "hit_at_k": _hit_at_k(recs, relevant, k),
                "precision_at_k": _precision_at_k(recs, relevant, k),
                "recall_at_k": _recall_at_k(recs, relevant, k),
                "ndcg_at_k": _ndcg_at_k(recs, relevant, k),
            })
        fold_frame = pd.DataFrame(fold_recs)
        rows.append(fold_frame.mean(numeric_only=True).to_dict())

    frame = pd.DataFrame(rows)
    return {
        "folds": float(len(rows)),
        "cv_hit_at_k": float(frame["hit_at_k"].mean()),
        "cv_precision_at_k": float(frame["precision_at_k"].mean()),
        "cv_recall_at_k": float(frame["recall_at_k"].mean()),
        "cv_ndcg_at_k": float(frame["ndcg_at_k"].mean()),
    }


def simulate_ab_test(df: pd.DataFrame, traffic_split: float = 0.5, k: int = 5) -> Dict[str, float]:
    if df.empty:
        return {"variant_a": 0.0, "variant_b": 0.0, "lift": 0.0}

    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    pivot = int(len(shuffled) * traffic_split)
    a_df = cast(pd.DataFrame, shuffled.iloc[:pivot].copy())
    b_df = cast(pd.DataFrame, shuffled.iloc[pivot:].copy())
    a_metrics = evaluate(a_df, k=k)
    b_metrics = evaluate(b_df, k=k)
    lift = b_metrics["hit_at_k"] - a_metrics["hit_at_k"]
    return {"variant_a": float(a_metrics["hit_at_k"]), "variant_b": float(b_metrics["hit_at_k"]), "lift": float(lift)}


def main() -> None:
    data_path = os.path.join(PROJECT_ROOT, "data", "raw", "Student_Performance.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    df = pd.read_csv(data_path)
    metrics = evaluate(df, k=5)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

