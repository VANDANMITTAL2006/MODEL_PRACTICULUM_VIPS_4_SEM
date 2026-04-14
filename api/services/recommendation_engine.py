"""Two-stage recommender adapters with backward-compatible API."""

from __future__ import annotations

import os
import sys
from typing import Dict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.recommender.candidate_generation import TOPIC_LIBRARY, build_candidates, content_candidates, collaborative_candidates
from ml.recommender.cold_start import cold_start_recommend
from ml.recommender.ranker import RecommendationRanker

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
DEFAULT_KNN_NEIGHBORS = 6
DEFAULT_KNN_METRIC = "euclidean"

_RANKER = RecommendationRanker()


def content_based_recommend(subject_weakness: str, quiz_score: float, num_topics: int = 3) -> Dict:
    topics = content_candidates(subject_weakness=subject_weakness, quiz_score=quiz_score, max_candidates=num_topics)
    return {
        "method": "content_based",
        "weak_area": subject_weakness,
        "topics": topics[:num_topics],
    }


def collaborative_recommend(
    student_features: Dict,
    df: pd.DataFrame,
    num_neighbors: int = DEFAULT_KNN_NEIGHBORS,
    num_topics: int = 3,
    metric: str = DEFAULT_KNN_METRIC,
) -> Dict:
    topics = collaborative_candidates(student_features, df, neighbors=num_neighbors, metric=metric, max_candidates=num_topics)
    return {"method": "collaborative", "topics": topics[:num_topics]}


def hybrid_recommend(
    subject_weakness: str,
    quiz_score: float,
    student_features: Dict,
    df: pd.DataFrame,
    num_topics: int = 5,
    num_neighbors: int = DEFAULT_KNN_NEIGHBORS,
    metric: str = DEFAULT_KNN_METRIC,
) -> Dict:
    if df.empty or len(df) < 2:
        fallback = cold_start_recommend(subject_weakness, quiz_score, num_topics)
        return {
            "weak_areas": [subject_weakness],
            "content_based_topics": fallback["recommended_topics"],
            "collaborative_topics": [],
            "recommended_topics": fallback["recommended_topics"],
            "recommendation_source": fallback.get("source", "cold_start"),
        }

    candidates = build_candidates(
        subject_weakness=subject_weakness,
        quiz_score=quiz_score,
        student_features=student_features,
        df=df,
        neighbors=num_neighbors,
        metric=metric,
    )
    ranked = _RANKER.rank(candidates=candidates, quiz_score=quiz_score, top_k=num_topics)

    return {
        "weak_areas": [subject_weakness],
        "content_based_topics": candidates.get("content", [])[:num_topics],
        "collaborative_topics": candidates.get("collaborative", [])[:num_topics],
        "recommended_topics": ranked,
        "recommendation_source": "hybrid_ranked",
    }


if __name__ == "__main__":
    csv_path = os.path.join(DATA_DIR, "Student_Performance.csv")
    frame = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    print(
        hybrid_recommend(
            subject_weakness="Algebra",
            quiz_score=47.0,
            student_features={"quiz_score": 47, "engagement_score": 40, "consistency_score": 52, "attempts": 4},
            df=frame,
        )
    )

