"""Candidate generation for content and collaborative retrieval."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

TOPIC_LIBRARY = {
    "Algebra": ["Basics of Algebra", "Linear Equations", "Quadratic Equations", "Polynomials", "Algebraic Expressions"],
    "Geometry": ["Points, Lines & Planes", "Angles & Triangles", "Circles & Area", "3D Geometry", "Coordinate Geometry"],
    "Calculus": ["Limits & Continuity", "Differentiation Basics", "Integration Techniques", "Applications of Derivatives", "Multivariable Calculus"],
    "Statistics": ["Data & Charts", "Probability Basics", "Distributions", "Hypothesis Testing", "Regression Analysis"],
    "Physics": ["Kinematics", "Newton's Laws", "Work & Energy", "Waves & Optics", "Electrostatics"],
    "Chemistry": ["Atomic Structure", "Periodic Table", "Chemical Bonding", "Redox Reactions", "Organic Chemistry Basics"],
    "Biology": ["Cell Biology", "Genetics & Heredity", "Human Physiology", "Ecology & Environment", "Evolution"],
    "History": ["Ancient Civilizations", "Medieval History", "Industrial Revolution", "World Wars", "Modern India"],
    "Literature": ["Poetry Analysis", "Essay Writing", "Short Story Elements", "Drama & Theatre", "Literary Devices"],
    "Computer Science": ["Intro to Programming", "Data Structures", "Algorithms", "Databases Basics", "Networking Fundamentals"],
}

KNN_FEATURES = ["quiz_score", "engagement_score", "consistency_score", "attempts"]
RICH_FEATURES = [
    "quiz_score",
    "engagement_score",
    "consistency_score",
    "attempts",
    "previous_score",
    "time_spent_hours",
    "attendance",
    "assignment_score",
    "learning_efficiency",
    "score_stability",
    "trend_momentum",
    "behavior_embedding_1",
    "behavior_embedding_2",
    "behavior_embedding_3",
]
CAT_FEATURES = ["gender", "learning_style", "parental_support", "stress_level", "subject_strength", "subject_weakness"]


def content_candidates(subject_weakness: str, quiz_score: float, max_candidates: int = 10) -> List[str]:
    topics = TOPIC_LIBRARY.get(subject_weakness, [])
    if not topics:
        all_topics = [t for items in TOPIC_LIBRARY.values() for t in items]
        return all_topics[:max_candidates]

    if quiz_score < 50:
        ordered = topics
    elif quiz_score < 70:
        ordered = topics[:2] + topics[2:]
    else:
        ordered = list(reversed(topics))
    return ordered[:max_candidates]


def _topic_pool_from_row(row: pd.Series, similarity: float) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    strength = str(row.get("subject_strength", ""))
    weakness = str(row.get("subject_weakness", ""))
    quiz_score = float(row.get("quiz_score", 0.0))
    final_score = float(row.get("final_score", quiz_score))

    strength_topics = TOPIC_LIBRARY.get(strength, [])
    weakness_topics = TOPIC_LIBRARY.get(weakness, [])

    for rank, topic in enumerate(strength_topics, start=1):
        boost = similarity * (0.9 + final_score / 120.0) / rank
        scores[topic] = scores.get(topic, 0.0) + boost

    for rank, topic in enumerate(weakness_topics, start=1):
        gap = max(0.0, 70.0 - quiz_score) / 70.0
        boost = similarity * (0.4 + gap) / rank
        scores[topic] = scores.get(topic, 0.0) + boost

    if quiz_score >= 75 and strength_topics:
        for topic in strength_topics[2:]:
            scores[topic] = scores.get(topic, 0.0) + similarity * 0.2

    return scores


def collaborative_candidates(
    student_features: Dict[str, float],
    df: pd.DataFrame,
    neighbors: int = 6,
    metric: str = "euclidean",
    max_candidates: int = 12,
) -> List[str]:
    if len(df) < 2:
        return []

    available_numeric = [c for c in RICH_FEATURES if c in df.columns]
    if not available_numeric:
        available_numeric = [c for c in KNN_FEATURES if c in df.columns]
    if not available_numeric:
        return []

    feature_frame = df[available_numeric].fillna(0).copy()
    query_frame = pd.DataFrame([student_features])
    query_numeric = [c for c in available_numeric if c in query_frame.columns]
    for col in available_numeric:
        if col not in query_frame.columns:
            query_frame[col] = float(feature_frame[col].median())

    for col in CAT_FEATURES:
        if col in df.columns:
            dummies = pd.get_dummies(df[col].fillna("Unknown"), prefix=col)
            feature_frame = pd.concat([feature_frame, dummies], axis=1)
            query_value = str(student_features.get(col, "Unknown"))
            query_dummy = pd.DataFrame([{name: 1.0 if name == f"{col}_{query_value}" else 0.0 for name in dummies.columns}])
            query_frame = pd.concat([query_frame, query_dummy], axis=1)

    feature_frame = feature_frame.fillna(0)
    query_frame = query_frame.reindex(columns=feature_frame.columns, fill_value=0)

    x = feature_frame.values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = NearestNeighbors(n_neighbors=min(neighbors, len(df) - 1), metric=metric)
    model.fit(x_scaled)

    query = query_frame.values.astype(float)
    query_scaled = scaler.transform(query)
    distances, indices = model.kneighbors(query_scaled)

    scores: Dict[str, float] = {}
    for i, row_idx in enumerate(indices[0]):
        row = df.iloc[int(row_idx)]
        dist = float(distances[0][i])
        similarity = 1.0 / (1.0 + dist)
        topic_scores = _topic_pool_from_row(row, similarity)
        for topic, score in topic_scores.items():
            scores[topic] = scores.get(topic, 0.0) + score

    if "subject_weakness" in student_features:
        subject_topics = TOPIC_LIBRARY.get(str(student_features.get("subject_weakness", "")), [])
        for rank, topic in enumerate(subject_topics, start=1):
            scores[topic] = scores.get(topic, 0.0) + 0.2 / rank

    ranked = [k for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return ranked[:max_candidates]


def build_candidates(
    subject_weakness: str,
    quiz_score: float,
    student_features: Dict[str, float],
    df: pd.DataFrame,
    neighbors: int = 6,
    metric: str = "euclidean",
) -> Dict[str, List[str]]:
    return {
        "content": content_candidates(subject_weakness, quiz_score),
        "collaborative": collaborative_candidates(student_features, df, neighbors=neighbors, metric=metric),
    }
