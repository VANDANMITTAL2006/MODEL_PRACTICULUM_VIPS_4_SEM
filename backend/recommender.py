"""
Hybrid Recommendation System:
1. Content-based: recommend topics where student score < threshold
2. Collaborative: KNN-based similar student lookup
3. Hybrid combiner
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

TOPIC_LIBRARY = {
    "Algebra": [
        "Basics of Algebra", "Linear Equations", "Quadratic Equations",
        "Polynomials", "Algebraic Expressions"
    ],
    "Geometry": [
        "Points, Lines & Planes", "Angles & Triangles", "Circles & Area",
        "3D Geometry", "Coordinate Geometry"
    ],
    "Calculus": [
        "Limits & Continuity", "Differentiation Basics", "Integration Techniques",
        "Applications of Derivatives", "Multivariable Calculus"
    ],
    "Statistics": [
        "Data & Charts", "Probability Basics", "Distributions",
        "Hypothesis Testing", "Regression Analysis"
    ],
    "Physics": [
        "Kinematics", "Newton's Laws", "Work & Energy",
        "Waves & Optics", "Electrostatics"
    ],
    "Chemistry": [
        "Atomic Structure", "Periodic Table", "Chemical Bonding",
        "Redox Reactions", "Organic Chemistry Basics"
    ],
    "Biology": [
        "Cell Biology", "Genetics & Heredity", "Human Physiology",
        "Ecology & Environment", "Evolution"
    ],
    "History": [
        "Ancient Civilizations", "Medieval History", "Industrial Revolution",
        "World Wars", "Modern India"
    ],
    "Literature": [
        "Poetry Analysis", "Essay Writing", "Short Story Elements",
        "Drama & Theatre", "Literary Devices"
    ],
    "Computer Science": [
        "Intro to Programming", "Data Structures", "Algorithms",
        "Databases Basics", "Networking Fundamentals"
    ],
}

SCORE_THRESHOLD = 60.0  # below this → weak area
KNN_FEATURES = ["quiz_score", "engagement_score", "consistency_score", "attempts"]
DEFAULT_KNN_NEIGHBORS = 3
DEFAULT_KNN_METRIC = "euclidean"


def _desired_difficulty(quiz_score: float) -> float:
    """Map quiz score to a preferred topic difficulty in [0, 1]."""
    if quiz_score < 60:
        return 0.2
    if quiz_score < 80:
        return 0.5
    return 0.8


def _topic_difficulty_score(topic: str, subject: str, quiz_score: float) -> float:
    """Higher score when topic index matches expected learner difficulty."""
    topics = TOPIC_LIBRARY.get(subject, [])
    if not topics or topic not in topics:
        return 0.5
    if len(topics) == 1:
        return 1.0

    level = topics.index(topic) / (len(topics) - 1)
    desired = _desired_difficulty(quiz_score)
    return max(0.0, 1.0 - abs(level - desired))


def _segment_from_features(quiz_score: float, engagement_score: float) -> str:
    """Simple segment rule used only for adaptive ranking weights."""
    if quiz_score >= 75 and engagement_score >= 60:
        return "fast"
    if quiz_score < 55 or engagement_score < 35:
        return "struggling"
    return "low_engagement"


def _hybrid_weights(quiz_score: float, engagement_score: float) -> tuple:
    """Return (content_w, collab_w, difficulty_w) by learner segment."""
    segment = _segment_from_features(quiz_score, engagement_score)
    if segment == "struggling":
        return (0.60, 0.20, 0.20)
    if segment == "fast":
        return (0.30, 0.50, 0.20)
    return (0.45, 0.35, 0.20)


def _rank_topics_from_list(topics: list) -> dict:
    """Convert ordered topics into a normalized relevance score dict."""
    if not topics:
        return {}
    n = len(topics)
    return {topic: (n - idx) / n for idx, topic in enumerate(topics)}


def content_based_recommend(
    subject_weakness: str,
    quiz_score: float,
    num_topics: int = 3
) -> dict:
    """Recommend topics from the student's weak subject."""
    topics = TOPIC_LIBRARY.get(subject_weakness, [])
    if not topics:
        # Fallback: pick from all
        all_topics = [t for ts in TOPIC_LIBRARY.values() for t in ts]
        np.random.shuffle(all_topics)
        topics = all_topics[:num_topics]

    if quiz_score < SCORE_THRESHOLD:
        # Student is weak — recommend foundational topics (first half)
        recommended = topics[: max(num_topics, len(topics) // 2)][:num_topics]
    else:
        # Medium performer — recommend mid-tier
        mid = len(topics) // 3
        recommended = topics[mid : mid + num_topics]

    return {
        "method": "content_based",
        "weak_area": subject_weakness,
        "topics": recommended[:num_topics],
    }


def collaborative_recommend(
    student_features: dict,
    df: pd.DataFrame,
    num_neighbors: int = DEFAULT_KNN_NEIGHBORS,
    num_topics: int = 3,
    metric: str = DEFAULT_KNN_METRIC,
) -> dict:
    """
    Find similar students using KNN and recommend topics their strongest subjects cover.
    student_features keys: quiz_score, engagement_score, consistency_score, attempts
    """
    knn_cols = [c for c in KNN_FEATURES if c in df.columns]
    available = [c for c in knn_cols if c in df.columns]
    if not available or len(df) < 2:
        return {"method": "collaborative", "topics": []}

    X_knn = df[available].fillna(0).values
    scaler = StandardScaler()
    X_knn_scaled = scaler.fit_transform(X_knn)

    model = NearestNeighbors(
        n_neighbors=min(num_neighbors, len(df) - 1),
        metric=metric,
    )
    model.fit(X_knn_scaled)

    query = np.array([[student_features.get(c, 0) for c in available]])
    query_scaled = scaler.transform(query)
    distances, indices = model.kneighbors(query_scaled)

    similar_students = df.iloc[indices[0]]
    # Weighted topic votes from similar students
    recommended_scores = {}
    for rank, (_, row) in enumerate(similar_students.iterrows()):
        dist = float(distances[0][rank]) if rank < len(distances[0]) else 1.0
        similarity = 1.0 / (1.0 + dist)
        strength = row.get("subject_strength", "")
        topics = TOPIC_LIBRARY.get(strength, [])

        if not topics:
            continue

        # Earlier topics are more foundational; keep mild bias by rank.
        for t_idx, topic in enumerate(topics):
            rank_bonus = (len(topics) - t_idx) / len(topics)
            score = similarity * rank_bonus
            recommended_scores[topic] = recommended_scores.get(topic, 0.0) + score

    ranked_topics = [
        topic for topic, _ in sorted(recommended_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        "method": "collaborative",
        "topics": ranked_topics[:num_topics],
    }


def hybrid_recommend(
    subject_weakness: str,
    quiz_score: float,
    student_features: dict,
    df: pd.DataFrame,
    num_topics: int = 5,
    num_neighbors: int = DEFAULT_KNN_NEIGHBORS,
    metric: str = DEFAULT_KNN_METRIC,
) -> dict:
    """Combine content-based + collaborative results."""
    cb = content_based_recommend(subject_weakness, quiz_score, num_topics)
    collab = collaborative_recommend(
        student_features,
        df,
        num_neighbors=num_neighbors,
        num_topics=max(num_topics, 8),
        metric=metric,
    )

    cb_scores = _rank_topics_from_list(cb["topics"])
    collab_scores = _rank_topics_from_list(collab["topics"])

    engagement = float(student_features.get("engagement_score", 50))
    w_cb, w_collab, w_diff = _hybrid_weights(quiz_score, engagement)

    candidates = set(cb_scores.keys()) | set(collab_scores.keys())
    scored = []
    for topic in candidates:
        content_score = cb_scores.get(topic, 0.0)
        collab_score = collab_scores.get(topic, 0.0)
        difficulty_score = _topic_difficulty_score(topic, subject_weakness, quiz_score)
        final_score = (w_cb * content_score) + (w_collab * collab_score) + (w_diff * difficulty_score)
        scored.append((topic, final_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    combined = [topic for topic, _ in scored]

    return {
        "weak_areas": [subject_weakness],
        "content_based_topics": cb["topics"],
        "collaborative_topics": collab["topics"],
        "recommended_topics": combined[:num_topics],
    }


if __name__ == "__main__":
    import pandas as pd
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "Student_Performance.csv"))
    except FileNotFoundError:
        df = pd.DataFrame()

    result = hybrid_recommend(
        subject_weakness="Algebra",
        quiz_score=45.0,
        student_features={"quiz_score": 45, "engagement_score": 30, "consistency_score": 55, "attempts": 3},
        df=df,
    )
    print(result)
