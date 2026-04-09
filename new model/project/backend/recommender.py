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
    num_neighbors: int = 5,
    num_topics: int = 3,
) -> dict:
    """
    Find similar students using KNN and recommend topics their strongest subjects cover.
    student_features keys: quiz_score, engagement_score, consistency_score, attempts
    """
    knn_cols = ["quiz_score", "engagement_score", "consistency_score", "attempts"]
    available = [c for c in knn_cols if c in df.columns]
    if not available or len(df) < 2:
        return {"method": "collaborative", "topics": []}

    X_knn = df[available].fillna(0).values
    model = NearestNeighbors(n_neighbors=min(num_neighbors, len(df) - 1), metric="euclidean")
    model.fit(X_knn)

    query = np.array([[student_features.get(c, 0) for c in available]])
    distances, indices = model.kneighbors(query)

    similar_students = df.iloc[indices[0]]
    # Pick topics from their strength subjects
    recommended = []
    seen = set()
    for _, row in similar_students.iterrows():
        strength = row.get("subject_strength", "")
        topics = TOPIC_LIBRARY.get(strength, [])
        for t in topics:
            if t not in seen:
                recommended.append(t)
                seen.add(t)
            if len(recommended) >= num_topics:
                break
        if len(recommended) >= num_topics:
            break

    return {
        "method": "collaborative",
        "topics": recommended[:num_topics],
    }


def hybrid_recommend(
    subject_weakness: str,
    quiz_score: float,
    student_features: dict,
    df: pd.DataFrame,
    num_topics: int = 5,
) -> dict:
    """Combine content-based + collaborative results."""
    cb = content_based_recommend(subject_weakness, quiz_score, num_topics)
    collab = collaborative_recommend(student_features, df, num_topics=num_topics)

    # Merge & deduplicate
    combined = list(dict.fromkeys(cb["topics"] + collab["topics"]))

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
