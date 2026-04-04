"""
Generate synthetic Student_Performance.csv dataset for the Personalized Learning System.
Run this script once to create the dataset before training the model.
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 500  # number of students

subjects = ["Algebra", "Geometry", "Calculus", "Statistics", "Physics",
            "Chemistry", "Biology", "History", "Literature", "Computer Science"]

learning_styles = ["Visual", "Auditory", "Kinesthetic", "Reading/Writing"]

def generate_dataset(n=N):
    data = {
        "student_id": [f"S{str(i).zfill(4)}" for i in range(1, n + 1)],
        "age": np.random.randint(14, 25, n),
        "gender": np.random.choice(["Male", "Female", "Other"], n, p=[0.48, 0.48, 0.04]),
        "learning_style": np.random.choice(learning_styles, n),
        "attendance": np.clip(np.random.normal(75, 15, n), 20, 100).round(1),
        "assignment_score": np.clip(np.random.normal(70, 18, n), 10, 100).round(1),
        "quiz_score": np.clip(np.random.normal(68, 20, n), 5, 100).round(1),
        "time_spent_hours": np.clip(np.random.normal(5, 2, n), 0.5, 15).round(2),
        "attempts": np.random.randint(1, 10, n),
        "subject_strength": np.random.choice(subjects, n),
        "subject_weakness": np.random.choice(subjects, n),
        "previous_score": np.clip(np.random.normal(65, 20, n), 10, 100).round(1),
        "internet_access": np.random.choice([0, 1], n, p=[0.2, 0.8]),
        "parental_support": np.random.choice(["Low", "Medium", "High"], n, p=[0.2, 0.5, 0.3]),
        "extracurricular": np.random.choice([0, 1], n, p=[0.4, 0.6]),
        "stress_level": np.random.choice(["Low", "Medium", "High"], n, p=[0.3, 0.4, 0.3]),
    }

    df = pd.DataFrame(data)

    # Make sure subject_strength != subject_weakness
    mask = df["subject_strength"] == df["subject_weakness"]
    df.loc[mask, "subject_weakness"] = df.loc[mask, "subject_strength"].apply(
        lambda s: np.random.choice([x for x in subjects if x != s])
    )

    # Engineered features
    df["engagement_score"] = (df["time_spent_hours"] * 10 + df["attempts"] * 5).round(2)
    df["engagement_score"] = np.clip(df["engagement_score"], 0, 100)

    df["consistency_score"] = ((df["attendance"] + df["assignment_score"]) / 2).round(2)

    df["learning_efficiency"] = (df["quiz_score"] / (df["time_spent_hours"] + 0.1)).round(2)

    # Final performance score (target)
    # Weighted combination of multiple factors
    noise = np.random.normal(0, 5, n)
    df["final_score"] = (
        0.30 * df["quiz_score"] +
        0.20 * df["assignment_score"] +
        0.15 * df["attendance"] +
        0.15 * df["consistency_score"] +
        0.10 * df["previous_score"] +
        0.05 * df["engagement_score"] +
        0.05 * df["learning_efficiency"] +
        noise
    )
    df["final_score"] = np.clip(df["final_score"], 10, 100).round(1)

    return df


if __name__ == "__main__":
    import os
    df = generate_dataset()
    out_path = os.path.join(os.path.dirname(__file__), "Student_Performance.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Dataset generated: {out_path}")
    print(f"   Shape: {df.shape}")
    print(df.head())
