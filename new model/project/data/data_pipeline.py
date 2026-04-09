"""
Data pipeline: preprocessing, feature engineering, normalization, encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

DATA_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(DATA_DIR, "..", "models")


def load_data(csv_path: str = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "Student_Performance.csv")
    df = pd.read_csv(csv_path)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: mean for numeric, mode for categorical."""
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features."""
    df = df.copy()
    # engagement_score
    if "engagement_score" not in df.columns:
        df["engagement_score"] = np.clip(
            df["time_spent_hours"] * 10 + df["attempts"] * 5, 0, 100
        ).round(2)

    # consistency_score
    if "consistency_score" not in df.columns:
        df["consistency_score"] = (
            (df["attendance"] + df["assignment_score"]) / 2
        ).round(2)

    # learning_efficiency
    if "learning_efficiency" not in df.columns:
        df["learning_efficiency"] = (
            df["quiz_score"] / (df["time_spent_hours"] + 0.1)
        ).round(2)

    return df


CATEGORICAL_COLS = ["gender", "learning_style", "parental_support", "stress_level"]
NUMERIC_FEATURES = [
    "age", "attendance", "assignment_score", "quiz_score",
    "time_spent_hours", "attempts", "previous_score",
    "engagement_score", "consistency_score", "learning_efficiency",
    "internet_access", "extracurricular"
]
TARGET = "final_score"


def encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """Label-encode categorical columns."""
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))
    return df, encoders


def normalize_features(X: pd.DataFrame, scaler: StandardScaler = None, fit: bool = True):
    """Scale numeric features."""
    if scaler is None:
        scaler = StandardScaler()
    cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    if fit:
        X[cols] = scaler.fit_transform(X[cols])
    else:
        X[cols] = scaler.transform(X[cols])
    return X, scaler


def preprocess(df: pd.DataFrame, encoders=None, scaler=None, fit=True):
    """Full preprocessing pipeline."""
    df = handle_missing_values(df)
    df = engineer_features(df)
    df, encoders = encode_categoricals(df, encoders=encoders, fit=fit)

    feature_cols = NUMERIC_FEATURES + [c for c in CATEGORICAL_COLS if c in df.columns]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df[TARGET].copy() if TARGET in df.columns else None

    X, scaler = normalize_features(X, scaler=scaler, fit=fit)

    return X, y, encoders, scaler, feature_cols


if __name__ == "__main__":
    df = load_data()
    X, y, encoders, scaler, feature_cols = preprocess(df)
    print("Preprocessed shape:", X.shape)
    print("Features:", feature_cols)
