"""Data pipeline with strict schema, feature contract, and leakage guards."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "ml", "artifacts")
TARGET = "final_score"
SCHEMA_VERSION = "2.0"

REQUIRED_COLUMNS = [
    "age",
    "gender",
    "learning_style",
    "attendance",
    "assignment_score",
    "quiz_score",
    "time_spent_hours",
    "attempts",
    "previous_score",
    "internet_access",
    "parental_support",
    "extracurricular",
    "stress_level",
]

CATEGORICAL_COLS = ["gender", "learning_style", "parental_support", "stress_level"]

NUMERIC_FEATURES = [
    "age",
    "attendance",
    "assignment_score",
    "quiz_score",
    "time_spent_hours",
    "attempts",
    "previous_score",
    "internet_access",
    "extracurricular",
    "engagement_score",
    "consistency_score",
    "learning_efficiency",
    "interaction_attendance_assignment",
    "interaction_quiz_attempts",
    "difficulty_adjusted_score",
    "score_stability",
    "history_prev_quiz_gap",
    "rolling_quiz_proxy",
    "recency_proxy",
    "frequency_proxy",
    "study_velocity",
    "trend_momentum",
    "behavior_embedding_1",
    "behavior_embedding_2",
    "behavior_embedding_3",
]

LEAKAGE_PATTERNS = ["target", "label", "future", "post", "final_score"]


@dataclass
class PreprocessingMetadata:
    schema_version: str
    created_at_utc: str
    feature_cols: List[str]
    dtypes: Dict[str, str]
    numeric_scaled_cols: List[str]
    categorical_encoded_cols: List[str]
    scaler_class: str
    encoder_class: str


def load_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "Student_Performance.csv")
    return pd.read_csv(csv_path)


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    if require_target and TARGET not in df.columns:
        raise ValueError(f"Dataset missing target column: {TARGET}")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].fillna(out[col].median())
    for col in out.select_dtypes(include=["object", "category"]).columns:
        mode = out[col].mode()
        fallback = mode.iloc[0] if not mode.empty else "Unknown"
        out[col] = out[col].fillna(fallback)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["engagement_score"] = np.clip(out["time_spent_hours"] * 10 + out["attempts"] * 5, 0, 100)
    out["consistency_score"] = ((out["attendance"] + out["assignment_score"]) / 2).clip(0, 100)
    out["learning_efficiency"] = (out["quiz_score"] / (out["time_spent_hours"] + 0.1)).clip(0, 200)

    out["interaction_attendance_assignment"] = (out["attendance"] * out["assignment_score"]) / 100.0
    out["interaction_quiz_attempts"] = out["quiz_score"] / (out["attempts"] + 1.0)

    difficulty_proxy = 100.0 - out["previous_score"].clip(0, 100)
    out["difficulty_adjusted_score"] = out["quiz_score"] - (0.15 * difficulty_proxy)
    out["score_stability"] = 100.0 - np.abs(out["quiz_score"] - out["previous_score"])

    out["history_prev_quiz_gap"] = out["quiz_score"] - out["previous_score"]
    out["rolling_quiz_proxy"] = (0.6 * out["quiz_score"]) + (0.4 * out["previous_score"])
    out["recency_proxy"] = np.clip(100.0 - (out["time_spent_hours"] * 4.0 + out["attempts"] * 3.0), 0, 100)
    out["frequency_proxy"] = (out["attempts"] / (out["time_spent_hours"] + 0.1)).clip(0, 20)
    out["study_velocity"] = out["quiz_score"] / (out["time_spent_hours"] + 0.1)
    out["trend_momentum"] = (out["quiz_score"] - out["previous_score"] + out["learning_efficiency"] * 0.05).clip(-100, 100)
    out["behavior_embedding_1"] = (
        0.45 * out["engagement_score"] + 0.35 * out["consistency_score"] + 0.20 * out["learning_efficiency"]
    ).clip(0, 200)
    out["behavior_embedding_2"] = (
        0.50 * out["quiz_score"] + 0.30 * out["previous_score"] + 0.20 * out["trend_momentum"]
    ).clip(0, 200)
    out["behavior_embedding_3"] = (
        0.55 * out["attendance"] + 0.45 * out["assignment_score"]
    ).clip(0, 100)
    return out


def assert_no_leakage_features(feature_cols: List[str]) -> None:
    for col in feature_cols:
        col_norm = col.lower().strip()
        if col_norm == TARGET:
            raise ValueError("Target column leaked into features")
        if any(token in col_norm for token in LEAKAGE_PATTERNS if token != "final_score"):
            raise ValueError(f"Potential leakage column detected: {col}")


def encode_categoricals(
    df: pd.DataFrame,
    encoders: Optional[Dict[str, LabelEncoder]] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    out = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in out.columns:
            continue
        values = out[col].astype(str)
        if fit:
            encoder = LabelEncoder()
            out[col] = encoder.fit_transform(values)
            encoders[col] = encoder
        else:
            if col not in encoders:
                raise ValueError(f"Missing encoder for categorical column: {col}")
            encoder = encoders[col]
            known = set(encoder.classes_)
            safe_values = values.map(lambda v: v if v in known else encoder.classes_[0])
            out[col] = encoder.transform(safe_values)

    return out, encoders


def normalize_features(
    X: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, StandardScaler]:
    out = X.copy()
    if scaler is None:
        scaler = StandardScaler()
    cols = [c for c in NUMERIC_FEATURES if c in out.columns]
    if cols:
        if fit:
            out[cols] = scaler.fit_transform(out[cols])
        else:
            out[cols] = scaler.transform(out[cols])
    return out, scaler


def feature_contract(df: pd.DataFrame) -> List[str]:
    ordered = [c for c in NUMERIC_FEATURES + CATEGORICAL_COLS if c in df.columns]
    assert_no_leakage_features(ordered)
    return ordered


def save_preprocessing_metadata(
    metadata_path: str,
    feature_cols: List[str],
    X: pd.DataFrame,
) -> None:
    metadata = PreprocessingMetadata(
        schema_version=SCHEMA_VERSION,
        created_at_utc=datetime.utcnow().isoformat(),
        feature_cols=feature_cols,
        dtypes={str(k): str(v) for k, v in X.dtypes.to_dict().items()},
        numeric_scaled_cols=[c for c in NUMERIC_FEATURES if c in feature_cols],
        categorical_encoded_cols=[c for c in CATEGORICAL_COLS if c in feature_cols],
        scaler_class="StandardScaler",
        encoder_class="LabelEncoder",
    )
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata.__dict__, f, indent=2)


def preprocess(
    df: pd.DataFrame,
    encoders: Optional[Dict[str, LabelEncoder]] = None,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
    require_target: bool = True,
    metadata_path: Optional[str] = None,
):
    validate_schema(df, require_target=require_target)
    transformed = handle_missing_values(df)
    transformed = engineer_features(transformed)
    transformed, encoders = encode_categoricals(transformed, encoders=encoders, fit=fit)

    feature_cols = feature_contract(transformed)
    X = transformed.reindex(columns=feature_cols).copy()
    y = transformed[TARGET].copy() if TARGET in transformed.columns else None
    X, scaler = normalize_features(X, scaler=scaler, fit=fit)

    if metadata_path is not None:
        save_preprocessing_metadata(metadata_path, feature_cols, X)
    return X, y, encoders, scaler, feature_cols


if __name__ == "__main__":
    frame = load_data()
    X_data, y_data, enc_map, std_scaler, cols = preprocess(frame, metadata_path=os.path.join(MODELS_DIR, "preprocessing_metadata.json"))
    print("Preprocessed shape:", X_data.shape)
    print("Target shape:", None if y_data is None else y_data.shape)
    print("Features:", len(cols))
