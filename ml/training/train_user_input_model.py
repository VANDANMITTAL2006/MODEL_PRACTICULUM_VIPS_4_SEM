"""Train an interpretable score model using only user-input features."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Student_Performance.csv")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "ml", "artifacts", "clean_user_input_model")

TARGET_COL = "final_score"
FEATURE_COLS = [
    "quiz_score",
    "time_spent_hours",
    "attendance",
    "engagement_score",
    "consistency_score",
    "previous_score",
    "subject_weakness",
]


def inspect_dataset_columns(csv_path: str = DATA_PATH) -> List[str]:
    frame = pd.read_csv(csv_path)
    return frame.columns.tolist()


def _build_preprocessor(feature_cols: List[str], frame: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = [col for col in feature_cols if is_numeric_dtype(frame[col])]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols


def _validate_columns(frame: pd.DataFrame, feature_cols: List[str], target_col: str) -> None:
    missing_features = [col for col in feature_cols if col not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing selected feature columns in dataset: {missing_features}")
    if target_col not in frame.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")


def train_user_input_model(
    csv_path: str = DATA_PATH,
    artifact_dir: str = ARTIFACT_DIR,
    feature_cols: List[str] | None = None,
    target_col: str = TARGET_COL,
    random_state: int = 42,
) -> Dict[str, Any]:
    frame = pd.read_csv(csv_path)
    selected_features = feature_cols or FEATURE_COLS

    _validate_columns(frame, selected_features, target_col)

    model_frame = frame[selected_features + [target_col]].copy()
    model_frame = model_frame.dropna(subset=[target_col])

    x = model_frame[selected_features]
    y = model_frame[target_col].astype(float)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    preprocessor, numeric_cols, categorical_cols = _build_preprocessor(selected_features, model_frame)

    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    model = RandomForestRegressor(
        n_estimators=350,
        max_depth=10,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train_processed, y_train)

    preds = np.clip(model.predict(x_test_processed), 0, 100)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }

    os.makedirs(artifact_dir, exist_ok=True)

    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]

    joblib.dump(model, os.path.join(artifact_dir, "model.pkl"))
    joblib.dump(preprocessor, os.path.join(artifact_dir, "preprocessor.pkl"))
    joblib.dump(scaler, os.path.join(artifact_dir, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(artifact_dir, "encoder.pkl"))
    joblib.dump(selected_features, os.path.join(artifact_dir, "feature_cols.pkl"))

    summary = {
        "dataset_path": csv_path,
        "all_dataset_columns": frame.columns.tolist(),
        "target_col": target_col,
        "selected_features": selected_features,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "metrics": metrics,
        "excluded_columns": [col for col in frame.columns if col not in selected_features + [target_col]],
    }

    with open(os.path.join(artifact_dir, "training_summary.json"), "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    return summary


if __name__ == "__main__":
    result = train_user_input_model()
    print(json.dumps(result, indent=2))
