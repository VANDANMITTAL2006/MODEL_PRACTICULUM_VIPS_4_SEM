"""Inference for clean user-input performance prediction."""

from __future__ import annotations

import os
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "ml", "artifacts", "clean_user_input_model")


class UserInputPerformancePredictor:
    def __init__(self, artifact_dir: str = DEFAULT_ARTIFACT_DIR) -> None:
        self.artifact_dir = artifact_dir
        self.model = joblib.load(os.path.join(artifact_dir, "model.pkl"))
        self.preprocessor = joblib.load(os.path.join(artifact_dir, "preprocessor.pkl"))
        self.feature_cols = joblib.load(os.path.join(artifact_dir, "feature_cols.pkl"))

    @staticmethod
    def _risk_level(score: float) -> str:
        if score < 50:
            return "high"
        if score <= 70:
            return "medium"
        return "low"

    def predict_user_performance(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        user_df = pd.DataFrame([user_input])
        missing_cols = [col for col in self.feature_cols if col not in user_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required user input fields: {missing_cols}")

        user_df = user_df[self.feature_cols]
        transformed = self.preprocessor.transform(user_df)

        score = float(np.clip(self.model.predict(transformed)[0], 0, 100))
        return {
            "predicted_score": score,
            "risk_level": self._risk_level(score),
        }


_predictor_cache: UserInputPerformancePredictor | None = None


def predict_user_performance(user_input: Dict[str, Any], artifact_dir: str = DEFAULT_ARTIFACT_DIR) -> Dict[str, Any]:
    global _predictor_cache
    if _predictor_cache is None or _predictor_cache.artifact_dir != artifact_dir:
        _predictor_cache = UserInputPerformancePredictor(artifact_dir=artifact_dir)
    return _predictor_cache.predict_user_performance(user_input)
