"""Stacking ensemble helpers."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold


MODEL_ORDER = ["xgb", "lgbm", "catboost", "rf"]


def generate_oof_predictions(base_models: Dict[str, object], X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = {name: np.zeros(len(X), dtype=float) for name in base_models}

    for train_idx, valid_idx in cv.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        for name, model in base_models.items():
            m = model.__class__(**model.get_params())
            m.fit(X_train, y_train)
            oof[name][valid_idx] = m.predict(X_valid)

    oof_frame = pd.DataFrame({k: oof[k] for k in MODEL_ORDER if k in oof})
    return oof_frame


def train_stacking_regressor(base_models: Dict[str, object], X: pd.DataFrame, y: pd.Series) -> Tuple[StackingRegressor, pd.DataFrame]:
    estimators = [(name, model) for name, model in base_models.items()]
    meta = RidgeCV(alphas=(0.1, 1.0, 3.0, 10.0))
    stack = StackingRegressor(estimators=estimators, final_estimator=meta, passthrough=True, n_jobs=-1)
    stack.fit(X, y)

    oof_frame = generate_oof_predictions(base_models, X, y)
    return stack, oof_frame


def blend_predict(base_models: Dict[str, object], stack_model, X: pd.DataFrame) -> np.ndarray:
    if stack_model is not None:
        return stack_model.predict(X)
    preds = [model.predict(X) for model in base_models.values()]
    return np.mean(np.vstack(preds), axis=0)
