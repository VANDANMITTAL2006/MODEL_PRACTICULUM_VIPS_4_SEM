"""Validation utilities for robust regression evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, learning_curve


@dataclass
class FoldMetrics:
    rmse: float
    mae: float
    r2: float
    mape: float
    calibration_error: float


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    true_bins = pd.qcut(y_true, q=min(10, len(y_true)), duplicates="drop")
    calib = pd.DataFrame({"bin": true_bins, "y": y_true, "p": y_pred}).groupby("bin", observed=True).mean(numeric_only=True)
    calibration_error = float(np.mean(np.abs(calib["y"] - calib["p"]))) if not calib.empty else 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "calibration_error": calibration_error,
    }


def repeated_kfold_cv(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 2,
    random_state: int = 42,
) -> Dict[str, float]:
    splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    rows: List[Dict[str, float]] = []

    for train_idx, valid_idx in splitter.split(X, y):
        est = clone(estimator)
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        est.fit(X_train, y_train)
        pred = est.predict(X_valid)
        rows.append(regression_metrics(y_valid.values, pred))

    frame = pd.DataFrame(rows)
    return {f"cv_{col}_mean": float(frame[col].mean()) for col in frame.columns} | {
        f"cv_{col}_std": float(frame[col].std()) for col in frame.columns
    }


def nested_cv_score(
    estimator_factory,
    X: pd.DataFrame,
    y: pd.Series,
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, float]:
    outer = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        est = estimator_factory(X_train, y_train, inner_splits, random_state + fold)
        pred = est.predict(X_test)
        row = regression_metrics(y_test.values, pred)
        row["fold"] = fold
        fold_rows.append(row)

    frame = pd.DataFrame(fold_rows)
    return {
        "nested_rmse_mean": float(frame["rmse"].mean()),
        "nested_mae_mean": float(frame["mae"].mean()),
        "nested_r2_mean": float(frame["r2"].mean()),
        "nested_mape_mean": float(frame["mape"].mean()),
    }


def segment_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segments: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    frame = pd.DataFrame({"y": y_true, "p": y_pred, "segment": list(segments)})
    results: Dict[str, Dict[str, float]] = {}
    for seg, chunk in frame.groupby("segment"):
        results[str(seg)] = regression_metrics(chunk["y"].values, chunk["p"].values)
    return results


def learning_curve_diagnostics(estimator, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator,
        X,
        y,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 6),
    )
    train_rmse = -train_scores.mean(axis=1)
    valid_rmse = -valid_scores.mean(axis=1)
    return {
        "learning_curve_train_rmse_last": float(train_rmse[-1]),
        "learning_curve_valid_rmse_last": float(valid_rmse[-1]),
        "train_val_gap_last": float(valid_rmse[-1] - train_rmse[-1]),
    }


def conformal_interval(
    y_true_calib: np.ndarray,
    y_pred_calib: np.ndarray,
    y_pred_target: np.ndarray,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    residuals = np.abs(np.asarray(y_true_calib) - np.asarray(y_pred_calib))
    q = float(np.quantile(residuals, 1.0 - alpha))
    pred = np.asarray(y_pred_target)
    return pred - q, pred + q

