"""Feature and training data monitoring helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd


def summarize_feature_distribution(X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for col in X.columns:
        series = X[col]
        summary[col] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
            "null_rate": float(series.isna().mean()),
        }
    return summary


def population_stability_index(train_col: pd.Series, prod_col: pd.Series, bins: int = 10) -> float:
    train_vals = train_col.to_numpy(dtype=float)
    prod_vals = prod_col.to_numpy(dtype=float)
    edges = np.histogram_bin_edges(train_vals, bins=bins)
    train_hist, _ = np.histogram(train_vals, bins=edges)
    prod_hist, _ = np.histogram(prod_vals, bins=edges)

    train_ratio = np.maximum(train_hist / max(train_hist.sum(), 1), 1e-6)
    prod_ratio = np.maximum(prod_hist / max(prod_hist.sum(), 1), 1e-6)
    return float(np.sum((prod_ratio - train_ratio) * np.log(prod_ratio / train_ratio)))


def save_monitoring_report(path: str, X_train: pd.DataFrame, X_reference: pd.DataFrame | None = None) -> None:
    report = {
        "created_at_utc": datetime.utcnow().isoformat(),
        "feature_summary": summarize_feature_distribution(X_train),
        "psi": {},
    }

    if X_reference is not None:
        common_cols = [c for c in X_train.columns if c in X_reference.columns]
        for col in common_cols:
            report["psi"][col] = population_stability_index(X_train[col], X_reference[col])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
