"""Tune recommender parameters with Optuna and fallback search."""

from __future__ import annotations

import json
import os
import sys
from itertools import product

import pandas as pd

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml.training.evaluate_recommender import evaluate


def _objective(df: pd.DataFrame):
    def fn(trial):
        num_neighbors = trial.suggest_int("num_neighbors", 3, 20)
        metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "cosine"])
        k = trial.suggest_int("k", 3, 8)
        metrics = evaluate(df, k=k, num_neighbors=num_neighbors, metric=metric)
        score = (
            0.30 * metrics["hit_at_k"]
            + 0.20 * metrics["ndcg_at_k"]
            + 0.15 * metrics["map_at_k"]
            + 0.15 * metrics["coverage"]
            + 0.15 * metrics.get("cold_start_hit_at_k", 0.0)
            + 0.05 * metrics.get("struggling_hit_at_k", 0.0)
        )
        return score

    return fn


def _fallback_search(df: pd.DataFrame) -> pd.DataFrame:
    neighbor_grid = [3, 5, 8, 10, 15]
    metric_grid = ["euclidean", "manhattan", "cosine"]
    k_grid = [3, 5, 7]
    rows = []
    for num_neighbors, metric, k in product(neighbor_grid, metric_grid, k_grid):
        metrics = evaluate(df, k=k, num_neighbors=num_neighbors, metric=metric)
        rows.append({"num_neighbors": num_neighbors, "metric": metric, "k": k, **metrics})
    return pd.DataFrame(rows).sort_values(by=["hit_at_k", "cold_start_hit_at_k", "ndcg_at_k", "map_at_k"], ascending=False)


def main() -> None:
    data_path = os.path.join(PROJECT_ROOT, "data", "raw", "Student_Performance.csv")
    df = pd.read_csv(data_path)

    if optuna is not None:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(_objective(df), n_trials=30)
        best = study.best_trial.params
        final_metrics = evaluate(df, k=int(best["k"]), num_neighbors=int(best["num_neighbors"]), metric=str(best["metric"]))
        rows = [{**best, **final_metrics, "objective": study.best_value}]
        out_df = pd.DataFrame(rows)
    else:
        out_df = _fallback_search(df)

    out_dir = os.path.join(PROJECT_ROOT, "ml", "artifacts")
    out_df.to_csv(os.path.join(out_dir, "recommender_tuning_results.csv"), index=False)
    best_row = out_df.iloc[0].to_dict()
    with open(os.path.join(out_dir, "recommender_tuning_best.json"), "w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2, default=str)

    print(json.dumps(best_row, indent=2, default=str))


if __name__ == "__main__":
    main()

