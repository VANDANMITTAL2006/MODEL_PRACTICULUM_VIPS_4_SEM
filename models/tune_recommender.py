"""
Grid-search tuner for recommendation settings.

Searches KNN neighbors and distance metric, reports the best config
for collaborative and hybrid recommendation quality.

Run:
  python models/tune_recommender.py
"""

from __future__ import annotations

import os
import sys
from itertools import product

import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, BASE_DIR)

from models.evaluate_recommender import evaluate


def main() -> None:
    data_path = os.path.join(BASE_DIR, "data", "Student_Performance.csv")
    df = pd.read_csv(data_path)

    neighbor_grid = [3, 5, 8, 10, 15]
    metric_grid = ["euclidean", "manhattan", "cosine"]

    results = []

    for num_neighbors, metric in product(neighbor_grid, metric_grid):
        metrics = evaluate(df, k=5, num_neighbors=num_neighbors, metric=metric)
        results.append(
            {
                "num_neighbors": num_neighbors,
                "metric": metric,
                "collab_hit_at_5": metrics["collab_hit_at_k"],
                "hybrid_hit_at_5": metrics["hybrid_hit_at_k"],
                "collab_precision_at_5": metrics["collab_precision_at_k"],
                "hybrid_precision_at_5": metrics["hybrid_precision_at_k"],
                "hybrid_coverage": metrics["hybrid_coverage"],
            }
        )

    out_df = pd.DataFrame(results).sort_values(
        by=["hybrid_hit_at_5", "hybrid_precision_at_5"], ascending=False
    )

    best = out_df.iloc[0]

    print("=" * 70)
    print("Recommender Tuning Results (Top 10 by hybrid Hit@5)")
    print("=" * 70)
    print(out_df.head(10).to_string(index=False))
    print("=" * 70)
    print("Best configuration")
    print(f"  num_neighbors: {int(best['num_neighbors'])}")
    print(f"  metric:        {best['metric']}")
    print(f"  hybrid Hit@5:  {best['hybrid_hit_at_5']:.4f}")
    print(f"  hybrid Prec@5: {best['hybrid_precision_at_5']:.4f}")
    print(f"  hybrid cover.: {best['hybrid_coverage']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
