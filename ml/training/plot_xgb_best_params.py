"""
Create a simple bar chart for the tuned XGBoost best parameters.
Saves the figure to the Desktop as a PNG.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

BEST_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 4,
    "n_estimators": 100,
    "subsample": 0.8,
}


def main() -> None:
    labels = list(BEST_PARAMS.keys())
    values = list(BEST_PARAMS.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#2563eb", "#16a34a", "#f59e0b", "#ef4444"])

    ax.set_title("XGBoost Best Parameters from GridSearchCV", fontsize=14, weight="bold")
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Selected Value")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    note = "Note: values are the selected tuning settings, not directly comparable metric scores."
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    output_path = r"C:\Users\SRISHTI MITTAL\OneDrive\Desktop\xgb_best_params_graph.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved graph to: {output_path}")


if __name__ == "__main__":
    main()
