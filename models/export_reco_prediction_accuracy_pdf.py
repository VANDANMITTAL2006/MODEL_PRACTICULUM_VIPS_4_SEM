"""
Generate combined Prediction + Recommendation accuracy report as PDF.

Output:
    Desktop/reco_prediction_accuracy_report.pdf
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, BASE_DIR)

from data.data_pipeline import load_data, preprocess
from models.evaluate_recommender import evaluate as evaluate_recommender


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def within_tolerance_accuracy(y_true, y_pred, tol: float = 10.0) -> float:
    """Share of predictions within +- tol score points."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) <= tol))


def evaluate_prediction_models(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    X, y, _, _, _ = preprocess(df.copy(), fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    xgb_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    xgb_base = XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
    grid = GridSearchCV(
        xgb_base,
        xgb_param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    xgb = grid.best_estimator_
    xgb_pred = xgb.predict(X_test)

    pred_metrics = {
        "Random Forest": {
            "MAE": float(mean_absolute_error(y_test, rf_pred)),
            "RMSE": rmse(y_test, rf_pred),
            "R2": float(r2_score(y_test, rf_pred)),
            "Within10": within_tolerance_accuracy(y_test, rf_pred, tol=10.0),
        },
        "XGBoost": {
            "MAE": float(mean_absolute_error(y_test, xgb_pred)),
            "RMSE": rmse(y_test, xgb_pred),
            "R2": float(r2_score(y_test, xgb_pred)),
            "Within10": within_tolerance_accuracy(y_test, xgb_pred, tol=10.0),
        },
        "XGBoost Best Params": grid.best_params_,
    }
    return pred_metrics


def build_pdf(output_path: str, pred: Dict[str, Dict[str, float]], reco: Dict[str, float]) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Recommendation + Prediction Accuracy Report",
    )

    styles = getSampleStyleSheet()
    title = styles["Title"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    body.leading = 14
    small = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=10, leading=13)

    story = []

    story.append(Paragraph("Recommendation + Prediction Accuracy Report", title))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("Dataset: Student_Performance.csv | Evaluation split: 80/20 | K=5 for recommendations", small))
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("1) Prediction Model Metrics (Regression)", h2))
    story.append(Paragraph("Lower MAE/RMSE is better. Higher R2 and Within+-10 accuracy are better.", body))

    for model_name in ["Random Forest", "XGBoost"]:
        m = pred[model_name]
        story.append(Paragraph(f"{model_name}", body))
        points = [
            f"MAE = {m['MAE']:.4f}",
            f"RMSE = {m['RMSE']:.4f}",
            f"R2 = {m['R2']:.4f}",
            f"Within+-10 accuracy = {m['Within10']:.4f}",
        ]
        story.append(ListFlowable([ListItem(Paragraph(p, small)) for p in points], bulletType="bullet", leftIndent=14))
        story.append(Spacer(1, 0.08 * cm))

    best_params = pred.get("XGBoost Best Params", {})
    story.append(Paragraph(f"XGBoost best parameters (GridSearchCV): {best_params}", small))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("How prediction metrics are calculated", h2))
    pred_formula = [
        "MAE = mean(|y_true - y_pred|)",
        "RMSE = sqrt(mean((y_true - y_pred)^2))",
        "R2 = 1 - (sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2))",
        "Within+-10 accuracy = fraction of samples where |error| <= 10",
    ]
    story.append(ListFlowable([ListItem(Paragraph(p, small)) for p in pred_formula], bulletType="bullet", leftIndent=14))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("2) Recommendation Metrics", h2))
    reco_points = [
        f"Content weak-subject alignment (sanity check) = {reco['content_weak_subject_alignment']:.4f}",
        f"Content difficulty-fit score (strict) = {reco['content_difficulty_fit']:.4f}",
        f"Collaborative Hit@5 = {reco['collab_hit_at_k']:.4f}",
        f"Collaborative Precision@5 = {reco['collab_precision_at_k']:.4f}",
        f"Collaborative Recall@5 = {reco['collab_recall_at_k']:.4f}",
        f"Collaborative Coverage = {reco['collab_coverage']:.4f}",
        f"Hybrid Hit@5 = {reco['hybrid_hit_at_k']:.4f}",
        f"Hybrid Precision@5 = {reco['hybrid_precision_at_k']:.4f}",
        f"Hybrid Recall@5 = {reco['hybrid_recall_at_k']:.4f}",
        f"Hybrid Coverage = {reco['hybrid_coverage']:.4f}",
    ]
    story.append(ListFlowable([ListItem(Paragraph(p, small)) for p in reco_points], bulletType="bullet", leftIndent=14))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("How recommendation metrics are calculated", h2))
    reco_formula = [
        "Hit@K = 1 if at least one relevant item appears in top-K, else 0 (averaged over users)",
        "Precision@K = (# relevant in top-K) / K (averaged)",
        "Recall@K = (# relevant in top-K) / (# relevant items) (averaged)",
        "Coverage = (# unique recommended topics) / (# total topics)",
        "Content difficulty-fit = fraction of recommended weak-subject topics that match expected difficulty band from quiz score",
    ]
    story.append(ListFlowable([ListItem(Paragraph(p, small)) for p in reco_formula], bulletType="bullet", leftIndent=14))

    doc.build(story)


def main() -> None:
    df = load_data(os.path.join(BASE_DIR, "data", "Student_Performance.csv"))

    pred_metrics = evaluate_prediction_models(df)
    reco_metrics = evaluate_recommender(df, k=5)

    output_pdf = r"C:\Users\SRISHTI MITTAL\OneDrive\Desktop\reco_prediction_accuracy_report.pdf"
    build_pdf(output_pdf, pred_metrics, reco_metrics)

    print("Combined report generated.")
    print(f"PDF path: {output_pdf}")


if __name__ == "__main__":
    main()
