"""Generate a complete ML workflow and accuracy PDF report with graphs.

Outputs:
- Desktop/full_ml_workflow_accuracy_report.pdf
- project/ml/artifacts/report_assets/*.png
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from ml.data.data_pipeline import load_data, preprocess
from ml.training.evaluate_recommender import evaluate as evaluate_recommender

ASSET_DIR = os.path.join(BASE_DIR, "ml", "artifacts", "report_assets")
os.makedirs(ASSET_DIR, exist_ok=True)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _within_tolerance_accuracy(y_true, y_pred, tol: float = 10.0) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) <= tol))


def evaluate_prediction_models(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    X, y, _, _, _ = preprocess(df.copy(), fit=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    grid = GridSearchCV(xgb_base, xgb_param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    xgb = grid.best_estimator_
    xgb_pred = xgb.predict(X_test)

    return {
        "Random Forest": {
            "MAE": float(mean_absolute_error(y_test, rf_pred)),
            "RMSE": _rmse(y_test, rf_pred),
            "R2": float(r2_score(y_test, rf_pred)),
            "Within10": _within_tolerance_accuracy(y_test, rf_pred, tol=10.0),
        },
        "XGBoost": {
            "MAE": float(mean_absolute_error(y_test, xgb_pred)),
            "RMSE": _rmse(y_test, xgb_pred),
            "R2": float(r2_score(y_test, xgb_pred)),
            "Within10": _within_tolerance_accuracy(y_test, xgb_pred, tol=10.0),
        },
        "XGBoost Best Params": grid.best_params_,
    }


def evaluate_previous_prediction_if_available() -> Optional[Dict[str, Dict[str, float]]]:
    prev_base = os.path.join(BASE_DIR, "legacy", "project")
    if not os.path.exists(prev_base):
        return None

    prev_data = os.path.join(prev_base, "data", "Student_Performance.csv")
    prev_enc = os.path.join(prev_base, "models", "encoders.pkl")
    prev_scaler = os.path.join(prev_base, "models", "scaler.pkl")
    prev_model = os.path.join(prev_base, "models", "model.pkl")
    prev_rf = os.path.join(prev_base, "models", "rf_model.pkl")
    prev_pipeline_file = os.path.join(prev_base, "data", "data_pipeline.py")

    required = [prev_data, prev_enc, prev_scaler, prev_model, prev_rf, prev_pipeline_file]
    if not all(os.path.exists(p) for p in required):
        return None

    import importlib.util

    spec = importlib.util.spec_from_file_location("prev_data_pipeline", prev_pipeline_file)
    if spec is None or spec.loader is None:
        return None
    prev_pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prev_pipeline)

    df_prev = prev_pipeline.load_data(prev_data)
    enc = joblib.load(prev_enc)
    scaler = joblib.load(prev_scaler)
    model = joblib.load(prev_model)
    rf = joblib.load(prev_rf)

    X, y, _, _, _ = prev_pipeline.preprocess(df_prev.copy(), encoders=enc, scaler=scaler, fit=False)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_pred = model.predict(X_test)
    rf_pred = rf.predict(X_test)

    return {
        "Random Forest": {
            "MAE": float(mean_absolute_error(y_test, rf_pred)),
            "RMSE": _rmse(y_test, rf_pred),
            "R2": float(r2_score(y_test, rf_pred)),
            "Within10": _within_tolerance_accuracy(y_test, rf_pred),
        },
        "XGBoost": {
            "MAE": float(mean_absolute_error(y_test, xgb_pred)),
            "RMSE": _rmse(y_test, xgb_pred),
            "R2": float(r2_score(y_test, xgb_pred)),
            "Within10": _within_tolerance_accuracy(y_test, xgb_pred),
        },
    }


def make_prediction_graph(pred_metrics: Dict[str, Dict[str, float]]) -> str:
    models = ["Random Forest", "XGBoost"]
    mae = [pred_metrics[m]["MAE"] for m in models]
    rmse = [pred_metrics[m]["RMSE"] for m in models]
    r2 = [pred_metrics[m]["R2"] for m in models]
    within = [pred_metrics[m]["Within10"] for m in models]

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    axs = axs.ravel()

    axs[0].bar(models, mae, color=["#4e79a7", "#f28e2b"])
    axs[0].set_title("MAE (lower is better)")
    axs[0].grid(axis="y", alpha=0.2)

    axs[1].bar(models, rmse, color=["#4e79a7", "#f28e2b"])
    axs[1].set_title("RMSE (lower is better)")
    axs[1].grid(axis="y", alpha=0.2)

    axs[2].bar(models, r2, color=["#4e79a7", "#f28e2b"])
    axs[2].set_title("R2 (higher is better)")
    axs[2].grid(axis="y", alpha=0.2)

    axs[3].bar(models, within, color=["#4e79a7", "#f28e2b"])
    axs[3].set_title("Within +/-10 Accuracy (higher is better)")
    axs[3].set_ylim(0, 1)
    axs[3].grid(axis="y", alpha=0.2)

    plt.suptitle("Prediction Model Accuracy Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    out = os.path.join(ASSET_DIR, "prediction_metrics_grid.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def make_reco_graph(reco_metrics: Dict[str, float]) -> str:
    keys = ["hit_at_k", "precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k", "coverage", "diversity", "novelty"]
    labels = ["Hit@5", "Precision@5", "Recall@5", "NDCG@5", "MAP@5", "Coverage", "Diversity", "Novelty"]
    values = [float(reco_metrics[k]) for k in keys]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(labels, values, color="#59a14f")
    ax.set_ylim(0, 1.05)
    ax.set_title("Recommendation Metrics")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=20)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = os.path.join(ASSET_DIR, "recommendation_metrics_bar.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def make_delta_graph(current_pred: Dict[str, Dict[str, float]], previous_pred: Optional[Dict[str, Dict[str, float]]]) -> Optional[str]:
    if not previous_pred:
        return None

    models = ["Random Forest", "XGBoost"]
    r2_delta = [current_pred[m]["R2"] - previous_pred[m]["R2"] for m in models]
    rmse_delta = [current_pred[m]["RMSE"] - previous_pred[m]["RMSE"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, r2_delta, width, label="R2 delta (current-prev)", color="#e15759")
    ax.bar(x + width / 2, rmse_delta, width, label="RMSE delta (current-prev)", color="#76b7b2")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title("Prediction Delta vs Previous Baseline")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    out = os.path.join(ASSET_DIR, "prediction_delta_vs_previous.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _table_from_rows(rows: List[List[str]]) -> Table:
    table = Table(rows, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdbdbd")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def build_pdf(
    output_path: str,
    pred_metrics: Dict[str, Dict[str, float]],
    reco_metrics: Dict[str, float],
    previous_pred: Optional[Dict[str, Dict[str, float]]],
    graph_pred: str,
    graph_reco: str,
    graph_delta: Optional[str],
) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Complete ML Workflow and Accuracy Report",
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]
    body = styles["BodyText"]
    body.leading = 14
    small = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=9.5, leading=12)

    story = []
    story.append(Paragraph("Complete ML Workflow and Accuracy Report", title_style))
    story.append(Paragraph("AI Personalized Learning System", h3))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("1) End-to-End Workflow", h2))
    workflow_points = [
        "Data ingestion: Student_Performance.csv is loaded from project/data.",
        "Preprocessing: strict schema checks, missing-value handling, categorical encoding, scaling.",
        "Feature engineering: engagement_score, consistency_score, learning_efficiency and interaction/history features.",
        "Prediction training: tuned tree models plus ensemble comparison, then best model selected.",
        "Segmentation: KMeans clustering produces learner segments for personalization.",
        "Recommendation: hybrid of content-based and collaborative ranking.",
        "Evaluation: prediction metrics (MAE, RMSE, R2, Within+/-10) and recommendation ranking metrics.",
        "Artifacts: trained models, encoders, scaler, feature list, and evaluation outputs are persisted in ml/artifacts/.",
    ]
    story.append(ListFlowable([ListItem(Paragraph(p, body)) for p in workflow_points], bulletType="bullet", leftIndent=14))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("2) ML Models and Algorithms", h2))
    model_points = [
        "Prediction models: Random Forest and XGBoost regressors.",
        "Advanced training pipeline also supports LightGBM, CatBoost, and stacking.",
        "Segmentation model: KMeans (k=3) over engagement and consistency-related features.",
        "Recommendation system: hybrid strategy combining content-based and nearest-neighbor collaborative filtering.",
        "Deployment artifacts include model.pkl, rf_model.pkl, encoders.pkl, scaler.pkl, kmeans.pkl, and feature_cols.pkl.",
    ]
    story.append(ListFlowable([ListItem(Paragraph(p, body)) for p in model_points], bulletType="bullet", leftIndent=14))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("3) Prediction Accuracy Details", h2))
    pred_rows = [["Model", "MAE", "RMSE", "R2", "Within+/-10"]]
    for model_name in ["Random Forest", "XGBoost"]:
        m = pred_metrics[model_name]
        pred_rows.append([
            model_name,
            f"{m['MAE']:.4f}",
            f"{m['RMSE']:.4f}",
            f"{m['R2']:.4f}",
            f"{m['Within10']:.4f}",
        ])
    story.append(_table_from_rows(pred_rows))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Image(graph_pred, width=17.0 * cm, height=11.0 * cm))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("Metric formulas:", h3))
    formula_points = [
        "MAE = mean absolute error.",
        "RMSE = square root of mean squared error.",
        "R2 = proportion of variance explained by model.",
        "Within+/-10 = fraction of predictions where absolute error <= 10.",
    ]
    story.append(ListFlowable([ListItem(Paragraph(p, small)) for p in formula_points], bulletType="bullet", leftIndent=14))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("4) Recommendation Accuracy Details", h2))
    reco_rows = [["Metric", "Value"]]
    reco_key_labels = [
        ("hit_at_k", "Hit@5"),
        ("precision_at_k", "Precision@5"),
        ("recall_at_k", "Recall@5"),
        ("ndcg_at_k", "NDCG@5"),
        ("map_at_k", "MAP@5"),
        ("coverage", "Coverage"),
        ("diversity", "Diversity"),
        ("novelty", "Novelty"),
        ("cold_start_hit_at_k", "Cold-start Hit@5"),
        ("struggling_hit_at_k", "Struggling Segment Hit@5"),
    ]
    for key, label in reco_key_labels:
        if key in reco_metrics:
            reco_rows.append([label, f"{float(reco_metrics[key]):.4f}"])

    story.append(_table_from_rows(reco_rows))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Image(graph_reco, width=17.0 * cm, height=8.0 * cm))

    if previous_pred is not None:
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph("5) Comparison with Previous Prediction Baseline", h2))
        prev_rows = [["Model", "Prev R2", "Now R2", "Delta R2", "Prev RMSE", "Now RMSE", "Delta RMSE"]]
        for model_name in ["Random Forest", "XGBoost"]:
            p = previous_pred[model_name]
            c = pred_metrics[model_name]
            prev_rows.append([
                model_name,
                f"{p['R2']:.4f}",
                f"{c['R2']:.4f}",
                f"{(c['R2'] - p['R2']):+.4f}",
                f"{p['RMSE']:.4f}",
                f"{c['RMSE']:.4f}",
                f"{(c['RMSE'] - p['RMSE']):+.4f}",
            ])
        story.append(_table_from_rows(prev_rows))
        if graph_delta:
            story.append(Spacer(1, 0.2 * cm))
            story.append(Image(graph_delta, width=16.0 * cm, height=8.0 * cm))

    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph("6) Interpretation Summary", h2))
    summary_points = [
        "Prediction quality is moderate-to-good when R2 is around 0.65 to 0.70.",
        "Recommendation quality currently has high catalog coverage/diversity but low precision/recall.",
        "System is usable end-to-end, with recommendation accuracy as the main improvement area.",
    ]
    story.append(ListFlowable([ListItem(Paragraph(p, body)) for p in summary_points], bulletType="bullet", leftIndent=14))

    doc.build(story)


def main() -> None:
    data_path = os.path.join(BASE_DIR, "data", "raw", "Student_Performance.csv")
    df = load_data(data_path)

    pred_metrics = evaluate_prediction_models(df)
    reco_metrics = evaluate_recommender(df, k=5)
    previous_pred = evaluate_previous_prediction_if_available()

    graph_pred = make_prediction_graph(pred_metrics)
    graph_reco = make_reco_graph(reco_metrics)
    graph_delta = make_delta_graph(pred_metrics, previous_pred)

    desktop_pdf = r"C:\Users\SRISHTI MITTAL\OneDrive\Desktop\full_ml_workflow_accuracy_report.pdf"
    build_pdf(
        output_path=desktop_pdf,
        pred_metrics=pred_metrics,
        reco_metrics=reco_metrics,
        previous_pred=previous_pred,
        graph_pred=graph_pred,
        graph_reco=graph_reco,
        graph_delta=graph_delta,
    )

    payload = {
        "pdf": desktop_pdf,
        "prediction": pred_metrics,
        "recommendation": reco_metrics,
        "previous_prediction": previous_pred,
        "assets": {
            "prediction_graph": graph_pred,
            "recommendation_graph": graph_reco,
            "delta_graph": graph_delta,
        },
    }
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

