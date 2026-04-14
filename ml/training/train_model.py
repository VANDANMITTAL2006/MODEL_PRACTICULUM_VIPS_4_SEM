"""Production training pipeline with Optuna tuning and stacked ensemble."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.data.data_pipeline import engineer_features, handle_missing_values, load_data, preprocess
from ml.training.ensemble import train_stacking_regressor
from ml.monitoring.feature_monitoring import save_monitoring_report
from ml.training.model_registry import ModelRegistry
from ml.training.optuna_tuning import tune_catboost, tune_lightgbm, tune_xgboost
from ml.training.validation import conformal_interval, learning_curve_diagnostics, regression_metrics, repeated_kfold_cv, segment_metrics

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "ml", "artifacts")
DATA_OUTPUT = os.path.join(PROJECT_ROOT, "data", "raw", "Student_Performance.csv")


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _fit_with_early_stopping(model, X_train, y_train, X_valid, y_valid):
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
    except TypeError:
        model.fit(X_train, y_train)
    return model


def _save_importance_reports(best_model, X_test: pd.DataFrame, y_test: pd.Series, out_dir: str) -> None:
    perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    frame = pd.DataFrame({
        "feature": X_test.columns,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False)
    frame.to_csv(os.path.join(out_dir, "feature_importance_permutation.csv"), index=False)

    try:
        import shap

        explainer = shap.Explainer(best_model, X_test)
        shap_values = explainer(X_test)
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        shap_df = pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(os.path.join(out_dir, "feature_importance_shap.csv"), index=False)
    except Exception:
        fallback = pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": np.nan})
        fallback.to_csv(os.path.join(out_dir, "feature_importance_shap.csv"), index=False)


def _select_feature_subset(model, X_valid: pd.DataFrame, y_valid: pd.Series, max_features: int = 16) -> list[str]:
    importance = permutation_importance(model, X_valid, y_valid, n_repeats=5, random_state=42, n_jobs=-1)
    frame = pd.DataFrame({"feature": X_valid.columns, "importance": importance.importances_mean})
    frame = frame.sort_values("importance", ascending=False)
    selected = frame[frame["importance"] > 0]["feature"].tolist()[:max_features]
    if len(selected) < min(max_features, len(frame)):
        selected = frame["feature"].tolist()[:max_features]
    return selected


def train() -> Dict[str, float]:
    df = load_data()
    metadata_path = os.path.join(MODELS_DIR, "preprocessing_metadata.json")
    X, y, encoders, scaler, feature_cols = preprocess(df, fit=True, metadata_path=metadata_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_params, xgb = tune_xgboost(X_train, y_train)
    xgb = _fit_with_early_stopping(xgb, X_train, y_train, X_test, y_test)

    lgbm_params, lgbm = tune_lightgbm(X_train, y_train)
    if lgbm is not None:
        lgbm = _fit_with_early_stopping(lgbm, X_train, y_train, X_test, y_test)

    cat_params, cat = tune_catboost(X_train, y_train)
    if cat is not None:
        cat = _fit_with_early_stopping(cat, X_train, y_train, X_test, y_test)

    rf = RandomForestRegressor(n_estimators=400, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    extra_trees = ExtraTreesRegressor(n_estimators=700, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1)
    extra_trees.fit(X_train, y_train)

    gbr = GradientBoostingRegressor(random_state=42, learning_rate=0.05, n_estimators=400, max_depth=3)
    gbr.fit(X_train, y_train)

    hgb = HistGradientBoostingRegressor(random_state=42, learning_rate=0.05, max_depth=8, max_iter=400)
    hgb.fit(X_train, y_train)

    base_models = {"xgb": xgb, "rf": rf, "extra_trees": extra_trees, "gbr": gbr, "hgb": hgb}
    if lgbm is not None:
        base_models["lgbm"] = lgbm
    if cat is not None:
        base_models["catboost"] = cat

    stack_model, oof = train_stacking_regressor(base_models, X_train, y_train)

    model_predictions = {}
    model_metrics_rows = []
    for name, model in base_models.items():
        pred = np.clip(model.predict(X_test), 0, 100)
        model_predictions[name] = pred
        metrics = regression_metrics(y_test.values, pred)
        model_metrics_rows.append({"model": name, **metrics})

    stack_pred = np.clip(stack_model.predict(X_test), 0, 100)
    stack_metrics = regression_metrics(y_test.values, stack_pred)
    model_metrics_rows.append({"model": "stacking", **stack_metrics})

    comparison_df = pd.DataFrame(model_metrics_rows).sort_values("rmse")
    comparison_df.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)

    best_row = comparison_df.iloc[0]
    best_model_name = str(best_row["model"])
    best_model = stack_model if best_model_name == "stacking" else base_models[best_model_name]

    selected_features = feature_cols
    selected_scaler = scaler
    best_metrics = {"rmse": float(best_row["rmse"]), "r2": float(best_row["r2"])}
    if best_model_name != "stacking" and len(feature_cols) > 8:
        candidate_features = _select_feature_subset(best_model, X_test, y_test, max_features=min(16, len(feature_cols)))
        if candidate_features:
            candidate_model = clone(best_model)
            candidate_model.fit(X_train[candidate_features], y_train)
            candidate_pred = np.clip(candidate_model.predict(X_test[candidate_features]), 0, 100)
            candidate_metrics = regression_metrics(y_test.values, candidate_pred)
            if candidate_metrics["rmse"] <= float(best_row["rmse"]):
                best_model = candidate_model
                best_model_name = f"{best_model_name}_selected"
                selected_features = candidate_features
                selected_scaler = StandardScaler().fit(X_train[selected_features])
                best_metrics = {"rmse": float(candidate_metrics["rmse"]), "r2": float(candidate_metrics["r2"])}

    X_train_best = X_train[selected_features] if selected_features != feature_cols else X_train
    X_test_best = X_test[selected_features] if selected_features != feature_cols else X_test
    X_selected = X[selected_features] if selected_features != feature_cols else X

    if selected_features != feature_cols:
        y_pred_calib = best_model.predict(X_train_best)
        y_pred_target = best_model.predict(X_test_best)
    else:
        y_pred_calib = best_model.predict(X_train)
        y_pred_target = stack_pred if best_model_name == "stacking" else model_predictions[best_model_name]
    lower, upper = conformal_interval(y_train.values, y_pred_calib, y_pred_target)

    seg_series = pd.cut(df.loc[y_test.index, "quiz_score"], bins=[-1, 55, 75, 101], labels=["struggling", "mid", "fast"])
    seg_metrics = segment_metrics(y_test.values, stack_pred, seg_series.astype(str).tolist())

    diagnostics = {
        "best_model": best_model_name,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "xgb_params": xgb_params,
        "lgbm_params": lgbm_params,
        "catboost_params": cat_params,
        "repeated_cv": repeated_kfold_cv(best_model, X_selected, y),
        "learning_curve": learning_curve_diagnostics(best_model, X_selected, y),
        "segment_metrics": seg_metrics,
        "uncertainty_mean_width": float(np.mean(upper - lower)),
        "selected_features": selected_features,
    }

    with open(os.path.join(MODELS_DIR, "training_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    _save_importance_reports(best_model, X_test_best, y_test, MODELS_DIR)
    save_monitoring_report(os.path.join(MODELS_DIR, "feature_monitoring.json"), X_train_best, X_test_best)

    registry = ModelRegistry(MODELS_DIR)
    version = registry.version()
    artifacts = {
        "model": registry.save_artifact(best_model, "model", version),
        "stack_model": registry.save_artifact(stack_model, "stack_model", version),
        "rf_model": registry.save_artifact(rf, "rf_model", version),
        "extra_trees_model": registry.save_artifact(extra_trees, "extra_trees_model", version),
        "gbr_model": registry.save_artifact(gbr, "gbr_model", version),
        "hgb_model": registry.save_artifact(hgb, "hgb_model", version),
        "scaler": registry.save_artifact(selected_scaler, "scaler", version),
        "encoders": registry.save_artifact(encoders, "encoders", version),
        "feature_cols": registry.save_artifact(selected_features, "feature_cols", version),
        "oof_predictions": registry.save_artifact(oof, "oof_predictions", version),
    }
    if lgbm is not None:
        artifacts["lgbm_model"] = registry.save_artifact(lgbm, "lgbm_model", version)
    if cat is not None:
        artifacts["catboost_model"] = registry.save_artifact(cat, "catboost_model", version)
    artifacts["xgb_model"] = registry.save_artifact(xgb, "xgb_model", version)

    df_raw = engineer_features(handle_missing_values(load_data()))
    cluster_features = ["engagement_score", "consistency_score", "learning_efficiency"]
    x_cluster = df_raw[cluster_features].fillna(0)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(x_cluster)

    centers = pd.DataFrame(kmeans.cluster_centers_, columns=cluster_features)
    centers["cluster_id"] = centers.index
    sorted_centers = centers.sort_values("engagement_score", ascending=False)
    mapping = {}
    labels = ["Fast Learner", "Low Engagement", "Struggling Learner"]
    for idx, (_, row) in enumerate(sorted_centers.iterrows()):
        mapping[int(row["cluster_id"])] = labels[idx]

    segment_labels = [mapping[label] for label in kmeans.labels_]
    df_raw["cluster_label"] = segment_labels
    full_pred = np.clip(best_model.predict(X_selected), 0, 100)
    df_raw["predicted_score"] = full_pred
    df_raw.to_csv(DATA_OUTPUT, index=False)

    artifacts["kmeans"] = registry.save_artifact(kmeans, "kmeans", version)
    artifacts["cluster_mapping"] = registry.save_artifact(mapping, "cluster_mapping", version)

    # Backward-compatible latest aliases.
    joblib.dump(best_model, os.path.join(MODELS_DIR, "model.pkl"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "rf_model.pkl"))
    joblib.dump(extra_trees, os.path.join(MODELS_DIR, "extra_trees_model.pkl"))
    joblib.dump(gbr, os.path.join(MODELS_DIR, "gbr_model.pkl"))
    joblib.dump(hgb, os.path.join(MODELS_DIR, "hgb_model.pkl"))
    joblib.dump(selected_scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans.pkl"))
    joblib.dump(mapping, os.path.join(MODELS_DIR, "cluster_mapping.pkl"))
    joblib.dump(selected_features, os.path.join(MODELS_DIR, "feature_cols.pkl"))

    manifest_path = registry.write_manifest(
        version=version,
        artifacts=artifacts,
        metrics={"stacking": stack_metrics, "rmse_best": float(best_metrics["rmse"]), "trained_at_utc": datetime.utcnow().isoformat()},
        params={"xgb": xgb_params, "lgbm": lgbm_params, "catboost": cat_params},
    )

    return {
        "best_model": best_model_name,
        "rmse": best_metrics["rmse"],
        "r2": best_metrics["r2"],
        "manifest": manifest_path,
    }


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))

