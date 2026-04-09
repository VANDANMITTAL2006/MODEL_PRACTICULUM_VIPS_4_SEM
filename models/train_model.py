"""
ML Model Training: RandomForest + XGBoost, KMeans clustering, hyperparameter tuning.
Run this script to generate model files in /models.

Usage:
  python models/train_model.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from xgboost import XGBRegressor

# Adjust path so we can import data_pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.data_pipeline import load_data, preprocess, NUMERIC_FEATURES, CATEGORICAL_COLS

MODELS_DIR = os.path.dirname(__file__)

# ─────────────────────── helpers ────────────────────────

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ─────────────────────── main ───────────────────────────

def train():
    print("=" * 60)
    print("  AI Personalized Learning — Model Training")
    print("=" * 60)

    # 1. Load & preprocess
    df = load_data()
    print(f"\n📊 Loaded dataset: {df.shape[0]} students, {df.shape[1]} features")

    X, y, encoders, scaler, feature_cols = preprocess(df, fit=True)
    print(f"✅ Features after preprocessing: {len(feature_cols)}")

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n📌 Train size: {len(X_train)}, Test size: {len(X_test)}")

    # ── A. Random Forest ──────────────────────────────────
    print("\n[1/2] Training RandomForestRegressor …")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_rmse = rmse(y_test, rf_pred)
    rf_r2   = r2_score(y_test, rf_pred)
    print(f"   RMSE: {rf_rmse:.3f}  |  R²: {rf_r2:.4f}")

    cv_rf = cross_val_score(rf, X, y, cv=5, scoring="r2")
    print(f"   CV R²: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")

    # ── B. XGBoost (primary) ─────────────────────────────
    print("\n[2/2] Training XGBoostRegressor (with GridSearch) …")
    xgb_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    xgb_base = XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
    grid_search = GridSearchCV(
        xgb_base, xgb_param_grid,
        cv=3, scoring="r2", n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    xgb = grid_search.best_estimator_
    print(f"   Best params: {grid_search.best_params_}")

    xgb_pred = xgb.predict(X_test)
    xgb_rmse = rmse(y_test, xgb_pred)
    xgb_r2   = r2_score(y_test, xgb_pred)
    print(f"   RMSE: {xgb_rmse:.3f}  |  R²: {xgb_r2:.4f}")

    cv_xgb = cross_val_score(xgb, X, y, cv=5, scoring="r2")
    print(f"   CV R²: {cv_xgb.mean():.4f} ± {cv_xgb.std():.4f}")

    # ── C. KMeans Clustering ─────────────────────────────
    print("\n🔵 Training KMeans clustering (student segmentation) …")
    # Use engagement & consistency for clustering
    cluster_features = ["engagement_score", "consistency_score", "learning_efficiency"]
    # Reload raw & reprocess to get unscaled cluster features
    df_raw = load_data()
    from data.data_pipeline import handle_missing_values, engineer_features
    df_raw = handle_missing_values(df_raw)
    df_raw = engineer_features(df_raw)
    X_cluster = df_raw[cluster_features].fillna(0)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    labels = kmeans.labels_

    # ── Map cluster IDs to human labels ─────────────────
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=cluster_features)
    # rank by mean engagement (higher = fast learner)
    centers["cluster_id"] = centers.index
    sorted_centers = centers.sort_values("engagement_score", ascending=False)
    mapping = {}
    category_names = ["Fast Learner", "Low Engagement", "Struggling Learner"]
    for rank, (_, row) in enumerate(sorted_centers.iterrows()):
        mapping[int(row["cluster_id"])] = category_names[rank]

    segment_labels = [mapping[l] for l in labels]
    print(f"   Cluster distribution: { {v: segment_labels.count(v) for v in set(segment_labels)} }")

    # ── Save artefacts ────────────────────────────────────
    print("\n💾 Saving model artefacts …")
    joblib.dump(xgb, os.path.join(MODELS_DIR, "model.pkl"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "rf_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans.pkl"))
    joblib.dump(mapping, os.path.join(MODELS_DIR, "cluster_mapping.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))

    # Save student segment labels back to CSV
    df_raw["cluster_label"] = segment_labels
    df_raw["predicted_score"] = xgb.predict(X)
    df_raw.to_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "Student_Performance.csv"),
        index=False
    )

    print("\n✅ All artefacts saved:")
    for fname in ["model.pkl", "rf_model.pkl", "scaler.pkl", "encoders.pkl",
                  "kmeans.pkl", "cluster_mapping.pkl", "feature_cols.pkl"]:
        fpath = os.path.join(MODELS_DIR, fname)
        print(f"   {fname}  ({os.path.getsize(fpath)//1024} KB)")

    print("\n🎉 Training complete!")
    return {
        "xgb_rmse": xgb_rmse,
        "xgb_r2": xgb_r2,
        "rf_rmse": rf_rmse,
        "rf_r2": rf_r2,
    }


if __name__ == "__main__":
    train()
