"""Retraining orchestration pipeline for retrieval and ranking models."""

from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

from api.core.feature_store import FeatureStore
from ml.inference.adaptive_learning import save_adaptive_artifacts, train_adaptive_models
from ml.inference.embeddings_model import build_implicit_events_from_performance, save_embedding_artifacts, train_two_tower_if_available
from ml.training.evaluate_recommender import evaluate
from ml.training.model_registry import ModelRegistry
from ml.training.tune_recommender import main as tune_recommender_main

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "ml", "artifacts")


def run_retraining(canary: bool = False) -> dict:
    store = FeatureStore()
    snapshot = store.materialize_offline_training_snapshot(datetime.utcnow().isoformat())

    events_path = os.path.join(DATA_DIR, "events", "events.csv")
    if os.path.exists(events_path):
        events = pd.read_csv(events_path)
    else:
        events = pd.DataFrame(columns=["user_id", "item_id", "event_type"])

    performance_path = os.path.join(DATA_DIR, "raw", "Student_Performance.csv")
    if os.path.exists(performance_path):
        performance_df = pd.read_csv(performance_path)
        implicit_events = build_implicit_events_from_performance(performance_df)
    else:
        performance_df = pd.DataFrame()
        implicit_events = pd.DataFrame(columns=["user_id", "item_id", "interaction_weight", "source"])

    if not events.empty:
        events = events.rename(columns={"event_type": "source"}).copy()
        events["interaction_weight"] = events["source"].map({
            "recommendation_clicked": 2.0,
            "lesson_completed": 2.5,
            "rating_submitted": 1.5,
            "lesson_started": 0.8,
            "recommendation_shown": 0.5,
        }).fillna(1.0)
        events = events[["user_id", "item_id", "interaction_weight", "source"]]

    training_events = pd.concat([implicit_events, events], ignore_index=True, sort=False)

    embeddings = train_two_tower_if_available(training_events)
    embed_path = os.path.join(MODELS_DIR, "embeddings.joblib")
    save_embedding_artifacts(embed_path, embeddings)

    bandit, sequence_model = train_adaptive_models(training_events)
    adaptive_manifest = save_adaptive_artifacts(MODELS_DIR, bandit, sequence_model)

    dataset = performance_df if not performance_df.empty else pd.read_csv(os.path.join(DATA_DIR, "raw", "Student_Performance.csv"))
    metrics = evaluate(dataset, k=5)
    tune_recommender_main()

    registry = ModelRegistry(MODELS_DIR)
    version = registry.version()
    manifest = registry.write_manifest(
        version=version,
        artifacts={"embeddings": embed_path, **adaptive_manifest},
        metrics={"offline_reco": metrics, "snapshot_rows": int(len(snapshot)), "training_events": int(len(training_events)), "canary": canary},
        params={"stage": "retraining_orchestrator"},
    )

    result = {"version": version, "manifest": manifest, "canary": canary, "metrics": metrics}
    out_path = os.path.join(MODELS_DIR, "retraining_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(run_retraining(canary=False), indent=2))

