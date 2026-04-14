"""Feature store with online Redis/in-memory access and offline snapshots."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from api.core.cache import CacheClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EVENT_LOG_DIR = os.path.join(DATA_DIR, "events")
os.makedirs(EVENT_LOG_DIR, exist_ok=True)


class FeatureStore:
    def __init__(self) -> None:
        self.cache = CacheClient(ttl_seconds=3600)
        self.user_namespace = "feature:user"
        self.item_namespace = "feature:item"

    def get_online_user_features(self, user_id: str) -> Dict[str, Any]:
        key = f"{self.user_namespace}:{user_id}"
        return self.cache.get(key) or {}

    def get_online_item_features(self, item_id: str) -> Dict[str, Any]:
        key = f"{self.item_namespace}:{item_id}"
        return self.cache.get(key) or {}

    def write_online_features(self, entity: str, values: Dict[str, Any], ttl: int = 3600) -> None:
        entity_type = entity.split(":", 1)[0]
        if entity_type not in {"user", "item"}:
            raise ValueError("entity must start with 'user:' or 'item:'")
        namespace = self.user_namespace if entity_type == "user" else self.item_namespace
        key = f"{namespace}:{entity.split(':', 1)[1]}"
        current = self.cache.get(key) or {}
        current.update(values)
        self.cache.set(key, current, ttl_seconds=ttl)

    def get_recent_user_sequence(self, user_id: str, limit: int = 20) -> list[str]:
        events_path = os.path.join(EVENT_LOG_DIR, "events.csv")
        if not os.path.exists(events_path):
            return []

        frame = pd.read_csv(events_path)
        if frame.empty or "user_id" not in frame.columns or "item_id" not in frame.columns:
            return []

        frame = frame[frame["user_id"].astype(str) == str(user_id)].copy()
        frame = frame[frame["item_id"].notna()]
        if frame.empty:
            return []

        if "timestamp_utc" in frame.columns:
            frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
            frame = frame.sort_values("timestamp_utc")
        sequence = frame["item_id"].astype(str).tolist()
        return sequence[-limit:]

    def materialize_offline_training_snapshot(self, as_of_timestamp: str) -> pd.DataFrame:
        as_of = pd.to_datetime(as_of_timestamp, utc=True)
        events_path = os.path.join(EVENT_LOG_DIR, "events.csv")
        if not os.path.exists(events_path):
            return pd.DataFrame()

        df = pd.read_csv(events_path)
        if df.empty:
            return df

        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df[df["timestamp_utc"] <= as_of]
        if df.empty:
            return df

        # Point-in-time safe aggregation: only use interactions up to as_of.
        features = (
            df.groupby("user_id", as_index=False)
            .agg(
                interactions=("event_type", "count"),
                clicks=("event_type", lambda s: int((s == "recommendation_clicked").sum())),
                completions=("event_type", lambda s: int((s == "lesson_completed").sum())),
                last_event_ts=("timestamp_utc", "max"),
            )
        )
        features["ctr"] = features["clicks"] / features["interactions"].clip(lower=1)
        features["completion_rate"] = features["completions"] / features["interactions"].clip(lower=1)

        out_path = os.path.join(EVENT_LOG_DIR, f"snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
        features.to_csv(out_path, index=False)
        return features


def _serialize_payload(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=str)

