"""Feedback ingestion pipeline with idempotency and dual-write storage."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict

import pandas as pd

from api.core.event_schema import EventBase
from api.core.feature_store import EVENT_LOG_DIR, FeatureStore


class FeedbackLoop:
    def __init__(self, store: FeatureStore | None = None, adaptive_callback=None) -> None:
        self.store = store or FeatureStore()
        self.adaptive_callback = adaptive_callback
        self.events_path = os.path.join(EVENT_LOG_DIR, "events.csv")
        self.dedup_path = os.path.join(EVENT_LOG_DIR, "dedup_keys.txt")
        os.makedirs(EVENT_LOG_DIR, exist_ok=True)

    def _known_keys(self) -> set[str]:
        if not os.path.exists(self.dedup_path):
            return set()
        with open(self.dedup_path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    def _persist_key(self, key: str) -> None:
        with open(self.dedup_path, "a", encoding="utf-8") as f:
            f.write(key + "\n")

    def ingest(self, event: EventBase) -> Dict[str, object]:
        key = event.resolved_idempotency_key()
        keys = self._known_keys()
        if key in keys:
            return {"accepted": False, "deduplicated": True, "idempotency_key": key}

        row = event.model_dump()
        row["idempotency_key"] = key
        row["ingested_at_utc"] = datetime.utcnow().isoformat()
        self._append_event_row(row)
        self._persist_key(key)
        self._update_online_features(event)
        if self.adaptive_callback is not None:
            self.adaptive_callback(event)
        return {"accepted": True, "deduplicated": False, "idempotency_key": key}

    def _append_event_row(self, row: Dict[str, object]) -> None:
        frame = pd.DataFrame([row])
        if os.path.exists(self.events_path):
            frame.to_csv(self.events_path, mode="a", index=False, header=False)
        else:
            frame.to_csv(self.events_path, index=False)

    def _update_online_features(self, event: EventBase) -> None:
        user_key = f"user:{event.user_id}"
        user_feats = self.store.get_online_user_features(event.user_id)
        interactions = int(user_feats.get("interactions", 0)) + 1
        clicks = int(user_feats.get("clicks", 0)) + int(event.event_type == "recommendation_clicked")
        completions = int(user_feats.get("completions", 0)) + int(event.event_type == "lesson_completed")
        rating = event.payload.get("rating") if event.event_type == "rating_submitted" else user_feats.get("latest_rating")

        self.store.write_online_features(
            entity=user_key,
            values={
                "interactions": interactions,
                "clicks": clicks,
                "completions": completions,
                "ctr": clicks / max(interactions, 1),
                "completion_rate": completions / max(interactions, 1),
                "latest_rating": rating,
                "last_event_type": event.event_type,
                "last_event_ts": event.timestamp_utc,
            },
            ttl=7200,
        )

        if event.item_id:
            item_key = f"item:{event.item_id}"
            item_feats = self.store.get_online_item_features(event.item_id)
            item_views = int(item_feats.get("views", 0)) + int(event.event_type == "recommendation_shown")
            item_clicks = int(item_feats.get("clicks", 0)) + int(event.event_type == "recommendation_clicked")
            self.store.write_online_features(
                entity=item_key,
                values={
                    "views": item_views,
                    "clicks": item_clicks,
                    "item_ctr": item_clicks / max(item_views, 1),
                    "last_event_ts": event.timestamp_utc,
                },
                ttl=7200,
            )

