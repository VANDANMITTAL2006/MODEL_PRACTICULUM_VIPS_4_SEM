"""Event schema contracts for recommendation lifecycle logging."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal[
    "recommendation_shown",
    "recommendation_clicked",
    "lesson_started",
    "lesson_completed",
    "rating_submitted",
]


class EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: EventType
    user_id: str
    item_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    payload: Dict[str, Any] = Field(default_factory=dict)
    idempotency_key: Optional[str] = None

    def resolved_idempotency_key(self) -> str:
        if self.idempotency_key:
            return self.idempotency_key
        material = {
            "event_type": self.event_type,
            "user_id": self.user_id,
            "item_id": self.item_id,
            "session_id": self.session_id,
            "timestamp_utc": self.timestamp_utc,
            "payload": self.payload,
        }
        raw = json.dumps(material, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def coerce_event(raw: Dict[str, Any]) -> EventBase:
    return EventBase(**raw)
