"""Caching layer with in-memory TTL fallback and optional Redis backend."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Optional

try:
    import redis
except ImportError:  # pragma: no cover
    redis = None


class TTLCache:
    def __init__(self, ttl_seconds: int = 120) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if item is None:
            return None
        if item["expires_at"] < time.time():
            self._store.pop(key, None)
            return None
        return item["value"]

    def set(self, key: str, value: Any) -> None:
        self._store[key] = {"value": value, "expires_at": time.time() + self.ttl_seconds}


class CacheClient:
    def __init__(self, ttl_seconds: int = 120) -> None:
        self.local = TTLCache(ttl_seconds=ttl_seconds)
        self.redis_client = None
        redis_url = os.getenv("REDIS_URL")
        if redis and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
            except Exception:
                self.redis_client = None

    @staticmethod
    def make_key(namespace: str, payload: Dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, default=str)
        digest = hashlib.md5(blob.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    def get(self, key: str) -> Optional[Any]:
        if self.redis_client:
            raw = self.redis_client.get(key)
            if raw:
                return json.loads(raw)
        return self.local.get(key)

    def set(self, key: str, value: Any, ttl_seconds: int = 120) -> None:
        if self.redis_client:
            self.redis_client.setex(key, ttl_seconds, json.dumps(value, default=str))
        self.local.set(key, value)
