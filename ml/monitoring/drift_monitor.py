"""Drift and online quality monitoring utilities."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd


def psi(train: pd.Series, current: pd.Series, bins: int = 10) -> float:
    t = train.to_numpy(dtype=float)
    c = current.to_numpy(dtype=float)
    edges = np.histogram_bin_edges(t, bins=bins)
    t_hist, _ = np.histogram(t, bins=edges)
    c_hist, _ = np.histogram(c, bins=edges)
    t_ratio = np.maximum(t_hist / max(t_hist.sum(), 1), 1e-6)
    c_ratio = np.maximum(c_hist / max(c_hist.sum(), 1), 1e-6)
    return float(np.sum((c_ratio - t_ratio) * np.log(c_ratio / t_ratio)))


class OnlineMetricsTracker:
    def __init__(self) -> None:
        self.latencies: List[float] = []
        self.cache_hits = 0
        self.requests = 0
        self.ctr_events = defaultdict(int)

    def record_request(self, latency_ms: float, cache_hit: bool) -> None:
        self.requests += 1
        self.latencies.append(latency_ms)
        self.cache_hits += int(cache_hit)

    def record_feedback(self, event_type: str) -> None:
        self.ctr_events[event_type] += 1

    def snapshot(self) -> Dict[str, float]:
        arr = sorted(self.latencies) if self.latencies else [0.0]
        def p(q: float) -> float:
            idx = int(q * (len(arr) - 1))
            return float(arr[idx])

        shown = max(self.ctr_events.get("recommendation_shown", 0), 1)
        clicks = self.ctr_events.get("recommendation_clicked", 0)
        completed = self.ctr_events.get("lesson_completed", 0)
        return {
            "latency_p50_ms": p(0.5),
            "latency_p95_ms": p(0.95),
            "latency_p99_ms": p(0.99),
            "cache_hit_rate": self.cache_hits / max(self.requests, 1),
            "ctr": clicks / shown,
            "completion_at_k_proxy": completed / shown,
        }


def emit_metrics(path: str, metrics: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"created_at_utc": datetime.utcnow().isoformat(), **metrics}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
