"""Deterministic experiment assignment and guardrail monitoring."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class GuardrailState:
    latencies_ms: List[float]
    failures: int
    requests: int
    low_engagement_drops: int


class ABTestingManager:
    def __init__(self, experiment_name: str = "reco_v3", split: int = 50) -> None:
        self.experiment_name = experiment_name
        self.split = max(1, min(split, 99))
        self.state = GuardrailState(latencies_ms=[], failures=0, requests=0, low_engagement_drops=0)

    def assign_bucket(self, user_id: str) -> str:
        digest = hashlib.md5(f"{self.experiment_name}:{user_id}".encode("utf-8")).hexdigest()
        bucket_num = int(digest[:8], 16) % 100
        return "treatment" if bucket_num < self.split else "control"

    def record_observation(self, latency_ms: float, success: bool, low_engagement_drop: bool = False) -> None:
        self.state.requests += 1
        self.state.latencies_ms.append(latency_ms)
        if not success:
            self.state.failures += 1
        if low_engagement_drop:
            self.state.low_engagement_drops += 1

    def guardrail_metrics(self) -> Dict[str, float]:
        n = max(self.state.requests, 1)
        latencies = sorted(self.state.latencies_ms) if self.state.latencies_ms else [0.0]
        p95_index = int(0.95 * (len(latencies) - 1))
        return {
            "failure_rate": self.state.failures / n,
            "latency_p95_ms": float(latencies[p95_index]),
            "low_engagement_drop_rate": self.state.low_engagement_drops / n,
        }
