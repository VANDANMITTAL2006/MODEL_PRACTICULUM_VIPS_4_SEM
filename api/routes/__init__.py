"""Structured logging and lightweight latency tracing."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict


LOGGER_NAME = "ml_backend"


def configure_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, event: str, payload: Dict[str, Any]) -> None:
    logger.info(json.dumps({"event": event, **payload}, default=str))


@contextmanager
def timed_span(logger: logging.Logger, name: str, context: Dict[str, Any] | None = None):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        log_event(logger, "timing", {"span": name, "latency_ms": round(elapsed_ms, 3), **(context or {})})

