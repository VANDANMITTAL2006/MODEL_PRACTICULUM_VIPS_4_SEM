"""Async task definitions with Celery-compatible optional wrappers."""

from __future__ import annotations

from typing import Callable

try:
    from celery import Celery
except ImportError:  # pragma: no cover
    Celery = None


def _identity_task(fn: Callable):
    fn.delay = fn
    return fn


if Celery is not None:
    app = Celery("ml_tasks")
    app.conf.update(task_always_eager=True)

    def task_decorator(name: str):
        return app.task(name=name)
else:
    app = None

    def task_decorator(name: str):
        return _identity_task


@task_decorator("refresh_embeddings")
def refresh_embeddings() -> str:
    return "refresh_embeddings completed"


@task_decorator("rebuild_ann_index")
def rebuild_ann_index() -> str:
    return "rebuild_ann_index completed"


@task_decorator("retrain_ranker")
def retrain_ranker() -> str:
    return "retrain_ranker completed"


@task_decorator("backfill_features")
def backfill_features() -> str:
    return "backfill_features completed"

