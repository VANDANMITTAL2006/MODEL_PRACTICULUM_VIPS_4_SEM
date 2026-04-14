"""Simple model registry and manifest utilities."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional

import joblib


class ModelRegistry:
    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def version(self) -> str:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def save_artifact(self, obj: Any, base_name: str, version: str) -> str:
        file_name = f"{base_name}_{version}.pkl"
        path = os.path.join(self.model_dir, file_name)
        joblib.dump(obj, path)
        return path

    def _git_hash(self) -> Optional[str]:
        try:
            value = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            return value.decode("utf-8").strip()
        except Exception:
            return None

    def write_manifest(self, version: str, artifacts: Dict[str, str], metrics: Dict[str, Any], params: Dict[str, Any]) -> str:
        manifest = {
            "version": version,
            "created_at_utc": datetime.utcnow().isoformat(),
            "git_hash": self._git_hash(),
            "artifacts": artifacts,
            "metrics": metrics,
            "params": params,
        }
        path = os.path.join(self.model_dir, f"manifest_{version}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        latest_path = os.path.join(self.model_dir, "manifest_latest.json")
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return path


def load_latest_manifest(model_dir: str) -> Dict[str, Any]:
    path = os.path.join(model_dir, "manifest_latest.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
