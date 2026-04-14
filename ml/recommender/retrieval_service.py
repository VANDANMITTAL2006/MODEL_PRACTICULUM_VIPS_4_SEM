"""Embedding-based candidate retrieval with fallback to existing generators."""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from ml.inference.embeddings_model import ann_query, build_ann_index, load_embedding_artifacts
from ml.recommender.candidate_generation import build_candidates
from ml.recommender.cold_start import cold_start_recommend

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EMBED_PATH = os.path.join(PROJECT_ROOT, "ml", "artifacts", "embeddings.joblib")


class RetrievalService:
    def __init__(self) -> None:
        self.artifacts = load_embedding_artifacts(EMBED_PATH)
        self.index_pack = None
        if self.artifacts is not None:
            self.index_pack = build_ann_index(self.artifacts.item_vectors)

    def retrieve(
        self,
        user_id: str,
        subject_weakness: str,
        quiz_score: float,
        student_features: Dict[str, float],
        df: pd.DataFrame,
        top_k: int = 20,
    ) -> Dict[str, List[str]]:
        if self.artifacts and user_id in self.artifacts.user_vectors and self.index_pack is not None:
            query_vec = np.asarray(self.artifacts.user_vectors[user_id], dtype="float32")
            ann_items = [item_id for item_id, _ in ann_query(self.index_pack, query_vec, top_k=top_k)]
            if ann_items:
                return {"ann": ann_items, "source": "ann"}

        if not self.artifacts or user_id not in self.artifacts.user_vectors:
            cold = cold_start_recommend(subject_weakness, quiz_score, num_topics=top_k)
            return {"ann": cold.get("recommended_topics", [])[:top_k], "source": cold.get("source", "cold_start")}

        fallback = build_candidates(
            subject_weakness=subject_weakness,
            quiz_score=quiz_score,
            student_features=student_features,
            df=df,
        )
        merged = list(dict.fromkeys(fallback.get("content", []) + fallback.get("collaborative", [])))
        return {"ann": merged[:top_k], "source": "fallback"}

