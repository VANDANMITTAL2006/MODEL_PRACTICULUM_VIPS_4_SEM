"""Adaptive learning components for live recommendation updates."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    Dataset = object


def _clip_reward(value: float) -> float:
    return float(np.clip(value, 0.0, 1.5))


def event_reward(event_type: str, payload: Dict[str, object] | None = None) -> float:
    payload = payload or {}
    if event_type == "recommendation_clicked":
        return 1.0
    if event_type == "lesson_completed":
        return 1.25
    if event_type == "rating_submitted":
        rating = float(payload.get("rating", 3.0))
        return _clip_reward(rating / 5.0)
    if event_type == "lesson_started":
        return 0.35
    if event_type == "recommendation_shown":
        return 0.05
    return 0.0


def build_event_sequences(events: pd.DataFrame) -> List[List[str]]:
    if events.empty or "item_id" not in events.columns:
        return []

    frame = events.copy()
    frame = frame[frame["item_id"].notna()].copy()
    if frame.empty:
        return []

    frame["timestamp_utc"] = pd.to_datetime(frame.get("timestamp_utc"), utc=True, errors="coerce")
    sort_cols = ["user_id"]
    if "session_id" in frame.columns:
        sort_cols.append("session_id")
    sort_cols.append("timestamp_utc")
    frame = frame.sort_values(sort_cols)

    group_cols = ["user_id"]
    if "session_id" in frame.columns:
        group_cols.append("session_id")

    sequences: List[List[str]] = []
    for _, chunk in frame.groupby(group_cols, dropna=False):
        sequence = [str(item) for item in chunk["item_id"].astype(str).tolist() if item and item != "nan"]
        if len(sequence) >= 2:
            sequences.append(sequence)
    return sequences


class ContextualBanditPolicy:
    """LinUCB-style contextual bandit for recommendation arms."""

    def __init__(self, context_dim: int = 9, alpha: float = 0.7) -> None:
        self.context_dim = int(context_dim)
        self.alpha = float(alpha)
        self.arm_matrices: Dict[str, np.ndarray] = {}
        self.arm_targets: Dict[str, np.ndarray] = {}
        self.arm_counts: Dict[str, int] = {}

    def _ensure_arm(self, arm: str) -> None:
        if arm not in self.arm_matrices:
            self.arm_matrices[arm] = np.eye(self.context_dim, dtype=float)
            self.arm_targets[arm] = np.zeros(self.context_dim, dtype=float)
            self.arm_counts[arm] = 0

    def _context_vector(self, context: Dict[str, float]) -> np.ndarray:
        vec = np.array(
            [
                1.0,
                float(context.get("quiz_score", 50.0)) / 100.0,
                float(context.get("engagement_score", 50.0)) / 100.0,
                float(context.get("consistency_score", 50.0)) / 100.0,
                float(context.get("freshness", 0.5)),
                float(context.get("difficulty_match", 0.5)),
                float(context.get("popularity", 0.3)),
                float(context.get("sequence_score", 0.0)),
                float(context.get("online_score", 0.0)),
            ],
            dtype=float,
        )
        if vec.size != self.context_dim:
            if vec.size < self.context_dim:
                vec = np.pad(vec, (0, self.context_dim - vec.size), constant_values=0.0)
            else:
                vec = vec[: self.context_dim]
        return vec

    def score(self, arm: str, context: Dict[str, float]) -> float:
        self._ensure_arm(arm)
        x = self._context_vector(context)
        a = self.arm_matrices[arm]
        b = self.arm_targets[arm]
        a_inv = np.linalg.inv(a)
        theta = a_inv @ b
        exploit = float(theta @ x)
        explore = float(self.alpha * np.sqrt(max(x @ a_inv @ x, 0.0)))
        popularity_bonus = float(0.01 * self.arm_counts.get(arm, 0))
        return exploit + explore + popularity_bonus

    def score_candidates(self, candidate_contexts: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        return {arm: self.score(arm, context) for arm, context in candidate_contexts.items()}

    def update(self, arm: str, context: Dict[str, float], reward: float) -> None:
        self._ensure_arm(arm)
        x = self._context_vector(context)
        r = _clip_reward(reward)
        self.arm_matrices[arm] += np.outer(x, x)
        self.arm_targets[arm] += r * x
        self.arm_counts[arm] += 1

    def state_dict(self) -> Dict[str, object]:
        return {
            "context_dim": self.context_dim,
            "alpha": self.alpha,
            "arm_matrices": self.arm_matrices,
            "arm_targets": self.arm_targets,
            "arm_counts": self.arm_counts,
        }

    @classmethod
    def load(cls, path: str) -> "ContextualBanditPolicy":
        if not os.path.exists(path):
            return cls()
        payload = joblib.load(path)
        policy = cls(context_dim=int(payload.get("context_dim", 9)), alpha=float(payload.get("alpha", 0.7)))
        policy.arm_matrices = payload.get("arm_matrices", {})
        policy.arm_targets = payload.get("arm_targets", {})
        policy.arm_counts = payload.get("arm_counts", {})
        return policy

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.state_dict(), path)


class _SequenceDataset(Dataset):
    def __init__(self, samples: List[Tuple[List[int], int]], pad_idx: int, max_len: int) -> None:
        self.samples = samples
        self.pad_idx = pad_idx
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        prefix, target = self.samples[index]
        prefix = prefix[-self.max_len :]
        pad_len = self.max_len - len(prefix)
        input_ids = [self.pad_idx] * pad_len + prefix
        attention_mask = [0] * pad_len + [1] * len(prefix)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


class _NextItemTransformer(nn.Module if nn else object):
    def __init__(self, vocab_size: int, d_model: int = 48, nhead: int = 4, num_layers: int = 2, max_len: int = 16):
        if not nn:
            return
        super().__init__()
        self.item_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, input_ids, attention_mask=None):
        if not nn:
            return None
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden = self.item_embedding(input_ids) + self.pos_embedding(positions)
        padding_mask = attention_mask == 0 if attention_mask is not None else None
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        if attention_mask is None:
            pooled = encoded[:, -1, :]
        else:
            lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            pooled = encoded[torch.arange(encoded.size(0), device=encoded.device), lengths]
        pooled = self.norm(pooled)
        return self.head(pooled)


@dataclass
class SequenceModelState:
    vocab: Dict[str, int]
    item_frequency: Dict[str, int]
    bigram_counts: Dict[str, Dict[str, int]]
    max_len: int
    d_model: int
    nhead: int
    num_layers: int
    state_dict: Dict[str, object] | None = None


class SequenceTransformerRecommender:
    def __init__(self, max_len: int = 16, d_model: int = 48, nhead: int = 4, num_layers: int = 2) -> None:
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.vocab: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
        self.item_frequency: Dict[str, int] = {}
        self.bigram_counts: DefaultDict[str, Counter[str]] = defaultdict(Counter)
        self.model = None

    @property
    def pad_idx(self) -> int:
        return self.vocab["<pad>"]

    @property
    def unk_idx(self) -> int:
        return self.vocab["<unk>"]

    def _build_vocab(self, sequences: Sequence[Sequence[str]]) -> None:
        counts = Counter(item for seq in sequences for item in seq)
        self.item_frequency = dict(counts)
        for item in counts:
            if item not in self.vocab:
                self.vocab[item] = len(self.vocab)

    def _encode(self, items: Sequence[str]) -> List[int]:
        return [self.vocab.get(item, self.unk_idx) for item in items]

    def _training_pairs(self, sequences: Sequence[Sequence[str]]) -> List[Tuple[List[int], int]]:
        pairs: List[Tuple[List[int], int]] = []
        for seq in sequences:
            encoded = self._encode(seq)
            for idx in range(1, len(encoded)):
                prefix = encoded[max(0, idx - self.max_len) : idx]
                target = encoded[idx]
                if target != self.pad_idx:
                    pairs.append((prefix, target))
        return pairs

    def fit(self, sequences: Sequence[Sequence[str]]) -> "SequenceTransformerRecommender":
        sequences = [list(seq) for seq in sequences if len(seq) >= 2]
        self._build_vocab(sequences)
        for seq in sequences:
            for left, right in zip(seq[:-1], seq[1:]):
                self.bigram_counts[left][right] += 1

        if not sequences or torch is None:
            return self

        samples = self._training_pairs(sequences)
        if len(samples) < 4:
            return self

        dataset = _SequenceDataset(samples, self.pad_idx, self.max_len)
        loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
        self.model = _NextItemTransformer(len(self.vocab), d_model=self.d_model, nhead=self.nhead, num_layers=self.num_layers, max_len=self.max_len)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()

        for _ in range(6):
            for input_ids, attention_mask, target in loader:
                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = loss_fn(logits, target)
                loss.backward()
                optimizer.step()
        self.model.eval()
        return self

    def _transition_score(self, history: Sequence[str], candidate: str) -> float:
        if not history:
            return float(self.item_frequency.get(candidate, 0))
        last = history[-1]
        row = self.bigram_counts.get(last, {})
        if not row:
            return float(self.item_frequency.get(candidate, 0))
        return float(row.get(candidate, 0))

    def score_candidates(self, history: Sequence[str], candidates: Sequence[str]) -> Dict[str, float]:
        if not candidates:
            return {}

        history = [str(item) for item in history if item and item != "nan"][-self.max_len :]
        fallback = {candidate: self._transition_score(history, candidate) for candidate in candidates}

        if self.model is None or torch is None or not history:
            denom = max(sum(max(score, 0.0) for score in fallback.values()), 1e-6)
            return {candidate: float(max(score, 0.0) / denom) for candidate, score in fallback.items()}

        encoded = self._encode(history)
        pad_len = self.max_len - len(encoded)
        input_ids = torch.tensor([[self.pad_idx] * pad_len + encoded], dtype=torch.long)
        attention_mask = torch.tensor([[0] * pad_len + [1] * len(encoded)], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)[0].detach().cpu().numpy()

        scores: Dict[str, float] = {}
        candidate_logits = []
        for candidate in candidates:
            idx = self.vocab.get(candidate, self.unk_idx)
            candidate_logits.append(float(logits[idx]))
        if candidate_logits:
            shifted = np.asarray(candidate_logits, dtype=float)
            shifted = shifted - shifted.max(initial=0.0)
            probs = np.exp(shifted)
            probs = probs / max(probs.sum(), 1e-6)
            for candidate, prob in zip(candidates, probs):
                scores[candidate] = float(prob)
        else:
            scores = {candidate: 0.0 for candidate in candidates}

        max_fallback = max(fallback.values()) if fallback else 1.0
        denom = max(max_fallback, 1e-6)
        for candidate in candidates:
            scores[candidate] = 0.75 * scores.get(candidate, 0.0) + 0.25 * float(fallback.get(candidate, 0.0) / denom)
        return scores

    def state_dict(self) -> Dict[str, object]:
        return {
            "vocab": self.vocab,
            "item_frequency": self.item_frequency,
            "bigram_counts": {key: dict(counter) for key, counter in self.bigram_counts.items()},
            "max_len": self.max_len,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "state_dict": self.model.state_dict() if self.model is not None and torch is not None else None,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if torch is not None:
            torch.save(self.state_dict(), path)
        else:
            joblib.dump(self.state_dict(), path)

    @classmethod
    def load(cls, path: str) -> "SequenceTransformerRecommender":
        recommender = cls()
        if not os.path.exists(path):
            return recommender
        payload = torch.load(path, map_location="cpu") if torch is not None else joblib.load(path)
        recommender.vocab = payload.get("vocab", recommender.vocab)
        recommender.item_frequency = payload.get("item_frequency", {})
        recommender.bigram_counts = defaultdict(Counter, {key: Counter(value) for key, value in payload.get("bigram_counts", {}).items()})
        recommender.max_len = int(payload.get("max_len", recommender.max_len))
        recommender.d_model = int(payload.get("d_model", recommender.d_model))
        recommender.nhead = int(payload.get("nhead", recommender.nhead))
        recommender.num_layers = int(payload.get("num_layers", recommender.num_layers))
        state_dict = payload.get("state_dict")
        if torch is not None and state_dict:
            recommender.model = _NextItemTransformer(len(recommender.vocab), d_model=recommender.d_model, nhead=recommender.nhead, num_layers=recommender.num_layers, max_len=recommender.max_len)
            recommender.model.load_state_dict(state_dict)
            recommender.model.eval()
        return recommender


def train_adaptive_models(events: pd.DataFrame) -> Tuple[ContextualBanditPolicy, SequenceTransformerRecommender]:
    bandit = ContextualBanditPolicy()
    if not events.empty and {"user_id", "item_id"}.issubset(events.columns):
        frame = events.copy()
        frame["reward"] = frame.apply(lambda row: event_reward(str(row.get("event_type", "")), row.get("payload", {})), axis=1)
        frame["timestamp_utc"] = pd.to_datetime(frame.get("timestamp_utc"), utc=True, errors="coerce")
        frame = frame.sort_values(["user_id", "timestamp_utc"])
        item_counts = frame.groupby("item_id")["reward"].agg(["mean", "count"]).fillna(0)
        item_counts["popularity"] = (item_counts["count"] / max(float(item_counts["count"].sum()), 1.0)).astype(float)

        for row in frame.itertuples(index=False):
            item_id = str(getattr(row, "item_id", ""))
            if not item_id or item_id == "nan":
                continue
            context = {
                "quiz_score": float(getattr(row, "quiz_score", 50.0)) if hasattr(row, "quiz_score") else 50.0,
                "engagement_score": float(getattr(row, "engagement_score", 50.0)) if hasattr(row, "engagement_score") else 50.0,
                "consistency_score": float(getattr(row, "consistency_score", 50.0)) if hasattr(row, "consistency_score") else 50.0,
                "freshness": 0.5,
                "difficulty_match": 0.5,
                "popularity": float(item_counts.loc[item_id, "popularity"]) if item_id in item_counts.index else 0.0,
                "sequence_score": 0.0,
                "online_score": 0.0,
            }
            bandit.update(item_id, context, float(getattr(row, "reward", 0.0)))

    sequence_model = SequenceTransformerRecommender()
    sequence_model.fit(build_event_sequences(events))
    return bandit, sequence_model


def save_adaptive_artifacts(directory: str, bandit: ContextualBanditPolicy, sequence_model: SequenceTransformerRecommender) -> Dict[str, str]:
    os.makedirs(directory, exist_ok=True)
    bandit_path = os.path.join(directory, "adaptive_bandit.joblib")
    sequence_path = os.path.join(directory, "adaptive_sequence.pt")
    bandit.save(bandit_path)
    sequence_model.save(sequence_path)
    manifest = {"bandit": bandit_path, "sequence": sequence_path}
    with open(os.path.join(directory, "adaptive_artifacts.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_adaptive_artifacts(directory: str) -> Tuple[ContextualBanditPolicy, SequenceTransformerRecommender]:
    bandit_path = os.path.join(directory, "adaptive_bandit.joblib")
    sequence_path = os.path.join(directory, "adaptive_sequence.pt")
    bandit = ContextualBanditPolicy.load(bandit_path)
    sequence_model = SequenceTransformerRecommender.load(sequence_path)
    return bandit, sequence_model