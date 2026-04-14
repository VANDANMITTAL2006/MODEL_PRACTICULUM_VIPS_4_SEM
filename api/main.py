"""FastAPI backend with production-grade recommendation orchestration."""

from __future__ import annotations

import datetime
import os
import sys
import time
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "artifacts")
sys.path.insert(0, BASE_DIR)

from api.core.ab_testing import ABTestingManager
from api.core.cache import CacheClient
from api.core.event_schema import EventBase
from api.core.feature_store import FeatureStore
from api.core.feedback_loop import FeedbackLoop
from api.core.observability import configure_logger, log_event, timed_span
from api.services.recommendation_engine import TOPIC_LIBRARY, content_based_recommend, hybrid_recommend
from api.schemas.contracts import AnalyzeUserRequest, FeedbackEventRequest, PredictRequest, QuizUpdateRequest, RecommendRequest, SCHEMA_VERSION
from ml.inference.user_input_predictor import predict_user_performance
from ml.data.data_pipeline import CATEGORICAL_COLS, NUMERIC_FEATURES, engineer_features, handle_missing_values
from ml.monitoring.drift_monitor import OnlineMetricsTracker, emit_metrics
from ml.training.model_registry import load_latest_manifest
from ml.recommender.adaptive import AdaptiveRecommender
from ml.recommender.ranker import rerank_diversity_novelty
from ml.recommender.ranking_service import RankingService
from ml.recommender.retrieval_service import RetrievalService


def _load_artifacts() -> Dict:
    loaded = {
        "model": None,
        "scaler": None,
        "encoders": None,
        "kmeans": None,
        "cluster_map": None,
        "feature_cols": None,
        "model_version": "legacy",
    }
    manifest = load_latest_manifest(MODEL_DIR)
    if manifest:
        loaded["model_version"] = manifest.get("version", "legacy")
        artifacts = manifest.get("artifacts", {})
        key_map = {
            "model": "model",
            "scaler": "scaler",
            "encoders": "encoders",
            "kmeans": "kmeans",
            "cluster_mapping": "cluster_map",
            "feature_cols": "feature_cols",
        }
        for artifact_key, local_key in key_map.items():
            path = artifacts.get(artifact_key)
            if path and os.path.exists(path):
                loaded[local_key] = joblib.load(path)

    if loaded["model"] is None:
        try:
            loaded["model"] = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
            loaded["scaler"] = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
            loaded["encoders"] = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
            loaded["kmeans"] = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
            loaded["cluster_map"] = joblib.load(os.path.join(MODEL_DIR, "cluster_mapping.pkl"))
            loaded["feature_cols"] = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
        except FileNotFoundError:
            pass
    return loaded


ARTEFACTS = _load_artifacts()
LOGGER = configure_logger()
CACHE = CacheClient(ttl_seconds=180)
FEATURE_STORE = FeatureStore()
ADAPTIVE_RECOMMENDER = AdaptiveRecommender(MODEL_DIR, feature_store=FEATURE_STORE)
FEEDBACK_LOOP = FeedbackLoop(store=FEATURE_STORE, adaptive_callback=ADAPTIVE_RECOMMENDER.observe_event)
RETRIEVAL_SERVICE = RetrievalService()
RANKING_SERVICE = RankingService()
AB_TEST = ABTestingManager(experiment_name="recommendation_v4", split=50)
ONLINE_METRICS = OnlineMetricsTracker()


def get_df() -> pd.DataFrame:
    csv_path = os.path.join(DATA_DIR, "Student_Performance.csv")
    return pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()


def _popular_by_segment(quiz_score: float, top_k: int) -> List[str]:
    if quiz_score < 55:
        subjects = ["Algebra", "Geometry", "Statistics"]
    elif quiz_score < 75:
        subjects = ["Physics", "Chemistry", "Computer Science"]
    else:
        subjects = ["Calculus", "Computer Science", "Statistics"]
    pool = []
    for subject in subjects:
        pool.extend(TOPIC_LIBRARY.get(subject, []))
    return pool[:top_k]


def _global_popular(top_k: int) -> List[str]:
    all_topics = [t for arr in TOPIC_LIBRARY.values() for t in arr]
    return all_topics[:top_k]


def _item_feature_map(candidates: List[str]) -> Dict[str, Dict[str, float]]:
    output: Dict[str, Dict[str, float]] = {}
    for topic in candidates:
        item_online = FEATURE_STORE.get_online_item_features(topic)
        recency_hours = float(item_online.get("recency_hours", 72.0))
        popularity = float(item_online.get("item_ctr", 0.3))
        output[topic] = {
            "embedding_similarity": float(item_online.get("embedding_similarity", 0.5)),
            "recency_hours": recency_hours,
            "popularity": popularity,
            "freshness": float(1.0 / (1.0 + recency_hours / 24.0)),
            "difficulty_match": float(item_online.get("difficulty_match", 0.5)),
        }
    return output


def _queue_shown_events(background_tasks: BackgroundTasks, user_id: str, session_id: str, topics: List[str]) -> None:
    for topic in topics:
        payload = FeedbackEventRequest(
            event_type="recommendation_shown",
            user_id=user_id,
            item_id=topic,
            session_id=session_id,
            payload={},
        )
        background_tasks.add_task(process_feedback_event, payload)


def process_feedback_event(req: FeedbackEventRequest) -> Dict[str, object]:
    if req.schema_version != SCHEMA_VERSION:
        return {"accepted": False, "reason": "schema_version_mismatch"}
    event = EventBase(
        event_type=req.event_type,
        user_id=req.user_id,
        item_id=req.item_id,
        session_id=req.session_id,
        payload=req.payload,
        idempotency_key=req.idempotency_key,
    )
    result = FEEDBACK_LOOP.ingest(event)
    ONLINE_METRICS.record_feedback(req.event_type)
    return result


def classify_learner(predicted_score: float, engagement_score: float) -> str:
    if predicted_score >= 75 and engagement_score >= 60:
        return "Fast Learner"
    if predicted_score < 55 or engagement_score < 35:
        return "Struggling Learner"
    return "Low Engagement"


def build_explanation(predicted_score: float, engagement_score: float, attempts: int, consistency_score: float) -> str:
    reasons = []
    if predicted_score < 55:
        reasons.append("low predicted performance")
    if engagement_score < 35:
        reasons.append("low engagement")
    if attempts > 6:
        reasons.append("high attempts")
    if consistency_score < 50:
        reasons.append("inconsistent performance")
    if not reasons:
        reasons.append("strong overall performance")
    return "This student shows " + ", ".join(reasons) + "."


def risk_level_from_score(predicted_score: float) -> str:
    if predicted_score < 50:
        return "high"
    if predicted_score < 70:
        return "medium"
    return "low"


def estimate_prediction_confidence(
    model,
    X: pd.DataFrame,
    predicted_score: float,
    engagement_score: float,
    consistency_score: float,
) -> float:
    # Prefer model uncertainty (tree spread) when available, else use a conservative heuristic.
    if hasattr(model, "estimators_"):
        try:
            estimator_preds = np.array([float(est.predict(X)[0]) for est in model.estimators_], dtype=float)
            spread = float(np.std(estimator_preds))
            confidence = 1.0 - (spread / 25.0)
            return float(np.clip(confidence, 0.25, 0.98))
        except Exception:
            pass

    distance_component = min(1.0, abs(predicted_score - 50.0) / 50.0)
    behavior_component = np.clip((engagement_score + consistency_score) / 200.0, 0.0, 1.0)
    confidence = 0.35 + 0.3 * distance_component + 0.35 * behavior_component
    return float(np.clip(confidence, 0.25, 0.95))


def topic_prediction_insight(base_predicted_score: float, topic: str, weak_subject: str | None) -> Tuple[float, str, str]:
    topic_lower = topic.lower()
    weak_subject_lower = (weak_subject or "").lower()

    adjustment = 0.0
    if weak_subject_lower and weak_subject_lower in topic_lower:
        adjustment -= 9.0
    if any(token in topic_lower for token in ["advanced", "calculus", "challenge", "optimization"]):
        adjustment -= 4.0
    if any(token in topic_lower for token in ["intro", "fundamentals", "basics", "starter"]):
        adjustment += 3.0

    predicted = float(np.clip(base_predicted_score + adjustment, 0, 100))
    risk_level = risk_level_from_score(predicted)
    if risk_level == "high":
        reason = f"Predicted low performance in {topic}; prioritize this topic to prevent learning gaps."
    elif risk_level == "medium":
        reason = f"Moderate predicted performance in {topic}; a focused practice burst can improve mastery."
    else:
        reason = f"Strong predicted performance in {topic}; use this as momentum-building practice."
    return predicted, risk_level, reason


def make_feature_vector(req_dict: Dict, feat_cols: list):
    """
    Transform request dict into properly engineered feature vector.
    CRITICAL: Must match training pipeline exactly.
    """
    # Step 1: Create base DataFrame from request
    frame = pd.DataFrame([req_dict])
    
    # Step 2: Handle missing values (same as training)
    frame = handle_missing_values(frame)
    
    # Step 3: Engineer features (same as training)
    frame = engineer_features(frame)
    
    # Step 4: Encode categoricals (same as training)
    encoders = ARTEFACTS["encoders"] or {}
    for col in CATEGORICAL_COLS:
        if col in frame.columns and col in encoders:
            encoder = encoders[col]
            value = str(frame[col].iloc[0])
            if value not in set(encoder.classes_):
                # Unknown category: use first class from training
                value = encoder.classes_[0]
            frame[col] = encoder.transform([value])
    
    # Step 5: Select only required features in exact order
    # CRITICAL: feat_cols defines the order and subset
    X = frame[feat_cols].copy()
    
    # Step 6: Ensure no NaNs
    X = X.fillna(0)
    
    # Step 7: Scale numeric features (same as training)
    num_cols = [c for c in NUMERIC_FEATURES if c in feat_cols]
    scaler = ARTEFACTS["scaler"]
    if scaler is not None and num_cols and len(num_cols) > 0:
        try:
            X[num_cols] = scaler.transform(X[num_cols])
        except Exception as e:
            LOGGER.warning(f"Scaler error: {e}. Proceeding without scaling.")
    
    return X


app = FastAPI(
    title="AI Personalized Learning System",
    description="Predict performance, segment learners, and recommend content.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    return {"message": "AI Personalized Learning System is running", "version": "2.0.0", "docs": "/docs"}


@app.get("/health", tags=["General"])
def health():
    model_loaded = ARTEFACTS["model"] is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_version": ARTEFACTS["model_version"],
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


@app.post("/analyze-user", tags=["Unified"])
def analyze_user(req: AnalyzeUserRequest):
    """Predict user performance from clean user input and return focused recommendations."""
    start = time.perf_counter()
    req_dict = req.model_dump()

    with timed_span(LOGGER, "analyze-user", {"subject_weakness": req.subject_weakness}):
        try:
            prediction = predict_user_performance(req_dict)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            LOGGER.error(f"Analyze-user prediction failed: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail="Prediction service unavailable")

        base_topics = content_based_recommend(
            subject_weakness=req.subject_weakness,
            quiz_score=req.quiz_score,
            num_topics=5,
        ).get("topics", [])

        predicted_score = float(prediction["predicted_score"])
        risk_level = str(prediction["risk_level"])

        if risk_level == "high":
            recommendations = base_topics[:5]
        elif risk_level == "medium":
            recommendations = base_topics[:4] + TOPIC_LIBRARY.get("Statistics", [])[:1]
        else:
            recommendations = base_topics[:3] + TOPIC_LIBRARY.get("Computer Science", [])[:2]

        deduped_recommendations = list(dict.fromkeys(recommendations))[:5]

    latency_ms = (time.perf_counter() - start) * 1000.0
    response = {
        "prediction": {
            "predicted_score": round(predicted_score, 2),
            "risk_level": risk_level,
        },
        "recommendations": deduped_recommendations,
        "latency_ms": round(latency_ms, 2),
    }
    log_event(LOGGER, "analyze_user", {"subject_weakness": req.subject_weakness, "latency_ms": latency_ms})
    return response


@app.post("/predict-performance", tags=["Prediction"])
def predict_performance(req: PredictRequest):
    start = time.perf_counter()
    if req.schema_version != SCHEMA_VERSION:
        raise HTTPException(status_code=422, detail=f"Unsupported schema_version {req.schema_version}")
    model = ARTEFACTS["model"]
    feature_cols = ARTEFACTS["feature_cols"]
    if model is None or feature_cols is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run python ml/training/train_model.py")

    req_dict = req.model_dump()
    with timed_span(LOGGER, "predict-performance", {"student_id": req.student_id}):
        try:
            X = make_feature_vector(req_dict, feature_cols)
            predicted_score = float(np.clip(model.predict(X)[0], 0, 100))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Feature error: {exc}")

        engagement_score = float(np.clip(req.time_spent_hours * 10 + req.attempts * 5, 0, 100))
        consistency_score = float((req.attendance + req.assignment_score) / 2)
        risk_level = risk_level_from_score(predicted_score)
        confidence = estimate_prediction_confidence(model, X, predicted_score, engagement_score, consistency_score)
        learning_type = classify_learner(predicted_score, engagement_score)
        explanation = build_explanation(predicted_score, engagement_score, req.attempts, consistency_score)

        weak_areas = []
        if req.quiz_score < 60:
            weak_areas.append(req.subject_weakness or "General Studies")
        if req.attendance < 60:
            weak_areas.append("Attendance / Consistency")
        if req.assignment_score < 55:
            weak_areas.append("Assignment Completion")
        if not weak_areas:
            weak_areas.append("None")

        cb = content_based_recommend(req.subject_weakness or "Algebra", req.quiz_score)

    latency_ms = (time.perf_counter() - start) * 1000.0
    response = {
        "student_id": req.student_id,
        "predicted_score": round(predicted_score, 2),
        "risk_level": risk_level,
        "confidence": round(confidence, 3),
        "learning_type": learning_type,
        "weak_areas": weak_areas,
        "recommended_topics": cb["topics"],
        "explanation": explanation,
        "engagement_score": round(engagement_score, 2),
        "consistency_score": round(consistency_score, 2),
        "model_version": ARTEFACTS["model_version"],
        "latency_ms": round(latency_ms, 2),
    }
    log_event(LOGGER, "predict_response", {"student_id": req.student_id, "latency_ms": response["latency_ms"]})
    return response


@app.post("/recommend-content", tags=["Recommendation"])
def recommend_content(req: RecommendRequest, background_tasks: BackgroundTasks):
    start = time.perf_counter()
    if req.schema_version != SCHEMA_VERSION:
        raise HTTPException(status_code=422, detail=f"Unsupported schema_version {req.schema_version}")

    payload = req.model_dump()
    df = get_df()
    student_features = {
        "quiz_score": req.quiz_score,
        "engagement_score": req.engagement_score or 50,
        "consistency_score": req.consistency_score or 60,
        "attempts": req.attempts or 3,
    }
    requested_k = req.num_topics or 5
    experiment_bucket = AB_TEST.assign_bucket(req.student_id or "anonymous")
    user_online = FEATURE_STORE.get_online_user_features(req.student_id or "anonymous")
    recent_history = list(req.recent_items or FEATURE_STORE.get_recent_user_sequence(req.student_id or "anonymous", limit=24))
    payload["recent_items"] = recent_history
    payload["recent_sequence_size"] = len(recent_history)
    payload["last_feedback_ts"] = user_online.get("last_event_ts")
    cache_key = CACHE.make_key("recommend-content", payload)
    cached = CACHE.get(cache_key)
    if cached is not None:
        cached["cache_hit"] = True
        ONLINE_METRICS.record_request(latency_ms=0.1, cache_hit=True)
        return cached

    with timed_span(LOGGER, "recommend-content", {"student_id": req.student_id, "bucket": experiment_bucket}):
        source = "ann_ranked"
        try:
            retrieval = RETRIEVAL_SERVICE.retrieve(
                user_id=req.student_id or "anonymous",
                subject_weakness=req.subject_weakness,
                quiz_score=req.quiz_score,
                student_features=student_features,
                df=df,
                top_k=max(12, requested_k * 3),
            )
            ann_candidates = retrieval.get("ann", [])
            ranked = RANKING_SERVICE.rank_candidates(
                candidates=ann_candidates,
                user_features=user_online,
                item_feature_map=_item_feature_map(ann_candidates),
                top_k=requested_k,
            )
            adaptive = ADAPTIVE_RECOMMENDER.rank(
                user_id=req.student_id or "anonymous",
                candidates=ranked or ann_candidates,
                user_features={
                    "quiz_score": req.quiz_score,
                    "engagement_score": student_features["engagement_score"],
                    "consistency_score": student_features["consistency_score"],
                },
                item_feature_map=_item_feature_map(ann_candidates),
                recent_history=recent_history,
                top_k=requested_k,
            )
            ranked = rerank_diversity_novelty(adaptive.get("recommended_topics", ranked), ann_candidates, requested_k)
            if not ranked:
                raise RuntimeError("empty ranked candidates")

            base_predicted_score = float(np.clip(0.45 * req.quiz_score + 0.35 * student_features["consistency_score"] + 0.2 * student_features["engagement_score"], 0, 100))
            predicted_recommendations = []
            for topic in ranked:
                pred_score, risk_level, reason = topic_prediction_insight(base_predicted_score, topic, req.subject_weakness)
                risk_priority = {"high": 0, "medium": 1, "low": 2}.get(risk_level, 1)
                predicted_recommendations.append(
                    {
                        "topic": topic,
                        "predicted_score": round(pred_score, 2),
                        "risk_level": risk_level,
                        "reason": reason,
                        "_priority": risk_priority,
                    }
                )

            predicted_recommendations.sort(key=lambda row: (row["_priority"], row["predicted_score"]))
            predicted_recommendations = [
                {
                    "topic": row["topic"],
                    "predicted_score": row["predicted_score"],
                    "risk_level": row["risk_level"],
                    "reason": row["reason"],
                }
                for row in predicted_recommendations[:requested_k]
            ]
            ranked_topic_titles = [row["topic"] for row in predicted_recommendations]

            result = {
                "weak_areas": [req.subject_weakness],
                "content_based_topics": ann_candidates[:requested_k],
                "collaborative_topics": ann_candidates[:requested_k],
                "recommended_topics": predicted_recommendations,
                "recommended_topic_titles": ranked_topic_titles,
            }
            source = f"{retrieval.get('source', source)}+{adaptive.get('source', 'adaptive_bandit_transformer')}"
        except Exception:
            # Fallback ladder: hybrid -> segment popular -> global popular
            source = "hybrid_fallback"
            try:
                result = hybrid_recommend(
                    subject_weakness=req.subject_weakness,
                    quiz_score=req.quiz_score,
                    student_features=student_features,
                    df=df,
                    num_topics=requested_k,
                )
                if not result.get("recommended_topics"):
                    raise RuntimeError("empty hybrid output")
            except Exception:
                source = "segment_popular_fallback"
                segment_topics = _popular_by_segment(req.quiz_score, requested_k)
                if segment_topics:
                    result = {
                        "weak_areas": [req.subject_weakness],
                        "content_based_topics": segment_topics,
                        "collaborative_topics": [],
                        "recommended_topics": segment_topics,
                    }
                else:
                    source = "global_popular_fallback"
                    global_topics = _global_popular(requested_k)
                    result = {
                        "weak_areas": [req.subject_weakness],
                        "content_based_topics": global_topics,
                        "collaborative_topics": [],
                        "recommended_topics": global_topics,
                    }

            fallback_topics = result.get("recommended_topics", [])
            base_predicted_score = float(np.clip(0.45 * req.quiz_score + 0.35 * student_features["consistency_score"] + 0.2 * student_features["engagement_score"], 0, 100))
            enriched_fallback = []
            for topic in fallback_topics:
                pred_score, risk_level, reason = topic_prediction_insight(base_predicted_score, topic, req.subject_weakness)
                enriched_fallback.append(
                    {
                        "topic": topic,
                        "predicted_score": round(pred_score, 2),
                        "risk_level": risk_level,
                        "reason": reason,
                    }
                )
            result["recommended_topics"] = enriched_fallback
            result["recommended_topic_titles"] = [row["topic"] for row in enriched_fallback]

    result["student_id"] = req.student_id
    result["model_version"] = ARTEFACTS["model_version"]
    result["experiment_bucket"] = experiment_bucket
    result["recommendation_source"] = source
    result["latency_ms"] = round((time.perf_counter() - start) * 1000.0, 2)
    session_id = req.session_id or f"{req.student_id or 'anonymous'}:{int(time.time())}"
    shown_topics = [
        row["topic"] if isinstance(row, dict) else row
        for row in result.get("recommended_topics", [])
        if (isinstance(row, dict) and row.get("topic")) or isinstance(row, str)
    ]
    _queue_shown_events(background_tasks, req.student_id or "anonymous", session_id, shown_topics)

    AB_TEST.record_observation(result["latency_ms"], success=True)
    ONLINE_METRICS.record_request(latency_ms=result["latency_ms"], cache_hit=False)

    metrics_path = os.path.join(MODEL_DIR, "online_metrics.json")
    emit_metrics(metrics_path, ONLINE_METRICS.snapshot())

    CACHE.set(cache_key, result)
    return result


@app.post("/feedback-event", tags=["Adaptation"])
def feedback_event(req: FeedbackEventRequest):
    if req.schema_version != SCHEMA_VERSION:
        raise HTTPException(status_code=422, detail=f"Unsupported schema_version {req.schema_version}")
    result = process_feedback_event(req)
    return {"status": "ok" if result.get("accepted") else "deduped", **result}


@app.get("/student-profile/{student_id}", tags=["Profile"])
def student_profile(student_id: str):
    df = get_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="Dataset not found. Run training first.")
    row_df = df[df["student_id"] == student_id]
    if row_df.empty:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found")

    row = row_df.iloc[0].to_dict()
    cluster = row.get("cluster_label", "Unknown")
    predicted = float(row.get("predicted_score", row.get("final_score", 0)))
    engagement = float(np.clip(row.get("time_spent_hours", 5) * 10 + row.get("attempts", 3) * 5, 0, 100))
    consistency = float((row.get("attendance", 70) + row.get("assignment_score", 70)) / 2)
    weak_areas = []
    if row.get("quiz_score", 70) < 60:
        weak_areas.append(row.get("subject_weakness", "General"))
    if row.get("attendance", 70) < 60:
        weak_areas.append("Attendance")

    return {
        "student_id": student_id,
        "age": row.get("age"),
        "gender": row.get("gender"),
        "learning_style": row.get("learning_style"),
        "subject_strength": row.get("subject_strength"),
        "subject_weakness": row.get("subject_weakness"),
        "attendance": row.get("attendance"),
        "assignment_score": row.get("assignment_score"),
        "quiz_score": row.get("quiz_score"),
        "final_score": row.get("final_score"),
        "predicted_score": round(predicted, 2),
        "engagement_score": round(engagement, 2),
        "consistency_score": round(consistency, 2),
        "cluster_label": cluster,
        "weak_areas": weak_areas,
        "score_history": [round(float(row.get("previous_score", predicted)) * 0.85, 1), round(float(row.get("previous_score", predicted)) * 0.92, 1), round(float(row.get("quiz_score", predicted)), 1), round(predicted, 1)],
        "streak": int(row.get("extracurricular", 0)) * 3 + 1,
        "internet_access": bool(row.get("internet_access", 1)),
        "parental_support": row.get("parental_support"),
        "stress_level": row.get("stress_level"),
        "model_version": ARTEFACTS["model_version"],
    }


@app.post("/update-after-quiz", tags=["Adaptation"])
def update_after_quiz(req: QuizUpdateRequest):
    if req.schema_version != SCHEMA_VERSION:
        raise HTTPException(status_code=422, detail=f"Unsupported schema_version {req.schema_version}")
    df = get_df()
    row_df = df[df["student_id"] == req.student_id] if not df.empty else pd.DataFrame()
    if not row_df.empty:
        row = row_df.iloc[0].to_dict()
        attendance = float(row.get("attendance", 75))
        assignment_score = float(row.get("assignment_score", 70))
        previous_score = float(row.get("quiz_score", 65))
        subject_weakness = str(row.get("subject_weakness", req.subject))
    else:
        attendance, assignment_score, previous_score = 75.0, 70.0, 65.0
        subject_weakness = req.subject

    req_dict = {
        "age": float(row_df.iloc[0].get("age", 18)) if not row_df.empty else 18.0,
        "gender": str(row_df.iloc[0].get("gender", "Male")) if not row_df.empty else "Male",
        "learning_style": str(row_df.iloc[0].get("learning_style", "Visual")) if not row_df.empty else "Visual",
        "attendance": attendance,
        "assignment_score": assignment_score,
        "quiz_score": req.new_quiz_score,
        "time_spent_hours": req.time_spent_hours,
        "attempts": req.attempts,
        "previous_score": previous_score,
        "internet_access": int(row_df.iloc[0].get("internet_access", 1)) if not row_df.empty else 1,
        "parental_support": str(row_df.iloc[0].get("parental_support", "Medium")) if not row_df.empty else "Medium",
        "extracurricular": int(row_df.iloc[0].get("extracurricular", 1)) if not row_df.empty else 1,
        "stress_level": str(row_df.iloc[0].get("stress_level", "Medium")) if not row_df.empty else "Medium",
        "subject_weakness": subject_weakness,
    }

    model = ARTEFACTS["model"]
    feature_cols = ARTEFACTS["feature_cols"]
    if model is None or feature_cols is None:
        new_predicted = round((req.new_quiz_score * 0.4 + attendance * 0.3 + assignment_score * 0.3), 2)
    else:
        X = make_feature_vector(req_dict, feature_cols)
        new_predicted = float(np.clip(model.predict(X)[0], 0, 100))

    engagement_score = float(np.clip(req.time_spent_hours * 10 + req.attempts * 5, 0, 100))
    consistency_score = float((attendance + assignment_score) / 2)
    learning_type = classify_learner(new_predicted, engagement_score)
    cb = content_based_recommend(subject_weakness, req.new_quiz_score)

    if not df.empty and not row_df.empty:
        idx = df[df["student_id"] == req.student_id].index[0]
        df.at[idx, "quiz_score"] = req.new_quiz_score
        df.at[idx, "time_spent_hours"] = req.time_spent_hours
        df.at[idx, "attempts"] = req.attempts
        df.at[idx, "predicted_score"] = new_predicted
        df.to_csv(os.path.join(DATA_DIR, "Student_Performance.csv"), index=False)

    return {
        "student_id": req.student_id,
        "subject": req.subject,
        "new_quiz_score": req.new_quiz_score,
        "new_predicted_score": round(new_predicted, 2),
        "learning_type": learning_type,
        "engagement_score": round(engagement_score, 2),
        "consistency_score": round(consistency_score, 2),
        "updated_recommendations": cb["topics"],
        "model_version": ARTEFACTS["model_version"],
        "message": f"Student {req.student_id} updated successfully after quiz on {req.subject}.",
    }

