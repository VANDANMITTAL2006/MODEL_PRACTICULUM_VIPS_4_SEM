"""
FastAPI Backend — AI Personalized Learning System
Endpoints:
  GET  /
  GET  /health
  POST /predict-performance
  POST /recommend-content
  GET  /student-profile/{student_id}
  POST /update-after-quiz
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import datetime

# ── Path setup ───────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, BASE_DIR)

from data.data_pipeline import (
    handle_missing_values, engineer_features,
    encode_categoricals, normalize_features,
    NUMERIC_FEATURES, CATEGORICAL_COLS, TARGET
)
from backend.recommender import hybrid_recommend, content_based_recommend, TOPIC_LIBRARY

# ── Load artefacts ────────────────────────────────────────
def load_artefacts():
    try:
        model        = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
        scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        encoders     = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
        kmeans       = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
        cluster_map  = joblib.load(os.path.join(MODEL_DIR, "cluster_mapping.pkl"))
        feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
        return model, scaler, encoders, kmeans, cluster_map, feature_cols
    except FileNotFoundError as e:
        return None, None, None, None, None, None

model, scaler, encoders, kmeans, cluster_map, feature_cols = load_artefacts()

def get_df():
    csv_path = os.path.join(DATA_DIR, "Student_Performance.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(
    title="AI Personalized Learning System",
    description="Predict performance, segment learners, and recommend content.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────
class PredictRequest(BaseModel):
    student_id: Optional[str] = "S0001"
    age: Optional[float] = 18
    gender: Optional[str] = "Male"
    learning_style: Optional[str] = "Visual"
    attendance: float = Field(..., ge=0, le=100, description="Attendance %")
    assignment_score: float = Field(..., ge=0, le=100)
    quiz_score: float = Field(..., ge=0, le=100)
    time_spent_hours: float = Field(..., ge=0.1, le=20)
    attempts: int = Field(..., ge=1, le=20)
    previous_score: Optional[float] = 65.0
    internet_access: Optional[int] = 1
    parental_support: Optional[str] = "Medium"
    extracurricular: Optional[int] = 1
    stress_level: Optional[str] = "Medium"
    subject_weakness: Optional[str] = "Algebra"

class RecommendRequest(BaseModel):
    student_id: Optional[str] = "S0001"
    quiz_score: float = Field(..., ge=0, le=100)
    subject_weakness: str = "Algebra"
    engagement_score: Optional[float] = 50.0
    consistency_score: Optional[float] = 60.0
    attempts: Optional[int] = 3
    num_topics: Optional[int] = 5

class QuizUpdateRequest(BaseModel):
    student_id: str
    subject: str
    new_quiz_score: float = Field(..., ge=0, le=100)
    time_spent_hours: float = Field(..., ge=0.1)
    attempts: int = Field(..., ge=1)

# ── Helpers ───────────────────────────────────────────────
SUBJECTS = list(TOPIC_LIBRARY.keys())

def classify_learner(predicted_score: float, engagement_score: float) -> str:
    if predicted_score >= 75 and engagement_score >= 60:
        return "Fast Learner"
    elif predicted_score < 55 or engagement_score < 35:
        return "Struggling Learner"
    else:
        return "Low Engagement"

def build_explanation(predicted_score, engagement_score, attempts, consistency_score) -> str:
    reasons = []
    if predicted_score < 55:
        reasons.append("low predicted performance")
    if engagement_score < 35:
        reasons.append("low engagement with study material")
    if attempts > 6:
        reasons.append("high number of attempts suggesting difficulty")
    if consistency_score < 50:
        reasons.append("inconsistent attendance and assignment completion")
    if not reasons:
        reasons.append("strong overall performance")
    return "This student shows " + ", ".join(reasons) + "."

def make_feature_vector(req_dict: dict, feat_cols: list):
    """Build a single-row feature vector from request data."""
    df_input = pd.DataFrame([req_dict])
    df_input = handle_missing_values(df_input)
    df_input = engineer_features(df_input)
    # encode categoricals
    for col in CATEGORICAL_COLS:
        if col in df_input.columns and encoders and col in encoders:
            le = encoders[col]
            val = str(df_input[col].iloc[0])
            if val not in le.classes_:
                val = le.classes_[0]
            df_input[col] = le.transform([val])
    # select features
    available = [c for c in feat_cols if c in df_input.columns]
    X = df_input[available].fillna(0)
    # scale
    num_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    if scaler and num_cols:
        X[num_cols] = scaler.transform(X[num_cols])
    return X

# ── Routes ────────────────────────────────────────────────
@app.get("/", tags=["General"])
def root():
    return {
        "message": "🎓 AI Personalized Learning System is running!",
        "version": "1.0.0",
        "docs": "/docs",
    }

@app.get("/health", tags=["General"])
def health():
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

@app.post("/predict-performance", tags=["Prediction"])
def predict_performance(req: PredictRequest):
    if model is None or feature_cols is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run: python models/train_model.py"
        )

    req_dict = req.dict()
    try:
        X = make_feature_vector(req_dict, feature_cols)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature error: {str(e)}")

    predicted_score = float(np.clip(model.predict(X)[0], 0, 100))

    # engagement & consistency
    engagement_score  = float(np.clip(req.time_spent_hours * 10 + req.attempts * 5, 0, 100))
    consistency_score = float((req.attendance + req.assignment_score) / 2)

    learning_type = classify_learner(predicted_score, engagement_score)
    explanation   = build_explanation(predicted_score, engagement_score, req.attempts, consistency_score)

    # Weak areas heuristic
    weak_areas = []
    if req.quiz_score < 60:
        weak_areas.append(req.subject_weakness or "General Studies")
    if req.attendance < 60:
        weak_areas.append("Attendance / Consistency")
    if req.assignment_score < 55:
        weak_areas.append("Assignment Completion")
    if not weak_areas:
        weak_areas.append("None — performing well!")

    # Content-based quick recommendations
    cb = content_based_recommend(req.subject_weakness or "Algebra", req.quiz_score)
    recommended_topics = cb["topics"]

    return {
        "student_id": req.student_id,
        "predicted_score": round(predicted_score, 2),
        "learning_type": learning_type,
        "weak_areas": weak_areas,
        "recommended_topics": recommended_topics,
        "explanation": explanation,
        "engagement_score": round(engagement_score, 2),
        "consistency_score": round(consistency_score, 2),
    }

@app.post("/recommend-content", tags=["Recommendation"])
def recommend_content(req: RecommendRequest):
    df = get_df()
    student_features = {
        "quiz_score": req.quiz_score,
        "engagement_score": req.engagement_score or 50,
        "consistency_score": req.consistency_score or 60,
        "attempts": req.attempts or 3,
    }
    result = hybrid_recommend(
        subject_weakness=req.subject_weakness,
        quiz_score=req.quiz_score,
        student_features=student_features,
        df=df,
        num_topics=req.num_topics or 5,
    )
    result["student_id"] = req.student_id
    return result

@app.get("/student-profile/{student_id}", tags=["Profile"])
def student_profile(student_id: str):
    df = get_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="Dataset not found. Run training first.")

    row_df = df[df["student_id"] == student_id]
    if row_df.empty:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' not found.")

    row = row_df.iloc[0].to_dict()

    # Convert numpy types to Python natives
    for k, v in row.items():
        if isinstance(v, (np.integer,)):
            row[k] = int(v)
        elif isinstance(v, (np.floating,)):
            row[k] = float(round(v, 2))

    cluster = row.get("cluster_label", "Unknown")
    predicted = row.get("predicted_score", row.get("final_score", 0))
    engagement = float(np.clip(row.get("time_spent_hours", 5) * 10 + row.get("attempts", 3) * 5, 0, 100))
    consistency = (row.get("attendance", 70) + row.get("assignment_score", 70)) / 2

    weak_areas = []
    if row.get("quiz_score", 70) < 60:
        weak_areas.append(row.get("subject_weakness", "General"))
    if row.get("attendance", 70) < 60:
        weak_areas.append("Attendance")

    # Score history mock
    score_history = [
        round(float(row.get("previous_score", predicted)) * 0.85, 1),
        round(float(row.get("previous_score", predicted)) * 0.92, 1),
        round(float(row.get("quiz_score", predicted)), 1),
        round(float(predicted), 1),
    ]

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
        "predicted_score": round(float(predicted), 2),
        "engagement_score": round(engagement, 2),
        "consistency_score": round(consistency, 2),
        "cluster_label": cluster,
        "weak_areas": weak_areas,
        "score_history": score_history,
        "streak": int(row.get("extracurricular", 0)) * 3 + 1,
        "internet_access": bool(row.get("internet_access", 1)),
        "parental_support": row.get("parental_support"),
        "stress_level": row.get("stress_level"),
    }

@app.post("/update-after-quiz", tags=["Adaptation"])
def update_after_quiz(req: QuizUpdateRequest):
    """Real-time update: recalculate prediction after a new quiz attempt."""
    df = get_df()

    row_df = df[df["student_id"] == req.student_id] if not df.empty else pd.DataFrame()

    if not row_df.empty:
        row = row_df.iloc[0].to_dict()
        attendance = float(row.get("attendance", 75))
        assignment_score = float(row.get("assignment_score", 70))
        previous_score = float(row.get("quiz_score", 65))
        age = float(row.get("age", 18))
        gender = str(row.get("gender", "Male"))
        learning_style = str(row.get("learning_style", "Visual"))
        parental_support = str(row.get("parental_support", "Medium"))
        stress_level = str(row.get("stress_level", "Medium"))
        internet_access = int(row.get("internet_access", 1))
        extracurricular = int(row.get("extracurricular", 1))
        subject_weakness = str(row.get("subject_weakness", req.subject))
    else:
        attendance, assignment_score, previous_score = 75.0, 70.0, 65.0
        age, gender, learning_style = 18.0, "Male", "Visual"
        parental_support, stress_level = "Medium", "Medium"
        internet_access, extracurricular = 1, 1
        subject_weakness = req.subject

    req_dict = {
        "age": age, "gender": gender, "learning_style": learning_style,
        "attendance": attendance, "assignment_score": assignment_score,
        "quiz_score": req.new_quiz_score,
        "time_spent_hours": req.time_spent_hours,
        "attempts": req.attempts,
        "previous_score": previous_score,
        "internet_access": internet_access,
        "parental_support": parental_support,
        "extracurricular": extracurricular,
        "stress_level": stress_level,
        "subject_weakness": subject_weakness,
    }

    if model is None or feature_cols is None:
        # Fallback heuristic
        new_predicted = round((req.new_quiz_score * 0.4 + attendance * 0.3 + assignment_score * 0.3), 2)
    else:
        try:
            X = make_feature_vector(req_dict, feature_cols)
            new_predicted = float(np.clip(model.predict(X)[0], 0, 100))
        except Exception:
            new_predicted = round((req.new_quiz_score * 0.4 + attendance * 0.3 + assignment_score * 0.3), 2)

    engagement_score  = float(np.clip(req.time_spent_hours * 10 + req.attempts * 5, 0, 100))
    consistency_score = float((attendance + assignment_score) / 2)
    learning_type = classify_learner(new_predicted, engagement_score)

    # Update CSV if possible
    if not df.empty and not row_df.empty:
        idx = df[df["student_id"] == req.student_id].index[0]
        df.at[idx, "quiz_score"]     = req.new_quiz_score
        df.at[idx, "time_spent_hours"] = req.time_spent_hours
        df.at[idx, "attempts"]       = req.attempts
        df.at[idx, "predicted_score"] = new_predicted
        csv_path = os.path.join(DATA_DIR, "Student_Performance.csv")
        df.to_csv(csv_path, index=False)

    cb = content_based_recommend(subject_weakness, req.new_quiz_score)

    return {
        "student_id": req.student_id,
        "subject": req.subject,
        "new_quiz_score": req.new_quiz_score,
        "new_predicted_score": round(new_predicted, 2),
        "learning_type": learning_type,
        "engagement_score": round(engagement_score, 2),
        "consistency_score": round(consistency_score, 2),
        "updated_recommendations": cb["topics"],
        "message": f"✅ Student '{req.student_id}' updated successfully after quiz on {req.subject}.",
    }
