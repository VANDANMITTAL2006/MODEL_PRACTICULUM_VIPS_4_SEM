"""API schemas with strict validation and schema versioning."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "2.0"


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictRequest(StrictBaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    student_id: Optional[str] = "S0001"
    age: Optional[float] = 18
    gender: Optional[str] = "Male"
    learning_style: Optional[str] = "Visual"
    attendance: float = Field(..., ge=0, le=100)
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


class RecommendRequest(StrictBaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    student_id: Optional[str] = "S0001"
    session_id: Optional[str] = None
    quiz_score: float = Field(..., ge=0, le=100)
    subject_weakness: str = "Algebra"
    engagement_score: Optional[float] = 50.0
    consistency_score: Optional[float] = 60.0
    attempts: Optional[int] = 3
    num_topics: Optional[int] = 5
    recent_items: list[str] = Field(default_factory=list)


class QuizUpdateRequest(StrictBaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    student_id: str
    subject: str
    new_quiz_score: float = Field(..., ge=0, le=100)
    time_spent_hours: float = Field(..., ge=0.1)
    attempts: int = Field(..., ge=1)


class FeedbackEventRequest(StrictBaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    event_type: str
    user_id: str
    item_id: Optional[str] = None
    session_id: Optional[str] = None
    payload: dict = Field(default_factory=dict)
    idempotency_key: Optional[str] = None


class RecoResponse(StrictBaseModel):
    student_id: Optional[str]
    weak_areas: List[str]
    content_based_topics: List[str]
    collaborative_topics: List[str]
    recommended_topics: List[str]
    model_version: str
    latency_ms: float

