from api.schemas.contracts import PredictRequest, RecommendRequest, SCHEMA_VERSION


def test_predict_schema_version_default():
    req = PredictRequest(attendance=80, assignment_score=75, quiz_score=70, time_spent_hours=3.0, attempts=2)
    assert req.schema_version == SCHEMA_VERSION


def test_recommend_schema_shape():
    req = RecommendRequest(quiz_score=65, subject_weakness="Algebra")
    payload = req.model_dump()
    assert "quiz_score" in payload
    assert "subject_weakness" in payload
    assert payload["schema_version"] == SCHEMA_VERSION

