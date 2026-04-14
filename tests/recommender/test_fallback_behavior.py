import pandas as pd

from api.services.recommendation_engine import hybrid_recommend


def test_hybrid_cold_start_fallback():
    result = hybrid_recommend(
        subject_weakness="Algebra",
        quiz_score=40,
        student_features={"quiz_score": 40, "engagement_score": 30, "consistency_score": 40, "attempts": 4},
        df=pd.DataFrame(),
        num_topics=5,
    )
    assert "recommended_topics" in result
    assert len(result["recommended_topics"]) > 0

