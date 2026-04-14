from ml.recommender.ranking_service import RankingService


def test_ranking_output_contract():
    service = RankingService()
    candidates = ["A", "B", "C", "D"]
    user_features = {"completion_rate": 0.4}
    item_map = {c: {"embedding_similarity": 0.6, "recency_hours": 24.0, "popularity": 0.3} for c in candidates}

    ranked = service.rank_candidates(candidates, user_features, item_map, top_k=3)

    assert isinstance(ranked, list)
    assert len(ranked) == 3
    assert all(item in candidates for item in ranked)


