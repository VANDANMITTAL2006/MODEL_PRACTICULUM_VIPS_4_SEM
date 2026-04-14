"""Cold-start strategies for new users or sparse history."""

from __future__ import annotations

from typing import Dict, List

from ml.recommender.candidate_generation import TOPIC_LIBRARY


def _topic_prior(subject_weakness: str, quiz_score: float) -> List[str]:
    topics = TOPIC_LIBRARY.get(subject_weakness, [])
    if not topics:
        return [topic for values in TOPIC_LIBRARY.values() for topic in values]

    if quiz_score < 50:
        return topics[:]
    if quiz_score < 70:
        return topics[:2] + topics[2:]
    return list(reversed(topics))


def cold_start_recommend(subject_weakness: str, quiz_score: float, num_topics: int = 5) -> Dict[str, List[str]]:
    ordered = _topic_prior(subject_weakness, quiz_score)
    if not ordered:
        all_topics = [t for arr in TOPIC_LIBRARY.values() for t in arr]
        return {"recommended_topics": all_topics[:num_topics], "cold_start": True}

    focused = ordered[:num_topics]
    if len(focused) < num_topics:
        all_topics = [t for arr in TOPIC_LIBRARY.values() for t in arr]
        for topic in all_topics:
            if topic not in focused:
                focused.append(topic)
            if len(focused) >= num_topics:
                break

    return {"recommended_topics": focused[:num_topics], "cold_start": True, "source": "content_popularity_hybrid"}

