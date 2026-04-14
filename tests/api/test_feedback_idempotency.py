from api.core.event_schema import EventBase
from api.core.feedback_loop import FeedbackLoop
from api.core.feature_store import FeatureStore


def test_feedback_idempotent_ingestion(tmp_path):
    store = FeatureStore()
    loop = FeedbackLoop(store=store)
    loop.events_path = str(tmp_path / "events.csv")
    loop.dedup_path = str(tmp_path / "dedup_keys.txt")

    ev = EventBase(event_type="recommendation_clicked", user_id="U1", item_id="I1", idempotency_key="abc")
    first = loop.ingest(ev)
    second = loop.ingest(ev)

    assert first["accepted"] is True
    assert second["deduplicated"] is True

