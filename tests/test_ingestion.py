from __future__ import annotations

from sqlalchemy import select

from memoria.models import Episode


def test_add_update_delete_episode(engine):
    created = engine.add_episode(
        namespace_id="test.ingest",
        user_id="u1",
        session_id="s1",
        content_raw="I prefer black coffee in the morning.",
        source_type="user_message",
        role="user",
    )
    assert created["id"] > 0

    updated = engine.update_episode(created["id"], "I prefer green tea in the morning.")
    assert "green tea" in updated["content_summary"]

    with engine.session_factory() as session:
        episode = session.scalar(select(Episode).where(Episode.id == created["id"]))
        assert episode is not None
        assert "green tea" in episode.content_raw

    engine.delete_episode(created["id"])
    with engine.session_factory() as session:
        episode = session.scalar(select(Episode).where(Episode.id == created["id"]))
        assert episode is None
