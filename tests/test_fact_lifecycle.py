from __future__ import annotations

from sqlalchemy import select

from memoria.models import Fact


def test_preference_fact_is_superseded_when_updated(engine):
    namespace = "test.facts"
    engine.add_episode(
        namespace_id=namespace,
        user_id="alex",
        session_id="s1",
        content_raw="I prefer tea.",
        source_type="user_message",
        role="user",
    )
    engine.consolidate(namespace)

    engine.add_episode(
        namespace_id=namespace,
        user_id="alex",
        session_id="s1",
        content_raw="I prefer coffee.",
        source_type="user_message",
        role="user",
    )
    engine.consolidate(namespace)

    with engine.session_factory() as session:
        facts = session.scalars(
            select(Fact).where(Fact.namespace_id == namespace, Fact.predicate == "prefers").order_by(Fact.id)
        ).all()
        assert len(facts) >= 2
        assert facts[0].valid_to is not None
        assert facts[-1].valid_to is None
        assert facts[0].superseded_by_fact_id == facts[-1].id
