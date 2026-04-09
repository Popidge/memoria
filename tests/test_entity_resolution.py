from __future__ import annotations

from tame.extraction import EntityCandidate


def test_entity_resolution_handles_exact_and_alias_matches(engine):
    with engine.session_factory() as session:
        entity = engine.resolver.resolve_or_create(
            session,
            "test.entities",
            EntityCandidate(name="Toronto Maple Leafs", aliases=["Leafs"], summary="NHL team"),
        )
        session.commit()

    with engine.session_factory() as session:
        resolved = engine.resolver.resolve_or_create(
            session,
            "test.entities",
            EntityCandidate(name="Leafs", summary="Alias lookup"),
        )
        session.commit()

    assert resolved.id == entity.id
