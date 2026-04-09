from __future__ import annotations


def test_search_returns_fact_and_entity_candidates(engine):
    namespace = "test.retrieval"
    engine.add_messages(
        [
            {"content": "Toronto Maple Leafs finished with 48 wins in the 2025-26 season.", "source_type": "tool_result"},
            {"content": "NHL procedure note: season summaries should cite playoff outcome.", "source_type": "tool_result"},
        ],
        namespace_id=namespace,
        user_id="sam",
        agent_id="sportsdesk",
        session_id="sports-1",
    )
    engine.consolidate(namespace)

    results = engine.search("Toronto Maple Leafs 2025-26 season", namespace_id=namespace, limit=6)
    keys = [item["node_key"] for item in results]
    assert any(key.startswith("Entity:") for key in keys)
    assert any(key.startswith("Fact:") or key.startswith("Episode:") for key in keys)
