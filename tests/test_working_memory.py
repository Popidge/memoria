from __future__ import annotations


def test_working_memory_prefers_summary_and_fact_before_episode_snippet(engine):
    namespace = "test.working"
    engine.add_messages(
        [
            {"content": "Toronto Maple Leafs finished the 2025-26 NHL season with 48 wins, 26 losses, and 8 overtime losses.", "source_type": "tool_result"},
            {"content": "Toronto Maple Leafs were eliminated in the second round of the 2025-26 Stanley Cup Playoffs.", "source_type": "tool_result"},
        ],
        namespace_id=namespace,
        user_id="sam",
        agent_id="sportsdesk",
        session_id="sports-2",
    )
    engine.consolidate(namespace)

    run = engine.start_run(namespace_id=namespace, user_id="sam", agent_id="sportsdesk", session_id="sports-2")
    items = engine.process_step(
        run["run_id"],
        step_type="user_input",
        text="How did the Toronto Maple Leafs perform in 2025-26?",
        metadata={"include_provenance": True},
    )
    assert items
    assert items[0]["content_type"] in {"summary_node", "fact_summary"}
    assert any(item["content_type"] == "fact_summary" for item in items)
