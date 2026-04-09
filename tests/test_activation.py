from __future__ import annotations


def test_activation_persists_trace_across_steps(engine):
    namespace = "test.activation"
    engine.add_messages(
        [
            {"content": "I prefer oat milk in coffee.", "source_type": "user_message"},
            {"content": "I am working on the Orion project.", "source_type": "user_message"},
        ],
        namespace_id=namespace,
        user_id="alex",
        agent_id="pa",
        session_id="sess-1",
    )
    engine.consolidate(namespace)

    run = engine.start_run(namespace_id=namespace, user_id="alex", agent_id="pa", session_id="sess-1")
    wm1 = engine.process_step(run["run_id"], step_type="user_input", text="Help me plan a meeting for Orion.")
    wm2 = engine.process_step(run["run_id"], step_type="reasoning_step", text="Which preferences matter for the meeting?")

    trace = engine.get_debug_trace(run["run_id"])
    assert len(trace["steps"]) == 2
    assert wm1
    assert wm2
    assert any(item["node_key"].startswith("SummaryNode:") or item["node_key"].startswith("Fact:") for item in wm2)
