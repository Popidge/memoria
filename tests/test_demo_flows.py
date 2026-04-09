from __future__ import annotations

from memoria.demo_data import demo_catalog
from memoria.evaluation import evaluate_demo


def test_sports_demo_shows_light_up_behavior(engine):
    demo = demo_catalog()["sports_query"]
    engine.add_messages(
        demo["episodes"],
        namespace_id=demo["namespace_id"],
        user_id=demo["user_id"],
        agent_id=demo["agent_id"],
        session_id=demo["session_id"],
    )
    engine.consolidate(demo["namespace_id"])

    run = engine.start_run(
        namespace_id=demo["namespace_id"],
        user_id=demo["user_id"],
        agent_id=demo["agent_id"],
        session_id=demo["session_id"],
    )
    for step in demo["steps"]:
        engine.process_step(run["run_id"], step_type=step["step_type"], text=step["text"], metadata=step.get("metadata"))

    trace = engine.get_debug_trace(run["run_id"])
    first_step = trace["steps"][0]
    keys = [item["node_key"] for item in first_step["activation"][:8]]
    assert any(key.startswith("Entity:") for key in keys)
    assert any(key.startswith("Fact:") for key in keys)
    assert first_step["working_memory"]


def test_evaluation_harness_reports_metrics(engine):
    result = evaluate_demo(engine, "personal_assistant")
    assert "baseline" in result
    assert "activation" in result
    assert result["activation"]["token_count"] >= 0
