from __future__ import annotations

from time import perf_counter

from sqlalchemy import select

from tame.demo_data import demo_catalog
from tame.engine import MemoryEngine
from tame.models import Episode
from tame.utils import word_count


def precision_at_k(items: list[str], expected_prefixes: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = items[:k]
    hits = sum(1 for item in top if any(item.startswith(prefix) for prefix in expected_prefixes))
    return hits / k


def evaluate_demo(engine: MemoryEngine, demo_name: str) -> dict:
    demo = demo_catalog()[demo_name]
    namespace_id = demo["namespace_id"]
    with engine.session_factory() as session:
        has_data = session.scalar(select(Episode.id).where(Episode.namespace_id == namespace_id).limit(1)) is not None
    if not has_data:
        engine.add_messages(
            demo["episodes"],
            namespace_id=demo["namespace_id"],
            user_id=demo["user_id"],
            agent_id=demo["agent_id"],
            session_id=demo["session_id"],
        )
        engine.consolidate(namespace_id=namespace_id)
    baseline_prompt = demo["steps"][-1]["text"]

    baseline_start = perf_counter()
    baseline = engine.search(baseline_prompt, namespace_id=namespace_id, limit=6)
    baseline_latency = perf_counter() - baseline_start

    run = engine.start_run(
        namespace_id=namespace_id,
        user_id=demo["user_id"],
        agent_id=demo["agent_id"],
        session_id=demo["session_id"],
    )
    activation_start = perf_counter()
    for step in demo["steps"]:
        engine.process_step(
            run["run_id"],
            step_type=step["step_type"],
            text=step["text"],
            metadata=step.get("metadata"),
        )
    activation_latency = perf_counter() - activation_start
    working_memory = engine.get_working_memory(run["run_id"], step_index=len(demo["steps"]) - 1)

    baseline_keys = [item["node_key"] for item in baseline]
    working_keys = [item["node_key"] for item in working_memory]
    expected = demo["expected"]
    relevant_activation = sum(1 for key in working_keys if any(key.startswith(prefix) for prefix in expected))
    irrelevant_activation = len(working_keys) - relevant_activation
    relevant_baseline = sum(1 for key in baseline_keys if any(key.startswith(prefix) for prefix in expected))
    irrelevant_baseline = len(baseline_keys) - relevant_baseline

    return {
        "demo": demo_name,
        "baseline": {
            "latency_seconds": baseline_latency,
            "relevant_nodes_surfaced": relevant_baseline,
            "irrelevant_nodes_surfaced": irrelevant_baseline,
            "precision_at_5": precision_at_k(baseline_keys, expected, 5),
            "token_count": sum(word_count(item["text"]) for item in baseline),
        },
        "activation": {
            "latency_seconds": activation_latency,
            "relevant_nodes_surfaced": relevant_activation,
            "irrelevant_nodes_surfaced": irrelevant_activation,
            "precision_at_5": precision_at_k(working_keys, expected, 5),
            "token_count": sum(word_count(item["content"]) for item in working_memory),
            "subjective_pass": None,
        },
    }
