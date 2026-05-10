from __future__ import annotations

from pathlib import Path
import json

from sqlalchemy import select
from typer.testing import CliRunner

from memoria.cli import app
from memoria.context import build_memory_context_packet, render_memory_context_packet
from memoria.models import ChunkEvidenceLink, EdgeDescriptor, EpisodeChunk, NodeDescriptor, NodeType
from memoria.sidecar import MemoriaSidecar
from memoria.workbench import RunSpec, WorkbenchService
from memoria.providers import ProviderConfig


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "memoryarena"


def test_episode_chunks_are_created_for_core_sources(engine):
    namespace = "test.v2.chunks"
    engine.add_messages(
        [
            {"role": "user", "source_type": "user_message", "content": "I prefer jasmine tea. The budget must stay under $50."},
            {"role": "assistant", "source_type": "agent_message", "content": "Decision: choose the quiet room for the workshop."},
            {"role": "tool", "source_type": "tool_result", "content": "Calendar lookup finished with three open slots."},
        ],
        namespace_id=namespace,
        user_id="alex",
        agent_id="assistant",
        session_id="chunk-session",
    )
    engine.add_episode(
        namespace_id=namespace,
        source_type="imported_doc",
        content_raw="Imported design note: Retrieval should cite atom-sized evidence.",
    )

    with engine.session_factory() as session:
        chunks = session.scalars(select(EpisodeChunk).where(EpisodeChunk.namespace_id == namespace)).all()

    chunk_types = {chunk.chunk_type for chunk in chunks}
    assert {"preference", "constraint", "decision", "tool_result", "reasoning_note"} <= chunk_types
    assert all(chunk.summary and chunk.embedding for chunk in chunks)


def test_consolidation_creates_descriptors_and_export_fields(engine):
    namespace = "test.v2.descriptors"
    engine.add_messages(
        [{"content": "I prefer oat milk for Orion project cafes.", "source_type": "user_message"}],
        namespace_id=namespace,
        user_id="alex",
        agent_id="assistant",
        session_id="descriptor-session",
    )
    engine.consolidate(namespace)

    with engine.session_factory() as session:
        fact_descriptors = session.scalars(
            select(NodeDescriptor).where(
                NodeDescriptor.namespace_id == namespace,
                NodeDescriptor.node_type == NodeType.FACT.value,
            )
        ).all()
        edge_descriptors = session.scalars(select(EdgeDescriptor)).all()
        evidence_links = session.scalars(select(ChunkEvidenceLink).where(ChunkEvidenceLink.namespace_id == namespace)).all()

    graph = engine.export_graph(namespace_id=namespace)

    assert any(descriptor.node_class == "preference" for descriptor in fact_descriptors)
    assert any(descriptor.relation_class == "supports" for descriptor in edge_descriptors)
    assert evidence_links
    assert any("node_class" in node and "evidence_count" in node for node in graph["nodes"])
    assert any("relation_class" in edge and "confidence" in edge for edge in graph["edges"])


def test_retrieval_and_activation_surface_precise_atoms_with_slots(engine):
    namespace = "test.v2.slots"
    engine.add_messages(
        [
            {
                "content": "Planning notes: choose the green badge design. The lunch constraint is vegetarian only.",
                "source_type": "user_message",
            }
        ],
        namespace_id=namespace,
        user_id="alex",
        agent_id="assistant",
        session_id="slot-session",
    )
    engine.consolidate(namespace)

    search = engine.search("What was the vegetarian lunch constraint?", namespace_id=namespace, limit=5)
    run = engine.start_run(namespace_id=namespace, user_id="alex", agent_id="assistant", session_id="slot-session")
    result = engine.process_step_result(
        run["run_id"],
        step_type="user_input",
        text="What was the vegetarian lunch constraint?",
        metadata={"include_provenance": True},
    )
    packet = build_memory_context_packet(
        run_id=run["run_id"],
        step_index=result["step_index"],
        query="What was the vegetarian lunch constraint?",
        working_memory=result["working_memory"],
        limit=4,
    )
    rendered = render_memory_context_packet(packet)

    assert any(item["node_key"].startswith("EpisodeChunk:") for item in search[:5])
    assert result["working_memory"]
    assert {item["slot"] for item in result["working_memory"]} <= {"primary", "linked", "ambient", "evidence"}
    assert all("node_class" in item and "evidence_ids" in item for item in result["working_memory"])
    assert packet["slots"]
    assert "<memoria_context>" in rendered


def test_sidecar_and_workbench_return_memory_context_packets(engine):
    sidecar = MemoriaSidecar(engine)
    namespace = "test.v2.packet"
    sidecar.ingest_messages(
        namespace_id=namespace,
        user_id="alex",
        agent_id="assistant",
        session_id="packet-session",
        messages=[{"role": "user", "content": "Remember that I prefer quiet cafes with oat milk."}],
    )
    recall = sidecar.recall(
        namespace_id=namespace,
        user_id="alex",
        agent_id="assistant",
        session_id="packet-session",
        text="Plan a quiet cafe meeting.",
    )

    service = WorkbenchService(engine)
    run = service.create_run(
        RunSpec(
            title="Packet run",
            provider=ProviderConfig(provider_type="replay", model_name="replay", replay_responses=["Stored."]),
            namespace_id="test.v2.workbench",
            prompt_limit=4,
        )
    )
    turn = service.send_user_message(run["id"], "Remember that Orion check-ins need oat milk.")

    assert recall["memory_context_packet"]["version"] == "memory-context-packet-v1"
    assert recall["prompt_addition"].startswith("<memoria_context>")
    assert turn["turn"]["memory_context_packet"]["version"] == "memory-context-packet-v1"


def test_memoryarena_experiment_cli_writes_scorecard(tmp_path: Path):
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    result = runner.invoke(
        app,
        [
            "memoryarena-experiment",
            "--label",
            "v2-test",
            "--db",
            f"sqlite:///{tmp_path / 'experiment.db'}",
            "--data-root",
            str(FIXTURE_ROOT),
            "--suite",
            "formal_reasoning_phys",
            "--family",
            "snapshot_lookup",
            "--strand",
            "paper_context_recall",
            "--artifact-root",
            str(artifact_root),
            "--jobs",
            "1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    scorecard = payload["scorecard"]
    assert scorecard["label"] == "v2-test"
    assert scorecard["snapshot_lookup"] >= 1
    assert "paper_context_recall" in scorecard["strand_counts"]
    assert (Path(payload["artifact_dir"]) / "scorecard.json").exists()
