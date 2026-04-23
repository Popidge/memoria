from __future__ import annotations

from pathlib import Path
import json

from typer.testing import CliRunner

import memoria.cli as cli_module
from memoria.cli import app
from memoria.memoryarena import (
    _classify_openclaw_response,
    _parse_json_output,
    _trace_step_type,
    build_memoryarena_derived_manifest,
    evaluate_memoryarena_agent,
    evaluate_memoryarena_offline,
    load_memoryarena_corpus,
    materialize_memoryarena_derived_manifest,
    write_public_memoryarena_summary,
)
from memoria.providers import ProviderConfig
from memoria.workbench import WorkbenchService


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "memoryarena"


def test_load_memoryarena_corpus_from_fixtures():
    corpus = load_memoryarena_corpus(data_root=FIXTURE_ROOT)

    assert corpus.source == str(FIXTURE_ROOT)
    assert corpus.counts["group_travel_planner"] == 1
    travel_task = next(task for task in corpus.tasks if task.suite == "group_travel_planner")
    assert "Jennifer" in travel_task.background_items[0]
    assert "Base person itinerary:" in travel_task.background_items[1]
    math_task = next(task for task in corpus.tasks if task.suite == "formal_reasoning_math")
    assert math_task.turns[1].background_items == ["Use the prior simplification when answering the follow-up."]


def test_build_memoryarena_manifest_and_materialize(tmp_path: Path):
    corpus = load_memoryarena_corpus(data_root=FIXTURE_ROOT)
    manifest = build_memoryarena_derived_manifest(corpus)

    assert manifest.manifest_version == "memoryarena-derived-v1"
    assert len(manifest.cases) == 20
    assert manifest.cases[0].case_id == "snapshot_lookup:bundled_shopping:0:turn_0"
    assert {case.strand for case in manifest.cases if case.suite == "bundled_shopping"} == {
        "incremental_state_tracking",
        "compatibility_constraints",
    }

    built = materialize_memoryarena_derived_manifest(corpus, artifact_root=tmp_path / "artifacts")

    assert Path(built["files"]["snapshot_lookup"]).exists()
    assert Path(built["files"]["learn_as_you_act"]).exists()
    assert built["summary"]["family_counts"] == {
        "snapshot_lookup": 10,
        "learn_as_you_act": 10,
    }


def test_offline_memoryarena_eval_writes_family_metrics(engine, tmp_path: Path):
    corpus = load_memoryarena_corpus(data_root=FIXTURE_ROOT)
    manifest = build_memoryarena_derived_manifest(corpus)

    result = evaluate_memoryarena_offline(
        engine,
        manifest,
        artifact_root=tmp_path / "artifacts",
    )

    assert result["summary"]["mode"] == "memoryarena_offline"
    assert result["summary"]["case_count"] == 20
    assert result["summary"]["family_case_counts"] == {
        "learn_as_you_act": 10,
        "snapshot_lookup": 10,
    }
    assert result["summary"]["write_success_rate"] is not None
    assert result["summary"]["deferred_recall_at_k"] is not None
    assert result["summary"]["corpus_learned_coverage"] is not None
    assert result["artifact_dir"] is not None
    cases_path = Path(result["artifact_dir"]) / "cases.jsonl"
    summary_path = Path(result["artifact_dir"]) / "summary.json"
    assert cases_path.exists()
    assert summary_path.exists()
    second_math_case = next(
        case
        for case in result["cases"]
        if case["family"] == "learn_as_you_act"
        and case["suite"] == "formal_reasoning_math"
        and case["turn_index"] == 1
    )
    assert second_math_case["support_expected_count"] >= 1
    assert second_math_case["deferred_recall_at_k"] is not None
    snapshot_case = next(case for case in result["cases"] if case["family"] == "snapshot_lookup")
    assert snapshot_case["trace"]["mode"] == "summary"
    assert snapshot_case["corpus_expected_artifact_count"] >= snapshot_case["corpus_learned_artifact_count"]


def test_offline_memoryarena_eval_legacy_full_trace(engine):
    corpus = load_memoryarena_corpus(data_root=FIXTURE_ROOT, suites=["formal_reasoning_math"])
    manifest = build_memoryarena_derived_manifest(corpus)

    result = evaluate_memoryarena_offline(
        engine,
        manifest,
        strategy="legacy",
        trace_mode="full",
    )

    assert result["config"]["strategy"] == "legacy"
    assert result["config"]["trace_mode"] == "full"
    assert result["summary"]["family_case_counts"] == {
        "learn_as_you_act": 2,
        "snapshot_lookup": 2,
    }
    assert "mode" not in result["cases"][0]["trace"]
    assert result["cases"][0]["trace"]["steps"]


def test_offline_memoryarena_eval_parallel_jobs_match_serial(tmp_path: Path):
    corpus = load_memoryarena_corpus(data_root=FIXTURE_ROOT)
    manifest = build_memoryarena_derived_manifest(corpus)
    from memoria.config import EngineConfig
    from memoria.engine import MemoryEngine

    serial_engine = MemoryEngine(
        config=EngineConfig(database_url=f"sqlite:///{tmp_path / 'serial.db'}", use_sentence_transformers=False)
    )
    parallel_engine = MemoryEngine(
        config=EngineConfig(database_url=f"sqlite:///{tmp_path / 'parallel.db'}", use_sentence_transformers=False)
    )

    serial = evaluate_memoryarena_offline(serial_engine, manifest)
    parallel = evaluate_memoryarena_offline(parallel_engine, manifest, jobs=2)

    assert parallel["config"]["worker_count"] == 2
    assert parallel["summary"]["case_count"] == serial["summary"]["case_count"]
    assert [case["case_id"] for case in parallel["cases"]] == [case["case_id"] for case in serial["cases"]]
    assert parallel["summary"]["family_case_counts"] == serial["summary"]["family_case_counts"]


def test_memoryarena_agent_eval_replay_smoke(engine, tmp_path: Path):
    corpus = load_memoryarena_corpus(
        data_root=FIXTURE_ROOT,
        suites=["group_travel_planner", "formal_reasoning_math", "formal_reasoning_phys"],
    )
    manifest = build_memoryarena_derived_manifest(corpus)
    service = WorkbenchService(engine)

    result = evaluate_memoryarena_agent(
        service,
        manifest,
        provider_config=ProviderConfig(provider_type="replay", model_name="replay"),
        artifact_root=tmp_path / "artifacts",
    )

    assert result["summary"]["mode"] == "memoryarena_agent"
    assert result["summary"]["case_count"] == 12
    assert result["summary"]["token_f1"] == 1.0
    assert result["summary"]["exact_match_rate"] == 1.0
    assert result["artifact_dir"] is not None
    travel_case = next(case for case in result["cases"] if case["suite"] == "group_travel_planner")
    assert travel_case["field_coverage"] == 1.0


def test_memoryarena_eval_cli_accepts_fixture_root(tmp_path: Path):
    runner = CliRunner()
    db_path = tmp_path / "cli.db"
    artifact_root = tmp_path / "bench-artifacts"

    result = runner.invoke(
        app,
        [
            "memoryarena-eval",
            "--db",
            f"sqlite:///{db_path}",
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
            "--strategy",
            "legacy",
            "--trace-mode",
            "summary",
        ],
    )

    assert result.exit_code == 0
    assert '"mode": "memoryarena_offline"' in result.stdout


def test_memoryarena_build_and_agent_eval_cli_accept_fixture_root(tmp_path: Path):
    runner = CliRunner()
    db_path = tmp_path / "cli-agent.db"
    artifact_root = tmp_path / "bench-artifacts"

    build_result = runner.invoke(
        app,
        [
            "memoryarena-build",
            "--data-root",
            str(FIXTURE_ROOT),
            "--artifact-root",
            str(artifact_root),
        ],
    )
    agent_result = runner.invoke(
        app,
        [
            "memoryarena-agent-eval",
            "--db",
            f"sqlite:///{db_path}",
            "--data-root",
            str(FIXTURE_ROOT),
            "--suite",
            "formal_reasoning_math",
            "--provider-type",
            "replay",
            "--artifact-root",
            str(artifact_root),
        ],
    )

    assert build_result.exit_code == 0
    assert '"snapshot_lookup"' in build_result.stdout
    assert agent_result.exit_code == 0
    assert '"mode": "memoryarena_agent"' in agent_result.stdout


def test_openclaw_response_classification_and_trace_step_type():
    reply_text, reply_format = _classify_openclaw_response({"payloads": [{"text": "hello world"}]})
    assert reply_text == "hello world"
    assert reply_format == "text"

    empty_text, empty_format = _classify_openclaw_response({})
    assert empty_text == ""
    assert empty_format == "empty_payload"

    step_type = _trace_step_type(
        {
            "activation": [
                {
                    "reason": {
                        "step_type": "assistant_message",
                    }
                }
            ]
        }
    )
    assert step_type == "assistant_message"

    payload = _parse_json_output("", '{"payloads":[{"text":"hello"}]}')
    assert payload["payloads"][0]["text"] == "hello"

    payload = _parse_json_output("", '[agent] run ended\n{"payloads":[{"text":"hello again"}]}')
    assert payload["payloads"][0]["text"] == "hello again"


def test_write_public_memoryarena_summary(tmp_path: Path):
    summary_path = tmp_path / "docs" / "benchmarks" / "memoryarena-latest.json"
    payload = {
        "generated_at": "2026-04-09T18:00:00+00:00",
        "selected_counts": {
            "task_count": 3,
            "turn_count": 9,
        },
        "modes": {
            "baseline": {
                "summary": {
                    "token_f1": 0.1,
                }
            }
        },
    }

    write_public_memoryarena_summary(summary_path, payload)

    assert summary_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8")) == payload


def test_memoryarena_compare_cli_writes_summary(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    summary_path = tmp_path / "memoryarena-latest.json"

    def fake_compare(*args, **kwargs):
        write_public_memoryarena_summary(
            kwargs["public_summary_path"],
            {
                "generated_at": "2026-04-09T18:00:00+00:00",
                "modes": {},
            },
        )
        return {
            "summary": {
                "generated_at": "2026-04-09T18:00:00+00:00",
                "modes": {},
            },
            "modes": {},
            "public_summary_path": str(kwargs["public_summary_path"]),
        }

    monkeypatch.setattr(cli_module, "compare_memoryarena_openclaw", fake_compare)

    result = runner.invoke(
        app,
        [
            "memoryarena-compare",
            "--data-root",
            str(FIXTURE_ROOT),
            "--suite",
            "group_travel_planner",
            "--public-summary-path",
            str(summary_path),
        ],
    )

    assert result.exit_code == 0
    assert summary_path.exists()
