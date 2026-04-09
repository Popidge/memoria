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
    evaluate_memoryarena_proxy,
    load_memoryarena_corpus,
    write_public_memoryarena_summary,
)


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


def test_proxy_evaluation_writes_artifacts(engine, tmp_path: Path):
    corpus = load_memoryarena_corpus(
        data_root=FIXTURE_ROOT,
        suites=["group_travel_planner", "formal_reasoning_math"],
    )

    result = evaluate_memoryarena_proxy(
        engine,
        corpus,
        artifact_root=tmp_path / "artifacts",
    )

    assert result["summary"]["mode"] == "memoryarena_proxy"
    assert result["summary"]["case_count"] == 4
    assert result["config"]["available_counts"] == {
        "group_travel_planner": 1,
        "formal_reasoning_math": 1,
    }
    assert result["config"]["selected_counts"]["task_count"] == 2
    assert result["config"]["selected_counts"]["turn_count"] == 4
    assert result["artifact_dir"] is not None
    cases_path = Path(result["artifact_dir"]) / "cases.jsonl"
    summary_path = Path(result["artifact_dir"]) / "summary.json"
    assert cases_path.exists()
    assert summary_path.exists()
    second_math_case = next(
        case for case in result["cases"] if case["suite"] == "formal_reasoning_math" and case["turn_index"] == 1
    )
    assert second_math_case["support_expected_count"] >= 1


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
            "--artifact-root",
            str(artifact_root),
        ],
    )

    assert result.exit_code == 0
    assert '"mode": "memoryarena_proxy"' in result.stdout


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
