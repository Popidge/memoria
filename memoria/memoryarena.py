from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired, run
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import json
import os
import shutil

from memoria.context import build_memory_context_packet, render_memory_context_packet
from memoria.engine import MemoryEngine
from memoria.utils import STOPWORDS, tokenize, word_count


DATASET_NAME = "ZexueHe/memoryarena"
DEFAULT_REVISION = "main"
ALL_SUITES = (
    "bundled_shopping",
    "progressive_search",
    "group_travel_planner",
    "formal_reasoning_math",
    "formal_reasoning_phys",
)
LIVE_SUPPORTED_SUITES = (
    "group_travel_planner",
    "formal_reasoning_math",
    "formal_reasoning_phys",
)
LIVE_MEMORY_MODES = (
    "baseline",
    "native",
    "prefetch",
)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "memoria" / "memoryarena"
DEFAULT_ARTIFACT_ROOT = Path("artifacts") / "benchmarks" / "memoryarena"
DEFAULT_SMOKE_LIMIT = 1
PUBLIC_SUMMARY_PATH = Path("docs") / "benchmarks" / "memoryarena-latest.json"
DERIVED_MANIFEST_VERSION = "memoryarena-derived-v1"
DERIVED_FAMILIES = (
    "snapshot_lookup",
    "learn_as_you_act",
)
SUPPORT_K = 5
OFFLINE_STRATEGIES = ("hybrid", "legacy")
OFFLINE_TRACE_MODES = ("summary", "full", "failures")

SUITE_STRANDS = {
    "bundled_shopping": ("incremental_state_tracking", "compatibility_constraints"),
    "progressive_search": ("entity_accumulation", "composed_fact_lookup"),
    "group_travel_planner": ("static_background_recall", "structured_plan_continuity"),
    "formal_reasoning_math": ("paper_context_recall", "reasoning_carryover"),
    "formal_reasoning_phys": ("paper_context_recall", "reasoning_carryover"),
}

BENCHMARK_WORKSPACE_FILES = {
    "AGENTS.md": """# AGENTS.md

Use this workspace only for isolated MemoryArena benchmark runs.

Read `SOUL.md`, `IDENTITY.md`, and `USER.md` at session start.
Do not read or write any external memory files outside this workspace.
""",
    "SOUL.md": """# SOUL.md

Be concise, careful, and helpful. Answer directly.
Prefer correctness over style flourishes.
""",
    "TOOLS.md": """# TOOLS.md

This workspace is benchmark-only. Avoid tool use unless strictly required.
""",
    "IDENTITY.md": """# IDENTITY.md

Name: Garland
Role: Benchmark harness assistant for MemoryArena evaluation.
""",
    "USER.md": """# USER.md

You are helping with isolated benchmark prompts. Treat all context as task data.
""",
    "HEARTBEAT.md": "# Keep empty during benchmarks.\n",
    "MEMORY.md": "# No manual long-term workspace memory for benchmark runs.\n",
}


@dataclass
class MemoryArenaTurn:
    suite: str
    task_id: str
    turn_index: int
    question: str
    gold_answer_text: str
    gold_answer_value: Any
    background_items: list[str]
    metadata: dict[str, Any]


@dataclass
class MemoryArenaTask:
    suite: str
    task_id: str
    turns: list[MemoryArenaTurn]
    background_items: list[str]
    metadata: dict[str, Any]


@dataclass
class MemoryArenaCorpus:
    dataset_name: str
    revision: str
    source: str
    suites: list[str]
    counts: dict[str, int]
    tasks: list[MemoryArenaTask]


@dataclass
class SupportArtifact:
    artifact_id: str
    kind: str
    text: str
    overlap: float


@dataclass
class MemoryArenaArtifact:
    artifact_id: str
    kind: str
    text: str
    value: Any
    metadata: dict[str, Any]


@dataclass
class MemoryArenaDerivedCase:
    case_id: str
    family: str
    suite: str
    strand: str
    task_id: str
    turn_index: int
    question: str
    gold_answer_text: str
    gold_answer_value: Any
    static_artifacts: list[dict[str, Any]]
    dynamic_artifacts_before_turn: list[dict[str, Any]]
    expected_support_ids: list[str]
    scorer: str
    scorer_payload: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class MemoryArenaDerivedManifest:
    manifest_version: str
    dataset_name: str
    revision: str
    source: str
    suites: list[str]
    counts: dict[str, int]
    cases: list[MemoryArenaDerivedCase]


def _timestamp_slug(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def _normalise_offline_jobs(jobs: int | str) -> int:
    if isinstance(jobs, str) and jobs.strip().lower() == "auto":
        return max(1, os.cpu_count() or 1)
    try:
        parsed = int(jobs)
    except (TypeError, ValueError) as exc:
        raise ValueError("jobs must be a positive integer or 'auto'") from exc
    if parsed < 1:
        raise ValueError("jobs must be a positive integer or 'auto'")
    return parsed


def _normalise_scalar(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def _serialise_plan_items(items: Iterable[dict[str, Any]]) -> str:
    rendered = []
    for item in items:
        days = item.get("days", "?")
        current_city = item.get("current_city", "?")
        transportation = item.get("transportation", "?")
        rendered.append(f"Day {days}: {current_city}; transport: {transportation}")
    return "\n".join(rendered)


def serialise_answer(value: Any) -> str:
    if isinstance(value, list) and value and all(isinstance(item, dict) and "days" in item for item in value):
        return _serialise_plan_items(value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    if isinstance(value, list):
        return "\n".join(serialise_answer(item) for item in value)
    return _normalise_scalar(value)


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("id", "unknown"))


def _background_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [item for item in (str(entry).strip() for entry in value) if item]
    return [_normalise_scalar(value)]


def _travel_backgrounds(row: dict[str, Any]) -> list[str]:
    base_person = row.get("base_person") or {}
    rendered: list[str] = []
    query = str(base_person.get("query") or "").strip()
    if query:
        rendered.append(query)
    plans = base_person.get("daily_plans") or []
    if isinstance(plans, list) and plans:
        rendered.append("Base person itinerary:\n" + _serialise_plan_items(plans))
    return rendered


def _turn_backgrounds(suite: str, row: dict[str, Any], turn_index: int, task_backgrounds: list[str]) -> list[str]:
    if suite == "group_travel_planner":
        return task_backgrounds
    backgrounds = row.get("backgrounds")
    if isinstance(backgrounds, list):
        if turn_index < len(backgrounds):
            candidate = str(backgrounds[turn_index]).strip()
            if candidate:
                return [candidate]
        return []
    return _background_list(backgrounds)


def normalise_memoryarena_task(suite: str, row: dict[str, Any]) -> MemoryArenaTask:
    task_id = _row_id(row)
    questions = list(row.get("questions") or [])
    answers = list(row.get("answers") or [])
    if len(questions) != len(answers):
        raise ValueError(f"{suite}:{task_id} has mismatched questions/answers lengths")
    if suite == "group_travel_planner":
        task_backgrounds = _travel_backgrounds(row)
    else:
        task_backgrounds = _background_list(row.get("backgrounds"))
    metadata = {
        "category": row.get("category"),
        "paper_name": row.get("paper_name"),
    }
    if suite == "group_travel_planner":
        base_person = row.get("base_person") or {}
        metadata["base_person_name"] = base_person.get("name")
    turns = [
        MemoryArenaTurn(
            suite=suite,
            task_id=task_id,
            turn_index=turn_index,
            question=str(question),
            gold_answer_text=serialise_answer(answer),
            gold_answer_value=answer,
            background_items=_turn_backgrounds(suite, row, turn_index, task_backgrounds),
            metadata={k: v for k, v in metadata.items() if v is not None},
        )
        for turn_index, (question, answer) in enumerate(zip(questions, answers, strict=True))
    ]
    return MemoryArenaTask(
        suite=suite,
        task_id=task_id,
        turns=turns,
        background_items=task_backgrounds,
        metadata={k: v for k, v in metadata.items() if v is not None},
    )


def _read_fixture_rows(data_root: Path, suite: str) -> list[dict[str, Any]]:
    path = data_root / suite / "data.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing fixture file for suite {suite}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_hf_rows(cache_dir: Path, suite: str, revision: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "MemoryArena support requires the optional benchmark dependencies. "
            "Install them with `uv sync --extra benchmark`."
        ) from exc
    dataset = load_dataset(DATASET_NAME, suite, split="test", revision=revision, cache_dir=str(cache_dir))
    return [dict(row) for row in dataset]


def load_memoryarena_corpus(
    *,
    suites: list[str] | None = None,
    revision: str = DEFAULT_REVISION,
    cache_dir: Path | None = None,
    data_root: Path | None = None,
    limit_per_suite: int | None = None,
) -> MemoryArenaCorpus:
    selected_suites = suites or list(ALL_SUITES)
    counts: dict[str, int] = {}
    tasks: list[MemoryArenaTask] = []
    if data_root is not None:
        source = str(data_root)
    else:
        source = DATASET_NAME
        cache_dir = cache_dir or DEFAULT_CACHE_DIR
    for suite in selected_suites:
        if suite not in ALL_SUITES:
            raise KeyError(f"Unknown MemoryArena suite: {suite}")
        rows = _read_fixture_rows(data_root, suite) if data_root is not None else _load_hf_rows(cache_dir, suite, revision)
        counts[suite] = len(rows)
        if limit_per_suite is not None:
            rows = rows[:limit_per_suite]
        tasks.extend(normalise_memoryarena_task(suite, row) for row in rows)
    return MemoryArenaCorpus(
        dataset_name=DATASET_NAME,
        revision=revision,
        source=source,
        suites=selected_suites,
        counts=counts,
        tasks=tasks,
    )


def _suite_strand(suite: str, turn_index: int) -> str:
    first_turn, later_turn = SUITE_STRANDS[suite]
    return first_turn if turn_index == 0 else later_turn


def _artifact_to_dict(artifact: MemoryArenaArtifact) -> dict[str, Any]:
    return {
        "artifact_id": artifact.artifact_id,
        "kind": artifact.kind,
        "text": artifact.text,
        "value": artifact.value,
        "metadata": artifact.metadata,
    }


def _case_to_dict(case: MemoryArenaDerivedCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "family": case.family,
        "suite": case.suite,
        "strand": case.strand,
        "task_id": case.task_id,
        "turn_index": case.turn_index,
        "question": case.question,
        "gold_answer_text": case.gold_answer_text,
        "gold_answer_value": case.gold_answer_value,
        "static_artifacts": case.static_artifacts,
        "dynamic_artifacts_before_turn": case.dynamic_artifacts_before_turn,
        "expected_support_ids": case.expected_support_ids,
        "scorer": case.scorer,
        "scorer_payload": case.scorer_payload,
        "metadata": case.metadata,
    }


def _build_travel_static_artifacts(task: MemoryArenaTask) -> list[MemoryArenaArtifact]:
    artifacts: list[MemoryArenaArtifact] = []
    for index, background in enumerate(task.background_items):
        kind = "travel_base_query" if index == 0 else "travel_base_plan"
        artifacts.append(
            MemoryArenaArtifact(
                artifact_id=f"{task.suite}:{task.task_id}:static:{kind}",
                kind=kind,
                text=background,
                value=background,
                metadata={"suite": task.suite, "task_id": task.task_id, "source": "task_background"},
            )
        )
    return artifacts


def _build_formal_static_artifacts(task: MemoryArenaTask, turn: MemoryArenaTurn) -> list[MemoryArenaArtifact]:
    return [
        MemoryArenaArtifact(
            artifact_id=f"{task.suite}:{task.task_id}:static:turn_{turn.turn_index}:{index}",
            kind="paper_context",
            text=background,
            value=background,
            metadata={
                "suite": task.suite,
                "task_id": task.task_id,
                "turn_index": turn.turn_index,
                "paper_name": turn.metadata.get("paper_name"),
            },
        )
        for index, background in enumerate(turn.background_items)
    ]


def _shopping_answer_artifact_text(answer_value: Any) -> str:
    if not isinstance(answer_value, dict):
        return serialise_answer(answer_value)
    asin = str(answer_value.get("target_asin") or "unknown").strip()
    attributes = [
        str(item).strip()
        for item in (answer_value.get("attributes") or [])
        if str(item).strip()
    ]
    if not attributes:
        return f"Selected product {asin}."
    return f"Selected product {asin} with attributes: {', '.join(attributes)}."


def _build_dynamic_artifact(task: MemoryArenaTask, turn: MemoryArenaTurn) -> MemoryArenaArtifact:
    if task.suite == "bundled_shopping":
        text = _shopping_answer_artifact_text(turn.gold_answer_value)
        kind = "shopping_selection"
    elif task.suite == "group_travel_planner":
        text = f"Traveler plan:\n{serialise_answer(turn.gold_answer_value)}"
        kind = "travel_plan"
    elif task.suite in {"formal_reasoning_math", "formal_reasoning_phys"}:
        paper_name = turn.metadata.get("paper_name")
        prefix = f"{paper_name}: " if paper_name else ""
        text = f"{prefix}{turn.gold_answer_text}".strip()
        kind = "reasoning_answer"
    else:
        text = turn.gold_answer_text
        kind = "search_result"
    return MemoryArenaArtifact(
        artifact_id=f"{task.suite}:{task.task_id}:dynamic:turn_{turn.turn_index}",
        kind=kind,
        text=text,
        value=turn.gold_answer_value,
        metadata={
            "suite": task.suite,
            "task_id": task.task_id,
            "turn_index": turn.turn_index,
            "question": turn.question,
        },
    )


def _static_artifacts_for_turn(task: MemoryArenaTask, turn: MemoryArenaTurn) -> list[MemoryArenaArtifact]:
    if task.suite == "group_travel_planner":
        return _build_travel_static_artifacts(task)
    if task.suite in {"formal_reasoning_math", "formal_reasoning_phys"}:
        return _build_formal_static_artifacts(task, turn)
    return []


def _expected_support_ids(
    suite: str,
    static_artifacts: list[MemoryArenaArtifact],
    dynamic_artifacts: list[MemoryArenaArtifact],
) -> list[str]:
    if suite in {"bundled_shopping", "progressive_search"}:
        return [artifact.artifact_id for artifact in dynamic_artifacts]
    support_ids = [artifact.artifact_id for artifact in static_artifacts]
    support_ids.extend(artifact.artifact_id for artifact in dynamic_artifacts)
    return support_ids


def _scorer_for_turn(turn: MemoryArenaTurn) -> tuple[str, dict[str, Any]]:
    if turn.suite == "group_travel_planner":
        return (
            "travel_field_coverage",
            {
                "field_values": _travel_field_values(turn.gold_answer_value),
            },
        )
    return (
        "token_f1",
        {
            "gold_answer_text": turn.gold_answer_text,
        },
    )


def build_memoryarena_derived_manifest(corpus: MemoryArenaCorpus) -> MemoryArenaDerivedManifest:
    cases: list[MemoryArenaDerivedCase] = []
    for task in corpus.tasks:
        prior_dynamic: list[MemoryArenaArtifact] = []
        for turn in task.turns:
            static_artifacts = _static_artifacts_for_turn(task, turn)
            expected_support_ids = _expected_support_ids(task.suite, static_artifacts, prior_dynamic)
            scorer, scorer_payload = _scorer_for_turn(turn)
            strand = _suite_strand(task.suite, turn.turn_index)
            metadata = {
                **task.metadata,
                **turn.metadata,
                "source_dataset": corpus.dataset_name,
                "source_revision": corpus.revision,
                "task_background_count": len(task.background_items),
                "turn_background_count": len(turn.background_items),
            }
            for family in DERIVED_FAMILIES:
                cases.append(
                    MemoryArenaDerivedCase(
                        case_id=f"{family}:{task.suite}:{task.task_id}:turn_{turn.turn_index}",
                        family=family,
                        suite=task.suite,
                        strand=strand,
                        task_id=task.task_id,
                        turn_index=turn.turn_index,
                        question=turn.question,
                        gold_answer_text=turn.gold_answer_text,
                        gold_answer_value=turn.gold_answer_value,
                        static_artifacts=[_artifact_to_dict(item) for item in static_artifacts],
                        dynamic_artifacts_before_turn=[_artifact_to_dict(item) for item in prior_dynamic],
                        expected_support_ids=list(expected_support_ids),
                        scorer=scorer,
                        scorer_payload=scorer_payload,
                        metadata=metadata,
                    )
                )
            prior_dynamic.append(_build_dynamic_artifact(task, turn))
    return MemoryArenaDerivedManifest(
        manifest_version=DERIVED_MANIFEST_VERSION,
        dataset_name=corpus.dataset_name,
        revision=corpus.revision,
        source=corpus.source,
        suites=corpus.suites,
        counts=corpus.counts,
        cases=cases,
    )


def _normalise_family_selection(family: str | None) -> list[str]:
    selected = (family or "all").strip()
    if selected == "all":
        return list(DERIVED_FAMILIES)
    if selected not in DERIVED_FAMILIES:
        raise KeyError(f"Unknown MemoryArena family: {selected}")
    return [selected]


def select_memoryarena_cases(
    manifest: MemoryArenaDerivedManifest,
    *,
    family: str | None = None,
    suites: list[str] | None = None,
    strands: list[str] | None = None,
) -> list[MemoryArenaDerivedCase]:
    selected_families = set(_normalise_family_selection(family))
    selected_suites = set(suites or [])
    selected_strands = set(strands or [])
    return [
        case
        for case in manifest.cases
        if case.family in selected_families
        and (not selected_suites or case.suite in selected_suites)
        and (not selected_strands or case.strand in selected_strands)
    ]


def materialize_memoryarena_derived_manifest(
    corpus: MemoryArenaCorpus,
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    manifest = build_memoryarena_derived_manifest(corpus)
    derived_root = artifact_root / "derived"
    derived_root.mkdir(parents=True, exist_ok=True)
    family_paths: dict[str, str] = {}
    family_counts: dict[str, int] = {}
    for family in DERIVED_FAMILIES:
        family_cases = [case for case in manifest.cases if case.family == family]
        family_counts[family] = len(family_cases)
        path = derived_root / f"{family}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for case in family_cases:
                handle.write(json.dumps(_case_to_dict(case), ensure_ascii=True) + "\n")
        family_paths[family] = str(path)
    summary = {
        "manifest_version": manifest.manifest_version,
        "dataset_name": manifest.dataset_name,
        "revision": manifest.revision,
        "source": manifest.source,
        "suites": manifest.suites,
        "available_counts": manifest.counts,
        "family_counts": family_counts,
        "suite_case_counts": {
            suite: sum(1 for case in manifest.cases if case.suite == suite)
            for suite in manifest.suites
        },
        "strand_case_counts": {
            strand: sum(1 for case in manifest.cases if case.strand == strand)
            for strand in sorted({case.strand for case in manifest.cases})
        },
    }
    summary_path = derived_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "derived_root": str(derived_root),
        "summary_path": str(summary_path),
        "summary": summary,
        "files": family_paths,
    }


def _content_to_text(item: dict[str, Any]) -> str:
    return str(item.get("content") or item.get("text") or "")


def _prompt_addition(working_memory: list[dict[str, Any]], limit: int = 4) -> str:
    packet = build_memory_context_packet(
        run_id="memoryarena",
        step_index=None,
        query="",
        working_memory=working_memory,
        limit=limit,
    )
    rendered = render_memory_context_packet(packet)
    if rendered:
        return rendered
    lines = ["Relevant Memoria recall for this turn:"]
    for item in working_memory[:limit]:
        content_type = str(item.get("content_type") or "memory")
        lines.append(f"- [{content_type}] {_content_to_text(item)}")
    if len(lines) == 1:
        return ""
    lines.append("Use only the items that are directly relevant to the current task.")
    return "\n".join(lines)


def _normalise_tokens(text: str) -> set[str]:
    return {
        token
        for token in tokenize(text)
        if token not in STOPWORDS and (len(token) >= 3 or token.isdigit())
    }


def _overlap_score(left: str, right: str) -> float:
    left_tokens = _normalise_tokens(left)
    right_tokens = _normalise_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _artifact_hit(artifact_text: str, candidate_text: str, threshold: float = 0.18) -> bool:
    if artifact_text.strip() and artifact_text.strip() in candidate_text:
        return True
    return _overlap_score(artifact_text, candidate_text) >= threshold


def _expected_support_artifacts(
    prior_artifacts: list[dict[str, str]],
    gold_answer_text: str,
    *,
    min_overlap: float = 0.18,
    limit: int = 5,
) -> list[SupportArtifact]:
    expected = [
        SupportArtifact(
            artifact_id=artifact["artifact_id"],
            kind=artifact["kind"],
            text=artifact["text"],
            overlap=_overlap_score(artifact["text"], gold_answer_text),
        )
        for artifact in prior_artifacts
    ]
    expected = [artifact for artifact in expected if artifact.overlap >= min_overlap]
    expected.sort(key=lambda artifact: (-artifact.overlap, artifact.artifact_id))
    return expected[:limit]


def _safe_mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _summary_from_cases(mode: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    ok_cases = [case for case in cases if case.get("status") == "ok"]
    return {
        "mode": mode,
        "case_count": len(cases),
        "family_case_counts": {
            family: sum(1 for case in cases if case.get("family") == family)
            for family in sorted({case.get("family") for case in cases if case.get("family") is not None})
        },
        "suite_case_counts": {
            suite: sum(1 for case in cases if case["suite"] == suite)
            for suite in sorted({case["suite"] for case in cases})
        },
        "strand_case_counts": {
            strand: sum(1 for case in cases if case.get("strand") == strand)
            for strand in sorted({case.get("strand") for case in cases if case.get("strand") is not None})
        },
        "support_recall_at_5": _safe_mean([case.get("support_recall_at_5") for case in cases]),
        "support_precision_at_5": _safe_mean([case.get("support_precision_at_5") for case in cases]),
        "support_recall_at_k": _safe_mean([case.get("support_recall_at_k") for case in cases]),
        "support_precision_at_k": _safe_mean([case.get("support_precision_at_k") for case in cases]),
        "prompt_support_coverage": _safe_mean([case.get("prompt_support_coverage") for case in cases]),
        "write_success_rate": _safe_mean([case.get("write_success_rate") for case in cases]),
        "deferred_recall_at_5": _safe_mean([case.get("deferred_recall_at_5") for case in cases]),
        "deferred_recall_at_k": _safe_mean([case.get("deferred_recall_at_k") for case in cases]),
        "token_f1": _safe_mean([case.get("token_f1") for case in cases]),
        "exact_match_rate": _safe_mean([1.0 if case.get("exact_match") else 0.0 for case in ok_cases]),
        "field_coverage": _safe_mean([case.get("field_coverage") for case in ok_cases]),
        "latency_ms": _safe_mean([case.get("latency_ms") for case in cases]),
        "prompt_token_count": _safe_mean([case.get("prompt_token_count") for case in cases]),
        "format_error_count": sum(1 for case in cases if case.get("status") == "format_error"),
        "blocked_count": sum(1 for case in cases if case.get("status") == "blocked"),
        "timeout_count": sum(1 for case in cases if case.get("status") == "timeout"),
        "failure_count": sum(1 for case in cases if case.get("status") == "failed"),
        "status_counts": {
            status: sum(1 for case in cases if case.get("status") == status)
            for status in sorted({case.get("status", "ok") for case in cases})
        },
    }


def _scorecard_from_cases(label: str, summary: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    def group_by(field: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for case in cases:
            key = str(case.get(field) or "unknown")
            groups.setdefault(key, []).append(case)
        return {
            key: {
                "case_count": len(group_cases),
                "support_recall_at_k": _safe_mean([case.get("support_recall_at_k") for case in group_cases]),
                "write_success_rate": _safe_mean([case.get("write_success_rate") for case in group_cases]),
                "deferred_recall_at_k": _safe_mean([case.get("deferred_recall_at_k") for case in group_cases]),
                "prompt_token_count": _safe_mean([case.get("prompt_token_count") for case in group_cases]),
                "latency_ms": _safe_mean([case.get("latency_ms") for case in group_cases]),
                "failure_count": sum(1 for case in group_cases if case.get("status") not in {None, "ok"}),
            }
            for key, group_cases in sorted(groups.items())
        }

    strand_counts = summary.get("strand_case_counts") or {}
    return {
        "label": label,
        "mode": "memoryarena_experiment",
        "case_count": summary.get("case_count", len(cases)),
        "snapshot_lookup": (summary.get("family_case_counts") or {}).get("snapshot_lookup", 0),
        "learn_as_you_act": (summary.get("family_case_counts") or {}).get("learn_as_you_act", 0),
        "strand_counts": strand_counts,
        "support_recall_at_k": summary.get("support_recall_at_k"),
        "write_success_rate": summary.get("write_success_rate"),
        "deferred_recall_at_k": summary.get("deferred_recall_at_k"),
        "prompt_token_count": summary.get("prompt_token_count"),
        "latency_ms": summary.get("latency_ms"),
        "by_family": group_by("family"),
        "by_suite": group_by("suite"),
        "by_strand": group_by("strand"),
        "failed_cases_by_strand": _failed_cases_by_strand(cases),
    }


def _failed_cases_by_strand(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    failed: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        status_failed = case.get("status") not in {None, "ok"}
        recall = case.get("support_recall_at_k")
        recall_failed = recall is not None and recall < 1.0
        if not status_failed and not recall_failed:
            continue
        strand = str(case.get("strand") or "unknown")
        failed.setdefault(strand, []).append(
            {
                "case_id": case.get("case_id"),
                "suite": case.get("suite"),
                "family": case.get("family"),
                "turn_index": case.get("turn_index"),
                "status": case.get("status", "ok"),
                "support_recall_at_k": recall,
                "prompt_support_coverage": case.get("prompt_support_coverage"),
            }
        )
    return failed


def evaluate_memoryarena_experiment(
    engine: MemoryEngine,
    manifest: MemoryArenaDerivedManifest,
    *,
    label: str,
    family: str = "all",
    suites: list[str] | None = None,
    strands: list[str] | None = None,
    artifact_root: Path | None = None,
    prompt_limit: int = 4,
    strategy: str = "hybrid",
    trace_mode: str = "failures",
    jobs: int | str = "auto",
) -> dict[str, Any]:
    result = evaluate_memoryarena_offline(
        engine,
        manifest,
        family=family,
        suites=suites,
        strands=strands,
        artifact_root=artifact_root,
        prompt_limit=prompt_limit,
        strategy=strategy,
        trace_mode=trace_mode,
        jobs=jobs,
    )
    scorecard = _scorecard_from_cases(label, result["summary"], result["cases"])
    result["summary"] = {**result["summary"], "mode": "memoryarena_experiment", "label": label}
    result["scorecard"] = scorecard
    if result.get("artifact_dir"):
        artifact_dir = Path(result["artifact_dir"])
        (artifact_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _selected_counts(corpus: MemoryArenaCorpus) -> dict[str, Any]:
    task_counts: dict[str, int] = {}
    turn_counts: dict[str, int] = {}
    for task in corpus.tasks:
        task_counts[task.suite] = task_counts.get(task.suite, 0) + 1
        turn_counts[task.suite] = turn_counts.get(task.suite, 0) + len(task.turns)
    return {
        "task_count": len(corpus.tasks),
        "turn_count": sum(len(task.turns) for task in corpus.tasks),
        "suite_task_counts": task_counts,
        "suite_turn_counts": turn_counts,
    }


def _write_artifacts(
    *,
    artifact_root: Path,
    mode: str,
    summary: dict[str, Any],
    cases: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    output_dir = artifact_root / _timestamp_slug()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    cases_path = output_dir / "cases.jsonl"
    config_path = output_dir / "config.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True) + "\n")
    return output_dir


def _artifact_map(case: MemoryArenaDerivedCase) -> dict[str, dict[str, Any]]:
    return {
        artifact["artifact_id"]: artifact
        for artifact in [*case.static_artifacts, *case.dynamic_artifacts_before_turn]
    }


def _expected_support_for_case(case: MemoryArenaDerivedCase) -> list[dict[str, Any]]:
    artifacts = _artifact_map(case)
    return [artifacts[artifact_id] for artifact_id in case.expected_support_ids if artifact_id in artifacts]


def _dynamic_artifact_for_case(case: MemoryArenaDerivedCase) -> dict[str, Any]:
    if case.suite == "bundled_shopping":
        text = _shopping_answer_artifact_text(case.gold_answer_value)
        kind = "shopping_selection"
    elif case.suite == "group_travel_planner":
        text = f"Traveler plan:\n{serialise_answer(case.gold_answer_value)}"
        kind = "travel_plan"
    elif case.suite in {"formal_reasoning_math", "formal_reasoning_phys"}:
        paper_name = case.metadata.get("paper_name")
        prefix = f"{paper_name}: " if paper_name else ""
        text = f"{prefix}{case.gold_answer_text}".strip()
        kind = "reasoning_answer"
    else:
        text = case.gold_answer_text
        kind = "search_result"
    return _artifact_to_dict(
        MemoryArenaArtifact(
            artifact_id=f"{case.suite}:{case.task_id}:dynamic:turn_{case.turn_index}",
            kind=kind,
            text=text,
            value=case.gold_answer_value,
            metadata={"suite": case.suite, "task_id": case.task_id, "turn_index": case.turn_index},
        )
    )


def _dedupe_artifacts(artifacts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        deduped.setdefault(artifact["artifact_id"], artifact)
    return list(deduped.values())


def _case_corpus_artifacts(case: MemoryArenaDerivedCase) -> list[dict[str, Any]]:
    return _dedupe_artifacts([*case.static_artifacts, *case.dynamic_artifacts_before_turn, _dynamic_artifact_for_case(case)])


def _compact_trace(run_id: str, step_result: dict[str, Any]) -> dict[str, Any]:
    debug = list(step_result.get("debug") or [])
    return {
        "run_id": run_id,
        "mode": "summary",
        "steps": [
            {
                "step_index": step_result.get("step_index"),
                "activation_count": len(debug),
                "working_memory_count": len(step_result.get("working_memory") or []),
                "top_activation": debug[:SUPPORT_K],
            }
        ],
    }


def _should_capture_failure_trace(case_payload: dict[str, Any]) -> bool:
    if case_payload.get("status") != "ok":
        return True
    recall = case_payload.get("support_recall_at_k")
    if recall is not None and recall < 1.0:
        return True
    coverage = case_payload.get("prompt_support_coverage")
    return coverage is not None and coverage < 1.0


def _attach_trace(
    engine: MemoryEngine,
    *,
    run_id: str,
    step_result: dict[str, Any],
    trace_mode: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if trace_mode == "full" or (trace_mode == "failures" and _should_capture_failure_trace(payload)):
        payload["trace"] = engine.get_debug_trace(run_id)
    else:
        payload["trace"] = _compact_trace(run_id, step_result)
    return payload


def _dynamic_expected_support_for_case(case: MemoryArenaDerivedCase) -> list[dict[str, Any]]:
    dynamic_ids = {artifact["artifact_id"] for artifact in case.dynamic_artifacts_before_turn}
    artifacts = _artifact_map(case)
    return [
        artifacts[artifact_id]
        for artifact_id in case.expected_support_ids
        if artifact_id in dynamic_ids and artifact_id in artifacts
    ]


def _artifact_messages(case: MemoryArenaDerivedCase, artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    static_ids = {artifact["artifact_id"] for artifact in case.static_artifacts}
    for artifact in artifacts:
        is_static = artifact["artifact_id"] in static_ids
        messages.append(
            {
                "role": "system" if is_static else "assistant",
                "source_type": "system_note" if is_static else "agent_message",
                "content": artifact["text"],
                "metadata_json": {
                    "suite": case.suite,
                    "task_id": case.task_id,
                    "turn_index": case.turn_index,
                    "artifact_id": artifact["artifact_id"],
                    "artifact_kind": artifact["kind"],
                    "benchmark_family": case.family,
                },
            }
        )
    return messages


def _support_metrics(case: MemoryArenaDerivedCase, working_memory: list[dict[str, Any]], prompt_addition: str) -> dict[str, Any]:
    expected_support = _expected_support_for_case(case)
    dynamic_support = _dynamic_expected_support_for_case(case)
    top_items = working_memory[:SUPPORT_K]
    hit_count = sum(
        1
        for artifact in expected_support
        if any(_artifact_hit(artifact["text"].lower(), _content_to_text(item).lower()) for item in top_items)
    )
    prompt_hits = sum(
        1 for artifact in expected_support if _artifact_hit(artifact["text"].lower(), prompt_addition.lower())
    )
    dynamic_hits = sum(
        1
        for artifact in dynamic_support
        if any(_artifact_hit(artifact["text"].lower(), _content_to_text(item).lower()) for item in top_items)
    )
    return {
        "expected_support": expected_support,
        "support_expected_count": len(expected_support),
        "support_recall_at_5": (hit_count / len(expected_support) if expected_support else None),
        "support_precision_at_5": (hit_count / min(len(top_items), SUPPORT_K)) if top_items else None,
        "support_recall_at_k": (hit_count / len(expected_support) if expected_support else None),
        "support_precision_at_k": (hit_count / min(len(top_items), SUPPORT_K)) if top_items else None,
        "prompt_support_coverage": (prompt_hits / len(expected_support) if expected_support else None),
        "deferred_recall_at_5": (dynamic_hits / len(dynamic_support) if dynamic_support else None),
        "deferred_recall_at_k": (dynamic_hits / len(dynamic_support) if dynamic_support else None),
    }


def _write_success(
    engine: MemoryEngine,
    *,
    namespace_id: str,
    artifact: dict[str, Any] | None,
) -> float | None:
    if artifact is None:
        return None
    results = engine.search(artifact["text"], namespace_id=namespace_id, limit=SUPPORT_K)
    success = any(_artifact_hit(artifact["text"].lower(), _content_to_text(item).lower()) for item in results)
    return 1.0 if success else 0.0


def _offline_case_payload(
    case: MemoryArenaDerivedCase,
    *,
    working_memory: list[dict[str, Any]],
    prompt_addition: str,
    latency_ms: float,
    trace: dict[str, Any],
    write_success_rate: float | None = None,
) -> dict[str, Any]:
    metrics = _support_metrics(case, working_memory, prompt_addition)
    memory_context_packet = build_memory_context_packet(
        run_id=trace.get("run_id", "memoryarena") if isinstance(trace, dict) else "memoryarena",
        step_index=None,
        query=case.question,
        working_memory=working_memory,
        limit=len(working_memory),
        metadata={"case_id": case.case_id, "family": case.family, "suite": case.suite, "strand": case.strand},
    )
    return {
        "mode": "memoryarena_offline",
        "status": "ok",
        "case_id": case.case_id,
        "family": case.family,
        "suite": case.suite,
        "strand": case.strand,
        "task_id": case.task_id,
        "turn_index": case.turn_index,
        "question": case.question,
        "gold_answer_text": case.gold_answer_text,
        "expected_support_ids": case.expected_support_ids,
        "expected_support": metrics["expected_support"],
        "working_memory": working_memory,
        "memory_context_packet": memory_context_packet,
        "prompt_addition": prompt_addition,
        "support_expected_count": metrics["support_expected_count"],
        "support_recall_at_5": metrics["support_recall_at_5"],
        "support_precision_at_5": metrics["support_precision_at_5"],
        "support_recall_at_k": metrics["support_recall_at_k"],
        "support_precision_at_k": metrics["support_precision_at_k"],
        "prompt_support_coverage": metrics["prompt_support_coverage"],
        "deferred_recall_at_5": metrics["deferred_recall_at_5"],
        "deferred_recall_at_k": metrics["deferred_recall_at_k"],
        "write_success_rate": write_success_rate,
        "prompt_token_count": word_count(prompt_addition),
        "latency_ms": latency_ms,
        "trace": trace,
    }


def _agent_case_payload(
    case: MemoryArenaDerivedCase,
    *,
    reply_text: str,
    prompt_addition: str,
    latency_ms: float | None,
    run_id: str,
) -> dict[str, Any]:
    scored = _score_live_case(
        suite=case.suite,
        gold_answer_text=case.gold_answer_text,
        gold_answer_value=case.gold_answer_value,
        reply_text=reply_text,
    )
    return {
        "mode": "memoryarena_agent",
        "status": "ok",
        "case_id": case.case_id,
        "family": case.family,
        "suite": case.suite,
        "strand": case.strand,
        "task_id": case.task_id,
        "turn_index": case.turn_index,
        "question": case.question,
        "gold_answer_text": case.gold_answer_text,
        "reply_text": reply_text,
        "token_f1": scored["token_f1"],
        "exact_match": scored["exact_match"],
        "field_coverage": scored.get("field_coverage"),
        "prompt_addition": prompt_addition,
        "prompt_token_count": word_count(prompt_addition),
        "latency_ms": latency_ms,
        "run_id": run_id,
    }


def _set_workbench_replay_response(workbench, run_id: str, response_text: str) -> None:
    from memoria.db import session_scope
    from memoria.models import ExperimentRun

    with session_scope(workbench.engine.session_factory) as session:
        run = session.get(ExperimentRun, run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        provider_config = dict(run.config_json.get("provider") or {})
        provider_config["replay_responses"] = [response_text]
        run.config_json = {
            **run.config_json,
            "provider": provider_config,
        }
        session.add(run)


def evaluate_memoryarena_offline(
    engine: MemoryEngine,
    manifest: MemoryArenaDerivedManifest,
    *,
    family: str = "all",
    suites: list[str] | None = None,
    strands: list[str] | None = None,
    artifact_root: Path | None = None,
    prompt_limit: int = 4,
    strategy: str = "hybrid",
    trace_mode: str = "summary",
    jobs: int | str = 1,
) -> dict[str, Any]:
    if strategy not in OFFLINE_STRATEGIES:
        raise KeyError(f"Unknown offline strategy: {strategy}")
    if trace_mode not in OFFLINE_TRACE_MODES:
        raise KeyError(f"Unknown offline trace mode: {trace_mode}")
    worker_count = _normalise_offline_jobs(jobs)
    selected_cases = sorted(
        select_memoryarena_cases(manifest, family=family, suites=suites, strands=strands),
        key=lambda case: (case.family, case.suite, case.task_id, case.turn_index),
    )
    if worker_count > 1 and selected_cases:
        grouped_cases: dict[tuple[str, str], list[MemoryArenaDerivedCase]] = {}
        for case in selected_cases:
            grouped_cases.setdefault((case.suite, case.task_id), []).append(case)
        task_groups = list(grouped_cases.values())
        worker_count = min(worker_count, len(task_groups))
        chunks = [task_groups[index::worker_count] for index in range(worker_count)]

        def run_chunk(worker_index: int, chunk: list[list[MemoryArenaDerivedCase]], temp_root: Path) -> dict[str, Any]:
            worker_cases = [case for group in chunk for case in group]
            worker_config = engine.config.model_copy(deep=True)
            worker_config.database_url = f"sqlite:///{temp_root / f'worker-{worker_index}.db'}"
            worker_engine = MemoryEngine(config=worker_config)
            worker_manifest = MemoryArenaDerivedManifest(
                manifest_version=manifest.manifest_version,
                dataset_name=manifest.dataset_name,
                revision=manifest.revision,
                source=manifest.source,
                suites=manifest.suites,
                counts=manifest.counts,
                cases=worker_cases,
            )
            return evaluate_memoryarena_offline(
                worker_engine,
                worker_manifest,
                family="all",
                suites=None,
                strands=None,
                artifact_root=None,
                prompt_limit=prompt_limit,
                strategy=strategy,
                trace_mode=trace_mode,
                jobs=1,
            )

        with TemporaryDirectory(prefix="memoria-memoryarena-offline-") as temp_dir:
            temp_root = Path(temp_dir)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                worker_results = list(
                    executor.map(
                        lambda item: run_chunk(item[0], item[1], temp_root),
                        enumerate(chunks),
                    )
                )
        family_order = {"learn_as_you_act": 0, "snapshot_lookup": 1}
        cases = [
            case
            for result in worker_results
            for case in result["cases"]
        ]
        cases.sort(key=lambda case: (family_order.get(case.get("family"), 9), case["suite"], case["task_id"], case["turn_index"]))
        summary = _summary_from_cases("memoryarena_offline", cases)
        coverage_values = [
            case.get("corpus_learned_coverage")
            for case in cases
            if case.get("corpus_learned_coverage") is not None
        ]
        if coverage_values:
            summary["corpus_learned_coverage"] = _safe_mean(coverage_values)
            summary["corpus_backfilled_artifact_count"] = sum(
                int(result["summary"].get("corpus_backfilled_artifact_count") or 0)
                for result in worker_results
            )
        config = {
            "manifest_version": manifest.manifest_version,
            "dataset_name": manifest.dataset_name,
            "revision": manifest.revision,
            "source": manifest.source,
            "suites": manifest.suites,
            "available_counts": manifest.counts,
            "selected_family": family,
            "selected_suites": suites or [],
            "selected_strands": strands or [],
            "selected_case_count": len(selected_cases),
            "prompt_limit": prompt_limit,
            "strategy": strategy,
            "trace_mode": trace_mode,
            "jobs": jobs,
            "worker_count": worker_count,
        }
        artifact_dir = None
        if artifact_root is not None:
            artifact_dir = _write_artifacts(
                artifact_root=artifact_root,
                mode="memoryarena_offline",
                summary=summary,
                cases=cases,
                config=config,
            )
        return {
            "summary": summary,
            "cases": cases,
            "config": config,
            "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
        }
    cases: list[dict[str, Any]] = []

    grouped_learn_cases: dict[tuple[str, str], list[MemoryArenaDerivedCase]] = {}
    for case in selected_cases:
        if case.family == "learn_as_you_act":
            grouped_learn_cases.setdefault((case.suite, case.task_id), []).append(case)

    task_states: dict[tuple[str, str], dict[str, Any]] = {}
    hybrid_backfilled_artifact_count = 0

    def evaluate_case(
        case: MemoryArenaDerivedCase,
        *,
        namespace_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        run_ctx = engine.start_run(
            namespace_id=namespace_id,
            user_id="memoryarena",
            agent_id="offline",
            session_id=session_id,
        )
        started_at = perf_counter()
        step_result = engine.process_step_result(
            run_ctx["run_id"],
            step_type="user_input",
            text=case.question,
            metadata={"case_id": case.case_id, "family": case.family, "suite": case.suite},
        )
        latency_ms = (perf_counter() - started_at) * 1000.0
        working_memory = step_result["working_memory"]
        prompt_addition = _prompt_addition(working_memory, limit=prompt_limit)
        payload = _offline_case_payload(
            case,
            working_memory=working_memory,
            prompt_addition=prompt_addition,
            latency_ms=latency_ms,
            trace={},
        )
        return _attach_trace(
            engine,
            run_id=run_ctx["run_id"],
            step_result=step_result,
            trace_mode=trace_mode,
            payload=payload,
        )

    def load_missing_artifacts(
        case: MemoryArenaDerivedCase,
        *,
        namespace_id: str,
        session_id: str,
        loaded_artifact_ids: set[str],
        artifacts: list[dict[str, Any]],
    ) -> int:
        missing = [artifact for artifact in _dedupe_artifacts(artifacts) if artifact["artifact_id"] not in loaded_artifact_ids]
        if not missing:
            return 0
        engine.add_messages(
            _artifact_messages(case, missing),
            namespace_id=namespace_id,
            user_id="memoryarena",
            agent_id="offline",
            session_id=session_id,
        )
        engine.consolidate(namespace_id=namespace_id)
        loaded_artifact_ids.update(artifact["artifact_id"] for artifact in missing)
        return len(missing)

    for (suite, task_id), task_cases in grouped_learn_cases.items():
        ordered_cases = sorted(task_cases, key=lambda item: item.turn_index)
        namespace_id = f"memoryarena.offline.learn.{suite}.{task_id}"
        session_id = f"learn:{suite}:{task_id}"
        loaded_artifact_ids: set[str] = set()
        for case in ordered_cases:
            load_missing_artifacts(
                case,
                namespace_id=namespace_id,
                session_id=session_id,
                loaded_artifact_ids=loaded_artifact_ids,
                artifacts=[*case.static_artifacts, *case.dynamic_artifacts_before_turn],
            )
            current_artifact_dict = _dynamic_artifact_for_case(case)
            payload = evaluate_case(
                case,
                namespace_id=namespace_id,
                session_id=session_id,
            )
            engine.add_messages(
                _artifact_messages(case, [current_artifact_dict]),
                namespace_id=namespace_id,
                user_id="memoryarena",
                agent_id="offline",
                session_id=session_id,
            )
            engine.consolidate(namespace_id=namespace_id)
            loaded_artifact_ids.add(current_artifact_dict["artifact_id"])
            payload["write_success_rate"] = _write_success(
                engine,
                namespace_id=namespace_id,
                artifact=current_artifact_dict,
            )
            cases.append(payload)
        task_states[(suite, task_id)] = {
            "namespace_id": namespace_id,
            "session_id": session_id,
            "loaded_artifact_ids": loaded_artifact_ids,
            "learned_artifact_ids": set(loaded_artifact_ids),
        }

    snapshot_cases = [case for case in selected_cases if case.family == "snapshot_lookup"]
    if strategy == "legacy":
        for case in snapshot_cases:
            namespace_id = f"memoryarena.offline.snapshot.{case.suite}.{case.task_id}.turn_{case.turn_index}"
            session_id = case.case_id
            artifacts_to_load = [*case.static_artifacts, *case.dynamic_artifacts_before_turn]
            if artifacts_to_load:
                engine.add_messages(
                    _artifact_messages(case, artifacts_to_load),
                    namespace_id=namespace_id,
                    user_id="memoryarena",
                    agent_id="offline",
                    session_id=session_id,
                )
                engine.consolidate(namespace_id=namespace_id)
            cases.append(evaluate_case(case, namespace_id=namespace_id, session_id=session_id))
    else:
        grouped_snapshot_cases: dict[tuple[str, str], list[MemoryArenaDerivedCase]] = {}
        for case in snapshot_cases:
            grouped_snapshot_cases.setdefault((case.suite, case.task_id), []).append(case)
        all_task_cases: dict[tuple[str, str], list[MemoryArenaDerivedCase]] = {}
        for case in selected_cases:
            all_task_cases.setdefault((case.suite, case.task_id), []).append(case)
        for (suite, task_id), task_cases in grouped_snapshot_cases.items():
            ordered_cases = sorted(task_cases, key=lambda item: item.turn_index)
            state = task_states.get((suite, task_id))
            if state is None:
                state = {
                    "namespace_id": f"memoryarena.offline.hybrid.{suite}.{task_id}",
                    "session_id": f"hybrid:{suite}:{task_id}",
                    "loaded_artifact_ids": set(),
                    "learned_artifact_ids": set(),
                }
                task_states[(suite, task_id)] = state
            expected_artifacts = _dedupe_artifacts(
                artifact
                for task_case in all_task_cases.get((suite, task_id), ordered_cases)
                for artifact in _case_corpus_artifacts(task_case)
            )
            learned_ids = set(state["learned_artifact_ids"])
            expected_ids = {artifact["artifact_id"] for artifact in expected_artifacts}
            backfilled_count = load_missing_artifacts(
                ordered_cases[-1],
                namespace_id=state["namespace_id"],
                session_id=state["session_id"],
                loaded_artifact_ids=state["loaded_artifact_ids"],
                artifacts=expected_artifacts,
            )
            hybrid_backfilled_artifact_count += backfilled_count
            coverage_rate = len(learned_ids & expected_ids) / len(expected_ids) if expected_ids else None
            for case in ordered_cases:
                payload = evaluate_case(
                    case,
                    namespace_id=state["namespace_id"],
                    session_id=case.case_id,
                )
                payload["corpus_expected_artifact_count"] = len(expected_ids)
                payload["corpus_learned_artifact_count"] = len(learned_ids & expected_ids)
                payload["corpus_learned_coverage"] = coverage_rate
                payload["corpus_backfilled_artifact_count"] = backfilled_count
                cases.append(payload)

    summary = _summary_from_cases("memoryarena_offline", cases)
    coverage_values = [
        case.get("corpus_learned_coverage")
        for case in cases
        if case.get("corpus_learned_coverage") is not None
    ]
    if coverage_values:
        summary["corpus_learned_coverage"] = _safe_mean(coverage_values)
        summary["corpus_backfilled_artifact_count"] = hybrid_backfilled_artifact_count
    config = {
        "manifest_version": manifest.manifest_version,
        "dataset_name": manifest.dataset_name,
        "revision": manifest.revision,
        "source": manifest.source,
        "suites": manifest.suites,
        "available_counts": manifest.counts,
        "selected_family": family,
        "selected_suites": suites or [],
        "selected_strands": strands or [],
        "selected_case_count": len(selected_cases),
        "prompt_limit": prompt_limit,
        "strategy": strategy,
        "trace_mode": trace_mode,
        "jobs": jobs,
    }
    artifact_dir = None
    if artifact_root is not None:
        artifact_dir = _write_artifacts(
            artifact_root=artifact_root,
            mode="memoryarena_offline",
            summary=summary,
            cases=cases,
            config=config,
        )
    return {
        "summary": summary,
        "cases": cases,
        "config": config,
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
    }


def evaluate_memoryarena_proxy(
    engine: MemoryEngine,
    corpus: MemoryArenaCorpus,
    *,
    artifact_root: Path | None = None,
    prompt_limit: int = 4,
    family: str = "all",
    strands: list[str] | None = None,
) -> dict[str, Any]:
    manifest = build_memoryarena_derived_manifest(corpus)
    return evaluate_memoryarena_offline(
        engine,
        manifest,
        family=family,
        suites=corpus.suites,
        strands=strands,
        artifact_root=artifact_root,
        prompt_limit=prompt_limit,
    )


def evaluate_memoryarena_agent(
    workbench,
    manifest: MemoryArenaDerivedManifest,
    *,
    provider_config: Any,
    family: str = "all",
    suites: list[str] | None = None,
    strands: list[str] | None = None,
    artifact_root: Path | None = None,
    prompt_limit: int = 4,
) -> dict[str, Any]:
    from memoria.workbench import RunSpec

    selected_cases = sorted(
        select_memoryarena_cases(manifest, family=family, suites=suites, strands=strands),
        key=lambda case: (case.family, case.suite, case.task_id, case.turn_index),
    )
    cases: list[dict[str, Any]] = []

    snapshot_cases = [case for case in selected_cases if case.family == "snapshot_lookup"]
    for case in snapshot_cases:
        run_provider = provider_config
        if provider_config.provider_type == "replay":
            run_provider = type(provider_config)(
                **{
                    **provider_config.__dict__,
                    "replay_responses": [case.gold_answer_text],
                }
            )
        run = workbench.create_run(
            RunSpec(
                title=f"MemoryArena {case.case_id}",
                provider=run_provider,
                namespace_id=f"memoryarena.agent.snapshot.{case.suite}.{case.task_id}.turn_{case.turn_index}",
                user_id="memoryarena",
                agent_id="agent-eval",
                session_id=case.case_id,
                system_prompt="Answer the benchmark question directly.",
                prompt_limit=prompt_limit,
                metadata={"case_id": case.case_id, "family": case.family},
            )
        )
        artifacts_to_load = [*case.static_artifacts, *case.dynamic_artifacts_before_turn]
        if artifacts_to_load:
            workbench.engine.add_messages(
                _artifact_messages(case, artifacts_to_load),
                namespace_id=run["namespace_id"],
                user_id=run["user_id"],
                agent_id=run["agent_id"],
                session_id=run["session_id"],
            )
            workbench.engine.consolidate(namespace_id=run["namespace_id"])
        if provider_config.provider_type == "replay":
            _set_workbench_replay_response(workbench, run["id"], case.gold_answer_text)
        turn = workbench.send_user_message(run["id"], case.question)
        cases.append(
            _agent_case_payload(
                case,
                reply_text=turn["turn"]["assistant_message"],
                prompt_addition=turn["turn"]["prompt_addition"],
                latency_ms=turn["turn"]["latency_ms"],
                run_id=run["id"],
            )
        )

    grouped_learn_cases: dict[tuple[str, str], list[MemoryArenaDerivedCase]] = {}
    for case in selected_cases:
        if case.family == "learn_as_you_act":
            grouped_learn_cases.setdefault((case.suite, case.task_id), []).append(case)
    for (suite, task_id), task_cases in grouped_learn_cases.items():
        ordered_cases = sorted(task_cases, key=lambda item: item.turn_index)
        run_provider = provider_config
        if provider_config.provider_type == "replay":
            run_provider = type(provider_config)(
                **{
                    **provider_config.__dict__,
                    "replay_responses": [case.gold_answer_text for case in ordered_cases],
                }
            )
        first_case = ordered_cases[0]
        run = workbench.create_run(
            RunSpec(
                title=f"MemoryArena learn {suite}:{task_id}",
                provider=run_provider,
                namespace_id=f"memoryarena.agent.learn.{suite}.{task_id}",
                user_id="memoryarena",
                agent_id="agent-eval",
                session_id=f"learn:{suite}:{task_id}",
                system_prompt="Answer the benchmark question directly.",
                prompt_limit=prompt_limit,
                metadata={"suite": suite, "task_id": task_id, "family": "learn_as_you_act"},
            )
        )
        loaded_artifact_ids: set[str] = set()
        if first_case.static_artifacts:
            workbench.engine.add_messages(
                _artifact_messages(first_case, first_case.static_artifacts),
                namespace_id=run["namespace_id"],
                user_id=run["user_id"],
                agent_id=run["agent_id"],
                session_id=run["session_id"],
            )
            workbench.engine.consolidate(namespace_id=run["namespace_id"])
            loaded_artifact_ids.update(artifact["artifact_id"] for artifact in first_case.static_artifacts)
        for case in ordered_cases:
            missing_dynamic = [
                artifact
                for artifact in case.dynamic_artifacts_before_turn
                if artifact["artifact_id"] not in loaded_artifact_ids
            ]
            if missing_dynamic:
                workbench.engine.add_messages(
                    _artifact_messages(case, missing_dynamic),
                    namespace_id=run["namespace_id"],
                    user_id=run["user_id"],
                    agent_id=run["agent_id"],
                    session_id=run["session_id"],
                )
                workbench.engine.consolidate(namespace_id=run["namespace_id"])
                loaded_artifact_ids.update(artifact["artifact_id"] for artifact in missing_dynamic)
            if provider_config.provider_type == "replay":
                _set_workbench_replay_response(workbench, run["id"], case.gold_answer_text)
            turn = workbench.send_user_message(run["id"], case.question)
            loaded_artifact_ids.add(f"{case.suite}:{case.task_id}:dynamic:turn_{case.turn_index}")
            cases.append(
                _agent_case_payload(
                    case,
                    reply_text=turn["turn"]["assistant_message"],
                    prompt_addition=turn["turn"]["prompt_addition"],
                    latency_ms=turn["turn"]["latency_ms"],
                    run_id=run["id"],
                )
            )

    summary = _summary_from_cases("memoryarena_agent", cases)
    config = {
        "manifest_version": manifest.manifest_version,
        "dataset_name": manifest.dataset_name,
        "revision": manifest.revision,
        "source": manifest.source,
        "suites": manifest.suites,
        "available_counts": manifest.counts,
        "selected_family": family,
        "selected_suites": suites or [],
        "selected_strands": strands or [],
        "selected_case_count": len(selected_cases),
        "provider": provider_config.__dict__,
        "prompt_limit": prompt_limit,
    }
    artifact_dir = None
    if artifact_root is not None:
        artifact_dir = _write_artifacts(
            artifact_root=artifact_root,
            mode="memoryarena_agent",
            summary=summary,
            cases=cases,
            config=config,
        )
    return {
        "summary": summary,
        "cases": cases,
        "config": config,
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
    }


def token_f1_score(reference: str, candidate: str) -> float:
    reference_tokens = list(_normalise_tokens(reference))
    candidate_tokens = list(_normalise_tokens(candidate))
    if not reference_tokens or not candidate_tokens:
        return 0.0
    reference_set = set(reference_tokens)
    candidate_set = set(candidate_tokens)
    overlap = len(reference_set & candidate_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(candidate_set)
    recall = overlap / len(reference_set)
    return 2 * precision * recall / (precision + recall)


def _travel_field_values(answer_value: Any) -> list[str]:
    if not isinstance(answer_value, list):
        return []
    values: list[str] = []
    for item in answer_value:
        if not isinstance(item, dict):
            continue
        for key in ("days", "current_city", "transportation"):
            value = item.get(key)
            if value is not None:
                values.append(str(value))
    return values


def _score_live_case(
    *,
    suite: str,
    gold_answer_text: str,
    gold_answer_value: Any,
    reply_text: str,
) -> dict[str, Any]:
    gold_text = gold_answer_text
    metrics = {
        "token_f1": token_f1_score(gold_text, reply_text),
        "exact_match": gold_text.strip().lower() == reply_text.strip().lower(),
    }
    if suite == "group_travel_planner":
        field_values = _travel_field_values(gold_answer_value)
        hits = sum(1 for value in field_values if value.strip() and value.strip().lower() in reply_text.lower())
        metrics["field_coverage"] = hits / len(field_values) if field_values else None
    return metrics


def _json_request(base_url: str, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Memoria sidecar request failed: {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Memoria sidecar at {base_url}: {exc.reason}") from exc


def _session_trace_path(session: dict[str, Any]) -> str:
    parts = [session["namespace_id"], session.get("user_id") or "-", session.get("agent_id") or "-", session.get("session_id") or "-"]
    return f"/sessions/{quote('::'.join(str(part) for part in parts))}/trace"


def _extract_openclaw_reply(payload: Any) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        payloads = payload.get("payloads")
        if isinstance(payloads, list):
            parts = [
                str(item.get("text")).strip()
                for item in payloads
                if isinstance(item, dict) and item.get("text")
            ]
            if parts:
                return "\n".join(parts)
        for key in ("reply", "message", "output", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        content = payload.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(item.get("text")).strip()
                for item in content
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
            ]
            if parts:
                return "\n".join(parts)
        messages = payload.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "assistant":
                    text = _extract_openclaw_reply(message)
                    if text:
                        return text
        response = payload.get("response")
        if response is not None:
            text = _extract_openclaw_reply(response)
            if text:
                return text
        result = payload.get("result")
        if result is not None:
            text = _extract_openclaw_reply(result)
            if text:
                return text
    return ""


def _classify_openclaw_response(payload: Any) -> tuple[str, str]:
    reply_text = _extract_openclaw_reply(payload).strip()
    if not reply_text:
        return "", "empty_payload"
    if reply_text in {"{}", "[]", "null"}:
        return "", "empty_payload"
    return reply_text, "text"


def _trace_step_type(step: dict[str, Any]) -> str | None:
    step_type = step.get("step_type")
    if isinstance(step_type, str) and step_type:
        return step_type
    activation = step.get("activation")
    if not isinstance(activation, list):
        return None
    for item in activation:
        if not isinstance(item, dict):
            continue
        reason = item.get("reason")
        if isinstance(reason, dict):
            candidate = reason.get("step_type")
            if isinstance(candidate, str) and candidate:
                return candidate
    return None


def _parse_json_output(stdout: str, stderr: str = "") -> Any:
    def try_parse_stream(stream: str) -> Any | None:
        stream = stream.strip()
        if not stream:
            return None
        try:
            return json.loads(stream)
        except json.JSONDecodeError:
            pass
        start = stream.find("{")
        end = stream.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = stream[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        for line in reversed(stream.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return None

    stdout = stdout.strip()
    stderr = stderr.strip()
    if not stdout and not stderr:
        return {}
    for stream in (stdout, stderr):
        parsed = try_parse_stream(stream)
        if parsed is not None:
            return parsed
    raise RuntimeError("Could not parse JSON output from openclaw agent --json")


def _active_openclaw_config_path() -> Path:
    override = os.environ.get("OPENCLAW_CONFIG_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".openclaw" / "openclaw.json"


def _active_openclaw_state_dir() -> Path:
    override = os.environ.get("OPENCLAW_STATE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return _active_openclaw_config_path().expanduser().resolve().parent


def _load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_benchmark_workspace(workspace_dir: Path) -> None:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    for name, content in BENCHMARK_WORKSPACE_FILES.items():
        (workspace_dir / name).write_text(content, encoding="utf-8")
    (workspace_dir / "memory").mkdir(exist_ok=True)


def _copy_openclaw_auth_state(source_state_dir: Path, target_state_dir: Path, agent_id: str) -> None:
    source_agent_dir = source_state_dir / "agents" / agent_id / "agent"
    target_agent_dir = target_state_dir / "agents" / agent_id / "agent"
    target_agent_dir.mkdir(parents=True, exist_ok=True)
    for name in ("auth-profiles.json", "auth-state.json", "models.json"):
        source = source_agent_dir / name
        if source.exists():
            shutil.copy2(source, target_agent_dir / name)


def _build_isolated_openclaw_config(
    *,
    base_config: dict[str, Any],
    memory_mode: str,
    workspace_dir: Path,
    plugin_path: Path,
    sidecar_url: str,
    namespace_id: str,
    prompt_limit: int,
    debug_commands: bool,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    agents = dict(config.get("agents") or {})
    defaults = dict(agents.get("defaults") or {})
    defaults["workspace"] = str(workspace_dir)
    defaults["skipBootstrap"] = True
    agents["defaults"] = defaults
    config["agents"] = agents

    commands = dict(config.get("commands") or {})
    if debug_commands:
        commands["debug"] = True
    if commands:
        config["commands"] = commands

    plugins = dict(config.get("plugins") or {})
    load = dict(plugins.get("load") or {})
    load_paths = [path for path in list(load.get("paths") or []) if Path(path) != plugin_path]
    entries = dict(plugins.get("entries") or {})
    slots = dict(plugins.get("slots") or {})
    installs = dict(plugins.get("installs") or {})

    if memory_mode == "native":
        load_paths.append(str(plugin_path))
        slots["contextEngine"] = "memoria"
        entries["memoria-openclaw"] = {
            "enabled": True,
            "config": {
                "baseUrl": sidecar_url,
                "namespaceId": namespace_id,
                "promptLimit": prompt_limit,
                "autoStoreAssistantTurns": False,
            },
        }
    else:
        slots.pop("contextEngine", None)
        entries.pop("memoria-openclaw", None)
        installs.pop("memoria-openclaw", None)

    load["paths"] = load_paths
    plugins["load"] = load
    if entries:
        plugins["entries"] = entries
    else:
        plugins.pop("entries", None)
    if slots:
        plugins["slots"] = slots
    else:
        plugins.pop("slots", None)
    if installs:
        plugins["installs"] = installs
    else:
        plugins.pop("installs", None)
    config["plugins"] = plugins
    return config


def _build_benchmark_message(
    turn: MemoryArenaTurn,
    *,
    include_background: bool,
    prompt_addition: str = "",
) -> str:
    sections: list[str] = []
    if include_background and turn.background_items:
        sections.append("Session context:\n" + "\n\n".join(item.strip() for item in turn.background_items if item.strip()))
    if prompt_addition.strip():
        sections.append(prompt_addition.strip())
    sections.append(f"User request:\n{turn.question.strip()}")
    return "\n\n".join(section for section in sections if section)


def _resolve_openclaw_version(openclaw_command: str) -> str:
    completed = run([openclaw_command, "--version"], check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _delta_metric(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    left_value = left.get(key)
    right_value = right.get(key)
    if left_value is None or right_value is None:
        return None
    return float(left_value) - float(right_value)


def _build_compare_summary(
    *,
    corpus: MemoryArenaCorpus,
    mode_results: dict[str, dict[str, Any]],
    openclaw_version: str,
    openclaw_command: str,
    memory_modes: list[str],
) -> dict[str, Any]:
    summaries = {
        mode: {
            "summary": result["summary"],
            "artifact_dir": result["artifact_dir"],
            "config": result["config"],
        }
        for mode, result in mode_results.items()
    }
    baseline = mode_results.get("baseline", {}).get("summary", {})
    native = mode_results.get("native", {}).get("summary", {})
    prefetch = mode_results.get("prefetch", {}).get("summary", {})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": corpus.dataset_name,
        "revision": corpus.revision,
        "source": corpus.source,
        "suites": corpus.suites,
        "available_counts": corpus.counts,
        "selected_counts": _selected_counts(corpus),
        "memory_modes": memory_modes,
        "openclaw_command": openclaw_command,
        "openclaw_version": openclaw_version,
        "modes": summaries,
        "comparisons": {
            "native_vs_baseline": {
                "token_f1_delta": _delta_metric(native, baseline, "token_f1"),
                "exact_match_rate_delta": _delta_metric(native, baseline, "exact_match_rate"),
                "field_coverage_delta": _delta_metric(native, baseline, "field_coverage"),
                "format_error_delta": _delta_metric(native, baseline, "format_error_count"),
            },
            "prefetch_vs_baseline": {
                "token_f1_delta": _delta_metric(prefetch, baseline, "token_f1"),
                "exact_match_rate_delta": _delta_metric(prefetch, baseline, "exact_match_rate"),
                "field_coverage_delta": _delta_metric(prefetch, baseline, "field_coverage"),
                "format_error_delta": _delta_metric(prefetch, baseline, "format_error_count"),
            },
        },
        "notes": [
            "Results are model-specific and not comparable to the official MemoryArena leaderboard.",
            "Native runs currently depend on the locally patched OpenClaw install on this machine.",
        ],
    }


def write_public_memoryarena_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def run_memoryarena_openclaw(
    corpus: MemoryArenaCorpus,
    *,
    sidecar_url: str,
    namespace_id: str = "openclaw.default",
    openclaw_command: str = "openclaw",
    openclaw_agent_id: str = "main",
    openclaw_profile: str | None = None,
    timeout_seconds: int = 90,
    local: bool = False,
    artifact_root: Path | None = None,
    memory_mode: str = "native",
    isolated_openclaw: bool = True,
    base_config_path: Path | None = None,
    prompt_limit: int = 4,
    debug_commands: bool = False,
    raw_stream: bool = False,
    raw_stream_path: Path | None = None,
) -> dict[str, Any]:
    if memory_mode not in LIVE_MEMORY_MODES:
        raise KeyError(f"Unknown MemoryArena memory mode: {memory_mode}")
    cases: list[dict[str, Any]] = []
    openclaw_version = _resolve_openclaw_version(openclaw_command)
    base_config_path = (base_config_path or _active_openclaw_config_path()).expanduser()
    base_state_dir = _active_openclaw_state_dir().expanduser()
    base_config = _load_json_file(base_config_path)
    plugin_path = Path(__file__).resolve().parent.parent / "integrations" / "openclaw"
    default_command_env = os.environ.copy()
    if raw_stream:
        default_command_env["OPENCLAW_RAW_STREAM"] = "1"
        if raw_stream_path is not None:
            default_command_env["OPENCLAW_RAW_STREAM_PATH"] = str(raw_stream_path)
    for task in corpus.tasks:
        task_temp_dir_handle = None
        try:
            if isolated_openclaw:
                task_temp_dir_handle = TemporaryDirectory(prefix="memoria-openclaw-benchmark-")
                sandbox_root = Path(task_temp_dir_handle.name)
                state_dir = sandbox_root / "state"
                config_path = sandbox_root / "openclaw.json"
                workspace_dir = sandbox_root / "workspace"
                _write_benchmark_workspace(workspace_dir)
                _copy_openclaw_auth_state(base_state_dir, state_dir, openclaw_agent_id)
                benchmark_config = _build_isolated_openclaw_config(
                    base_config=base_config,
                    memory_mode=memory_mode,
                    workspace_dir=workspace_dir,
                    plugin_path=plugin_path,
                    sidecar_url=sidecar_url,
                    namespace_id=namespace_id,
                    prompt_limit=prompt_limit,
                    debug_commands=debug_commands,
                )
                _write_json_file(config_path, benchmark_config)
                command_env = default_command_env.copy()
                command_env["OPENCLAW_CONFIG_PATH"] = str(config_path)
                command_env["OPENCLAW_STATE_DIR"] = str(state_dir)
            else:
                command_env = default_command_env.copy()
                config_path = base_config_path
                state_dir = base_state_dir
            session_id = f"memoryarena-{task.suite}-{task.task_id}"
            session = {
                "namespace_id": namespace_id,
                "user_id": None,
                "agent_id": openclaw_agent_id,
                "session_id": session_id,
            }
            if task.suite not in LIVE_SUPPORTED_SUITES:
                for turn in task.turns:
                    cases.append(
                        {
                            "mode": "memoryarena_openclaw",
                            "status": "env_unavailable",
                            "suite": task.suite,
                            "task_id": task.task_id,
                            "turn_index": turn.turn_index,
                            "question": turn.question,
                            "gold_answer_text": turn.gold_answer_text,
                            "memory_mode": memory_mode,
                        }
                    )
                continue
            if memory_mode != "baseline":
                _json_request(sidecar_url, "POST", "/sessions/bootstrap", session)
                if task.background_items:
                    _json_request(
                        sidecar_url,
                        "POST",
                        "/sessions/ingest",
                        {
                            **session,
                            "messages": [
                                {
                                    "role": "system",
                                    "source_type": "system_note",
                                    "content": item,
                                    "metadata": {"suite": task.suite, "task_id": task.task_id, "kind": "background"},
                                }
                                for item in task.background_items
                            ],
                        },
                    )
            task_blocked_error: str | None = None
            for turn in task.turns:
                if task_blocked_error is not None:
                    cases.append(
                        {
                            "mode": "memoryarena_openclaw",
                            "status": "blocked",
                            "suite": task.suite,
                            "task_id": task.task_id,
                            "turn_index": turn.turn_index,
                            "question": turn.question,
                            "gold_answer_text": turn.gold_answer_text,
                            "reply_text": "",
                            "reply_format": "blocked",
                            "latency_ms": None,
                            "memory_mode": memory_mode,
                            "prefetched_prompt_addition": "",
                            "trace_step_types": [],
                            "trace": None,
                            "openclaw_payload": None,
                            "error": task_blocked_error,
                        }
                    )
                    continue
                recall_result = None
                if memory_mode == "prefetch":
                    recall_result = _json_request(
                        sidecar_url,
                        "POST",
                        "/sessions/recall",
                        {
                            **session,
                            "text": turn.question,
                            "step_type": "user_input",
                            "metadata": {"suite": task.suite, "task_id": task.task_id, "turn_index": turn.turn_index},
                            "prompt_limit": prompt_limit,
                        },
                    )
                benchmark_message = _build_benchmark_message(
                    turn,
                    include_background=turn.turn_index == 0,
                    prompt_addition=recall_result.get("prompt_addition", "") if recall_result else "",
                )
                command = [openclaw_command]
                if openclaw_profile:
                    command.extend(["--profile", openclaw_profile])
                command.extend(["agent", "--json", "--session-id", session_id, "--agent", openclaw_agent_id, "-m", benchmark_message])
                if local:
                    command.append("--local")
                started_at = perf_counter()
                try:
                    completed = run(
                        command,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=timeout_seconds,
                        env=command_env,
                    )
                    try:
                        payload = _parse_json_output(completed.stdout, completed.stderr)
                        reply_text, reply_format = _classify_openclaw_response(payload)
                        if reply_format == "text":
                            status = "ok"
                            error = None
                        else:
                            status = "format_error"
                            error = "OpenClaw returned no usable assistant text"
                    except RuntimeError as exc:
                        payload = {"stdout": completed.stdout, "stderr": completed.stderr}
                        reply_text = ""
                        reply_format = "parse_error"
                        status = "failed"
                        error = str(exc)
                except TimeoutExpired as exc:
                    payload = {"stdout": exc.stdout or "", "stderr": exc.stderr or ""}
                    reply_text = ""
                    reply_format = "timeout"
                    status = "timeout"
                    error = f"OpenClaw timed out after {timeout_seconds} seconds"
                except CalledProcessError as exc:
                    payload = {"stdout": exc.stdout or "", "stderr": exc.stderr or ""}
                    reply_text = ""
                    reply_format = "process_error"
                    status = "failed"
                    error = exc.stderr.strip() or exc.stdout.strip() or "OpenClaw agent invocation failed"
                latency_ms = (perf_counter() - started_at) * 1000.0
                metrics = (
                    _score_live_case(
                        suite=turn.suite,
                        gold_answer_text=turn.gold_answer_text,
                        gold_answer_value=turn.gold_answer_value,
                        reply_text=reply_text,
                    )
                    if status == "ok"
                    else {}
                )
                trace = None
                trace_steps: list[dict[str, Any]] = []
                if memory_mode == "prefetch" and status == "ok":
                    _json_request(
                        sidecar_url,
                        "POST",
                        "/sessions/after-turn",
                        {
                            **session,
                            "messages": [
                                {
                                    "role": "user",
                                    "source_type": "user_message",
                                    "content": turn.question,
                                    "metadata": {"suite": task.suite, "task_id": task.task_id, "turn_index": turn.turn_index},
                                },
                                {
                                    "role": "assistant",
                                    "source_type": "agent_message",
                                    "content": reply_text,
                                    "metadata": {"suite": task.suite, "task_id": task.task_id, "turn_index": turn.turn_index},
                                },
                            ],
                        },
                    )
                if memory_mode != "baseline":
                    trace = _json_request(
                        sidecar_url,
                        "GET",
                        _session_trace_path(session),
                    )
                    trace_steps = list((trace.get("trace") or {}).get("steps") or [])
                if status == "timeout":
                    task_blocked_error = (
                        f"Previous turn timed out in session {session_id}; skipped remaining turns to avoid stale locks."
                    )
                elif status == "failed" and error and "session file locked" in error.lower():
                    task_blocked_error = (
                        f"Previous turn left a locked OpenClaw session in {session_id}; skipped remaining turns."
                    )
                cases.append(
                    {
                        "mode": "memoryarena_openclaw",
                        "status": status,
                        "suite": task.suite,
                        "task_id": task.task_id,
                        "turn_index": turn.turn_index,
                        "question": turn.question,
                        "gold_answer_text": turn.gold_answer_text,
                        "reply_text": reply_text,
                        "reply_format": reply_format,
                        "latency_ms": latency_ms,
                        "memory_mode": memory_mode,
                        "prefetched_prompt_addition": recall_result.get("prompt_addition", "") if recall_result else "",
                        "trace_step_types": [_trace_step_type(step) for step in trace_steps],
                        "trace": trace,
                        "openclaw_payload": payload,
                        "error": error,
                        **metrics,
                    }
                )
        finally:
            if task_temp_dir_handle is not None:
                task_temp_dir_handle.cleanup()
    summary = _summary_from_cases("memoryarena_openclaw", cases)
    config = {
        "dataset_name": corpus.dataset_name,
        "revision": corpus.revision,
        "source": corpus.source,
        "suites": corpus.suites,
        "available_counts": corpus.counts,
        "selected_counts": _selected_counts(corpus),
        "sidecar_url": sidecar_url,
        "namespace_id": namespace_id,
        "openclaw_command": openclaw_command,
        "openclaw_version": openclaw_version,
        "openclaw_agent_id": openclaw_agent_id,
        "openclaw_profile": openclaw_profile,
        "local": local,
        "timeout_seconds": timeout_seconds,
        "memory_mode": memory_mode,
        "isolated_openclaw": isolated_openclaw,
        "base_config_path": str(base_config_path),
        "debug_commands": debug_commands,
        "raw_stream": raw_stream,
        "raw_stream_path": str(raw_stream_path) if raw_stream_path is not None else None,
    }
    artifact_dir = None
    if artifact_root is not None:
        artifact_dir = _write_artifacts(
            artifact_root=artifact_root,
            mode="memoryarena_openclaw",
            summary=summary,
            cases=cases,
            config=config,
        )
    return {
        "summary": summary,
        "cases": cases,
        "config": config,
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
    }


def compare_memoryarena_openclaw(
    corpus: MemoryArenaCorpus,
    *,
    sidecar_url: str,
    namespace_id: str = "openclaw.default",
    openclaw_command: str = "openclaw",
    openclaw_agent_id: str = "main",
    openclaw_profile: str | None = None,
    timeout_seconds: int = 90,
    local: bool = False,
    artifact_root: Path | None = None,
    memory_modes: list[str] | None = None,
    isolated_openclaw: bool = True,
    base_config_path: Path | None = None,
    prompt_limit: int = 4,
    debug_commands: bool = False,
    raw_stream: bool = False,
    raw_stream_path: Path | None = None,
    public_summary_path: Path | None = None,
) -> dict[str, Any]:
    selected_modes = memory_modes or list(LIVE_MEMORY_MODES)
    mode_results: dict[str, dict[str, Any]] = {}
    for mode in selected_modes:
        mode_results[mode] = run_memoryarena_openclaw(
            corpus,
            sidecar_url=sidecar_url,
            namespace_id=namespace_id,
            openclaw_command=openclaw_command,
            openclaw_agent_id=openclaw_agent_id,
            openclaw_profile=openclaw_profile,
            timeout_seconds=timeout_seconds,
            local=local,
            artifact_root=artifact_root,
            memory_mode=mode,
            isolated_openclaw=isolated_openclaw,
            base_config_path=base_config_path,
            prompt_limit=prompt_limit,
            debug_commands=debug_commands,
            raw_stream=raw_stream,
            raw_stream_path=raw_stream_path,
        )
    summary = _build_compare_summary(
        corpus=corpus,
        mode_results=mode_results,
        openclaw_version=_resolve_openclaw_version(openclaw_command),
        openclaw_command=openclaw_command,
        memory_modes=selected_modes,
    )
    if public_summary_path is not None:
        write_public_memoryarena_summary(public_summary_path, summary)
    return {
        "summary": summary,
        "modes": mode_results,
        "public_summary_path": str(public_summary_path) if public_summary_path is not None else None,
    }
