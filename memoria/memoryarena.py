from __future__ import annotations

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


def _timestamp_slug(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


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


def _content_to_text(item: dict[str, Any]) -> str:
    return str(item.get("content") or item.get("text") or "")


def _prompt_addition(working_memory: list[dict[str, Any]], limit: int = 4) -> str:
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
        "suite_case_counts": {
            suite: sum(1 for case in cases if case["suite"] == suite)
            for suite in sorted({case["suite"] for case in cases})
        },
        "support_recall_at_5": _safe_mean([case.get("support_recall_at_5") for case in cases]),
        "support_precision_at_5": _safe_mean([case.get("support_precision_at_5") for case in cases]),
        "prompt_support_coverage": _safe_mean([case.get("prompt_support_coverage") for case in cases]),
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


def evaluate_memoryarena_proxy(
    engine: MemoryEngine,
    corpus: MemoryArenaCorpus,
    *,
    artifact_root: Path | None = None,
    prompt_limit: int = 4,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for task in corpus.tasks:
        namespace_id = f"memoryarena.proxy.{task.suite}.{task.task_id}"
        session_id = f"{task.suite}-{task.task_id}"
        if task.background_items:
            engine.add_messages(
                [
                    {
                        "role": "system",
                        "source_type": "system_note",
                        "content": item,
                        "metadata_json": {
                            "suite": task.suite,
                            "task_id": task.task_id,
                            "kind": "background",
                        },
                    }
                    for item in task.background_items
                ],
                namespace_id=namespace_id,
                user_id="memoryarena",
                agent_id="proxy",
                session_id=session_id,
            )
            engine.consolidate(namespace_id=namespace_id)
        run_ctx = engine.start_run(
            namespace_id=namespace_id,
            user_id="memoryarena",
            agent_id="proxy",
            session_id=session_id,
        )
        prior_artifacts = [
            {
                "artifact_id": f"background-{idx}",
                "kind": "background",
                "text": background,
            }
            for idx, background in enumerate(task.background_items)
        ]
        for turn in task.turns:
            started_at = perf_counter()
            working_memory = engine.process_step(
                run_ctx["run_id"],
                step_type="user_input",
                text=turn.question,
                metadata={"suite": task.suite, "task_id": task.task_id, "turn_index": turn.turn_index},
            )
            latency_ms = (perf_counter() - started_at) * 1000.0
            prompt_addition = _prompt_addition(working_memory, limit=prompt_limit)
            expected_support = _expected_support_artifacts(prior_artifacts, turn.gold_answer_text)
            top_items = working_memory[:5]
            hit_count = sum(
                1
                for artifact in expected_support
                if any(_artifact_hit(artifact.text.lower(), _content_to_text(item).lower()) for item in top_items)
            )
            prompt_hits = sum(
                1 for artifact in expected_support if _artifact_hit(artifact.text.lower(), prompt_addition.lower())
            )
            case = {
                "mode": "memoryarena_proxy",
                "status": "ok",
                "suite": task.suite,
                "task_id": task.task_id,
                "turn_index": turn.turn_index,
                "question": turn.question,
                "gold_answer_text": turn.gold_answer_text,
                "expected_support": [asdict(artifact) for artifact in expected_support],
                "working_memory": working_memory,
                "prompt_addition": prompt_addition,
                "support_expected_count": len(expected_support),
                "support_recall_at_5": (
                    hit_count / len(expected_support) if expected_support else None
                ),
                "support_precision_at_5": (hit_count / min(len(top_items), 5)) if top_items else None,
                "prompt_support_coverage": (
                    prompt_hits / len(expected_support) if expected_support else None
                ),
                "prompt_token_count": word_count(prompt_addition),
                "latency_ms": latency_ms,
                "trace": engine.get_debug_trace(run_ctx["run_id"]),
            }
            cases.append(case)
            engine.add_messages(
                [
                    {
                        "role": "assistant",
                        "source_type": "agent_message",
                        "content": turn.gold_answer_text,
                        "metadata_json": {
                            "suite": task.suite,
                            "task_id": task.task_id,
                            "turn_index": turn.turn_index,
                            "kind": "teacher_forced_answer",
                        },
                    }
                ],
                namespace_id=namespace_id,
                user_id="memoryarena",
                agent_id="proxy",
                session_id=session_id,
            )
            engine.consolidate(namespace_id=namespace_id)
            prior_artifacts.append(
                {
                    "artifact_id": f"answer-{turn.turn_index}",
                    "kind": "gold_answer",
                    "text": turn.gold_answer_text,
                }
            )
    summary = _summary_from_cases("memoryarena_proxy", cases)
    config = {
        "dataset_name": corpus.dataset_name,
        "revision": corpus.revision,
        "source": corpus.source,
        "suites": corpus.suites,
        "available_counts": corpus.counts,
        "selected_counts": _selected_counts(corpus),
        "prompt_limit": prompt_limit,
    }
    artifact_dir = None
    if artifact_root is not None:
        artifact_dir = _write_artifacts(
            artifact_root=artifact_root,
            mode="memoryarena_proxy",
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


def _score_live_case(turn: MemoryArenaTurn, reply_text: str) -> dict[str, Any]:
    gold_text = turn.gold_answer_text
    metrics = {
        "token_f1": token_f1_score(gold_text, reply_text),
        "exact_match": gold_text.strip().lower() == reply_text.strip().lower(),
    }
    if turn.suite == "group_travel_planner":
        field_values = _travel_field_values(turn.gold_answer_value)
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
                metrics = _score_live_case(turn, reply_text) if status == "ok" else {}
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
