from __future__ import annotations

from pathlib import Path
import json

from sqlalchemy import select
import typer

from memoria.demo_data import demo_catalog, load_demo
from memoria.engine import MemoryEngine
from memoria.memoryarena import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_CACHE_DIR,
    DEFAULT_REVISION,
    DEFAULT_SMOKE_LIMIT,
    LIVE_SUPPORTED_SUITES,
    LIVE_MEMORY_MODES,
    PUBLIC_SUMMARY_PATH,
    compare_memoryarena_openclaw,
    evaluate_memoryarena_proxy,
    load_memoryarena_corpus,
    run_memoryarena_openclaw,
)
from memoria.models import Episode
from memoria.sidecar import create_sidecar_server

app = typer.Typer(help="Memoria prototype CLI.")


def _engine(db: str) -> MemoryEngine:
    return MemoryEngine(database_url=db)


@app.command("init")
def init_command(db: str = typer.Option("sqlite:///memoria.db", help="SQLite database URL.")) -> None:
    _engine(db)
    typer.echo(f"Initialised database at {db}")


@app.command("ingest-demo")
def ingest_demo(
    name: str = typer.Argument(..., help="Demo name."),
    db: str = typer.Option("sqlite:///memoria.db", help="SQLite database URL."),
) -> None:
    engine = _engine(db)
    demo = load_demo(name)
    engine.add_messages(
        demo["episodes"],
        namespace_id=demo["namespace_id"],
        user_id=demo["user_id"],
        agent_id=demo["agent_id"],
        session_id=demo["session_id"],
    )
    stats = engine.consolidate(namespace_id=demo["namespace_id"])
    typer.echo(json.dumps({"demo": name, "stats": stats}, indent=2))


@app.command("run-demo")
def run_demo(
    name: str = typer.Argument(..., help="Demo name."),
    db: str = typer.Option("sqlite:///memoria.db", help="SQLite database URL."),
) -> None:
    engine = _engine(db)
    demo = load_demo(name)
    with engine.session_factory() as session:
        has_data = session.scalar(select(Episode.id).where(Episode.namespace_id == demo["namespace_id"]).limit(1)) is not None
    if not has_data:
        engine.add_messages(
            demo["episodes"],
            namespace_id=demo["namespace_id"],
            user_id=demo["user_id"],
            agent_id=demo["agent_id"],
            session_id=demo["session_id"],
        )
        engine.consolidate(namespace_id=demo["namespace_id"])
    run = engine.start_run(
        namespace_id=demo["namespace_id"],
        user_id=demo["user_id"],
        agent_id=demo["agent_id"],
        session_id=demo["session_id"],
    )
    typer.echo(f"run_id={run['run_id']}")
    for step in demo["steps"]:
        working_memory = engine.process_step(
            run["run_id"],
            step_type=step["step_type"],
            text=step["text"],
            metadata=step.get("metadata"),
        )
        typer.echo(json.dumps({"step_type": step["step_type"], "working_memory": working_memory}, indent=2))
    typer.echo(json.dumps(engine.get_debug_trace(run["run_id"]), indent=2))


@app.command("search")
def search_command(
    query: str = typer.Argument(..., help="Query string."),
    namespace_id: str = typer.Option("default", help="Namespace id."),
    db: str = typer.Option("sqlite:///memoria.db", help="SQLite database URL."),
) -> None:
    engine = _engine(db)
    typer.echo(json.dumps(engine.search(query, namespace_id=namespace_id), indent=2))


@app.command("consolidate")
def consolidate_command(
    namespace_id: str = typer.Option("default", help="Namespace id."),
    force: bool = typer.Option(False, help="Reprocess all episodes."),
    db: str = typer.Option("sqlite:///memoria.db", help="SQLite database URL."),
) -> None:
    engine = _engine(db)
    typer.echo(json.dumps(engine.consolidate(namespace_id=namespace_id, force=force), indent=2))


@app.command("export-graph")
def export_graph_command(
    namespace_id: str = typer.Option("default", help="Namespace id."),
    out: Path = typer.Option(Path("graph.json"), help="Output json path."),
    db: str = typer.Option("sqlite:///memoria.db", help="SQLite database URL."),
) -> None:
    engine = _engine(db)
    payload = engine.export_graph(namespace_id=namespace_id, output_path=out)
    typer.echo(f"Exported {len(payload['nodes'])} nodes and {len(payload['edges'])} edges to {out}")


@app.command("evaluate")
def evaluate_command(
    name: str = typer.Option("all", help="Demo name or 'all'."),
    db: str = typer.Option("sqlite:///memoria.db", help="SQLite database URL."),
) -> None:
    from memoria.evaluation import evaluate_demo

    engine = _engine(db)
    names = list(demo_catalog()) if name == "all" else [name]
    results = [evaluate_demo(engine, demo_name) for demo_name in names]
    typer.echo(json.dumps(results, indent=2))


@app.command("memoryarena-sync")
def memoryarena_sync_command(
    suite: list[str] = typer.Option([], "--suite", help="MemoryArena suite to sync. Repeat to select multiple."),
    revision: str = typer.Option(DEFAULT_REVISION, help="Dataset revision to cache."),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, help="Dataset cache directory."),
    data_root: Path | None = typer.Option(None, help="Optional local fixture root for offline development."),
    smoke: bool = typer.Option(False, help="Only touch the smoke slice instead of every case."),
) -> None:
    corpus = load_memoryarena_corpus(
        suites=suite or None,
        revision=revision,
        cache_dir=cache_dir,
        data_root=data_root,
        limit_per_suite=DEFAULT_SMOKE_LIMIT if smoke else None,
    )
    typer.echo(
        json.dumps(
            {
                "dataset_name": corpus.dataset_name,
                "revision": corpus.revision,
                "source": corpus.source,
                "suites": corpus.suites,
                "counts": corpus.counts,
                "loaded_task_count": len(corpus.tasks),
            },
            indent=2,
        )
    )


@app.command("memoryarena-eval")
def memoryarena_eval_command(
    db: str = typer.Option("sqlite:///memoria.db", help="SQLite database URL."),
    suite: list[str] = typer.Option([], "--suite", help="MemoryArena suite to evaluate. Repeat to select multiple."),
    revision: str = typer.Option(DEFAULT_REVISION, help="Dataset revision to evaluate."),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, help="Dataset cache directory."),
    data_root: Path | None = typer.Option(None, help="Optional local fixture root for offline development."),
    smoke: bool = typer.Option(True, "--smoke/--full", help="Run the smoke slice by default."),
    limit: int | None = typer.Option(None, help="Optional per-suite task limit."),
    artifact_root: Path = typer.Option(DEFAULT_ARTIFACT_ROOT, help="Artifact directory root."),
) -> None:
    engine = _engine(db)
    limit_per_suite = limit if limit is not None else (DEFAULT_SMOKE_LIMIT if smoke else None)
    corpus = load_memoryarena_corpus(
        suites=suite or None,
        revision=revision,
        cache_dir=cache_dir,
        data_root=data_root,
        limit_per_suite=limit_per_suite,
    )
    result = evaluate_memoryarena_proxy(
        engine,
        corpus,
        artifact_root=artifact_root,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("memoryarena-openclaw")
def memoryarena_openclaw_command(
    suite: list[str] = typer.Option([], "--suite", help="MemoryArena suite to run through OpenClaw. Repeat to select multiple."),
    revision: str = typer.Option(DEFAULT_REVISION, help="Dataset revision to evaluate."),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, help="Dataset cache directory."),
    data_root: Path | None = typer.Option(None, help="Optional local fixture root for offline development."),
    smoke: bool = typer.Option(True, "--smoke/--full", help="Run the smoke slice by default."),
    limit: int | None = typer.Option(None, help="Optional per-suite task limit."),
    artifact_root: Path = typer.Option(DEFAULT_ARTIFACT_ROOT, help="Artifact directory root."),
    sidecar_url: str = typer.Option("http://127.0.0.1:18733", help="Memoria sidecar base URL."),
    namespace_id: str = typer.Option("openclaw.default", help="Memoria namespace id used by the OpenClaw context engine."),
    openclaw_command: str = typer.Option("openclaw", help="OpenClaw executable to invoke."),
    openclaw_agent_id: str = typer.Option("main", help="OpenClaw agent id to use for benchmark runs."),
    openclaw_profile: str | None = typer.Option(None, help="Optional OpenClaw profile."),
    timeout_seconds: int = typer.Option(90, help="Per-turn timeout for OpenClaw agent runs."),
    local: bool = typer.Option(False, "--local/--gateway", help="Use gateway-backed OpenClaw execution by default."),
    memory_mode: str = typer.Option("native", help="Memory mode: baseline, native, or prefetch."),
    isolated_openclaw: bool = typer.Option(
        True,
        "--isolated-openclaw/--current-openclaw",
        help="Run benchmarks against an isolated temporary OpenClaw config/state/workspace by default.",
    ),
    openclaw_base_config: Path | None = typer.Option(None, help="Optional base OpenClaw config to clone for isolated runs."),
    prompt_limit: int = typer.Option(4, help="Prompt addition working-memory limit for Memoria-backed modes."),
    debug_commands: bool = typer.Option(False, help="Enable commands.debug in the isolated OpenClaw config."),
    raw_stream: bool = typer.Option(False, help="Enable OpenClaw raw stream logging for this run."),
    raw_stream_path: Path | None = typer.Option(None, help="Optional raw stream jsonl path."),
) -> None:
    if memory_mode not in LIVE_MEMORY_MODES:
        raise typer.BadParameter(f"memory_mode must be one of: {', '.join(LIVE_MEMORY_MODES)}")
    selected_suites = suite or list(LIVE_SUPPORTED_SUITES)
    limit_per_suite = limit if limit is not None else (DEFAULT_SMOKE_LIMIT if smoke else None)
    corpus = load_memoryarena_corpus(
        suites=selected_suites,
        revision=revision,
        cache_dir=cache_dir,
        data_root=data_root,
        limit_per_suite=limit_per_suite,
    )
    result = run_memoryarena_openclaw(
        corpus,
        sidecar_url=sidecar_url,
        namespace_id=namespace_id,
        openclaw_command=openclaw_command,
        openclaw_agent_id=openclaw_agent_id,
        openclaw_profile=openclaw_profile,
        timeout_seconds=timeout_seconds,
        local=local,
        artifact_root=artifact_root,
        memory_mode=memory_mode,
        isolated_openclaw=isolated_openclaw,
        base_config_path=openclaw_base_config,
        prompt_limit=prompt_limit,
        debug_commands=debug_commands,
        raw_stream=raw_stream,
        raw_stream_path=raw_stream_path,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("memoryarena-compare")
def memoryarena_compare_command(
    suite: list[str] = typer.Option([], "--suite", help="MemoryArena suite to run through OpenClaw. Repeat to select multiple."),
    revision: str = typer.Option(DEFAULT_REVISION, help="Dataset revision to evaluate."),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, help="Dataset cache directory."),
    data_root: Path | None = typer.Option(None, help="Optional local fixture root for offline development."),
    smoke: bool = typer.Option(True, "--smoke/--full", help="Run the smoke slice by default."),
    limit: int | None = typer.Option(None, help="Optional per-suite task limit."),
    artifact_root: Path = typer.Option(DEFAULT_ARTIFACT_ROOT, help="Artifact directory root."),
    sidecar_url: str = typer.Option("http://127.0.0.1:18733", help="Memoria sidecar base URL."),
    namespace_id: str = typer.Option("openclaw.default", help="Memoria namespace id used by the OpenClaw context engine."),
    openclaw_command: str = typer.Option("openclaw", help="OpenClaw executable to invoke."),
    openclaw_agent_id: str = typer.Option("main", help="OpenClaw agent id to use for benchmark runs."),
    openclaw_profile: str | None = typer.Option(None, help="Optional OpenClaw profile."),
    timeout_seconds: int = typer.Option(90, help="Per-turn timeout for OpenClaw agent runs."),
    local: bool = typer.Option(False, "--local/--gateway", help="Use gateway-backed OpenClaw execution by default."),
    memory_mode: list[str] = typer.Option(list(LIVE_MEMORY_MODES), "--memory-mode", help="Repeat to select baseline/native/prefetch benchmark modes."),
    isolated_openclaw: bool = typer.Option(
        True,
        "--isolated-openclaw/--current-openclaw",
        help="Run benchmarks against an isolated temporary OpenClaw config/state/workspace by default.",
    ),
    openclaw_base_config: Path | None = typer.Option(None, help="Optional base OpenClaw config to clone for isolated runs."),
    prompt_limit: int = typer.Option(4, help="Prompt addition working-memory limit for Memoria-backed modes."),
    debug_commands: bool = typer.Option(False, help="Enable commands.debug in the isolated OpenClaw config."),
    raw_stream: bool = typer.Option(False, help="Enable OpenClaw raw stream logging for this run."),
    raw_stream_path: Path | None = typer.Option(None, help="Optional raw stream jsonl path."),
    public_summary_path: Path = typer.Option(PUBLIC_SUMMARY_PATH, help="Path to write the compact public benchmark summary."),
) -> None:
    for mode in memory_mode:
        if mode not in LIVE_MEMORY_MODES:
            raise typer.BadParameter(f"memory_mode must be one of: {', '.join(LIVE_MEMORY_MODES)}")
    selected_suites = suite or list(LIVE_SUPPORTED_SUITES)
    limit_per_suite = limit if limit is not None else (DEFAULT_SMOKE_LIMIT if smoke else None)
    corpus = load_memoryarena_corpus(
        suites=selected_suites,
        revision=revision,
        cache_dir=cache_dir,
        data_root=data_root,
        limit_per_suite=limit_per_suite,
    )
    result = compare_memoryarena_openclaw(
        corpus,
        sidecar_url=sidecar_url,
        namespace_id=namespace_id,
        openclaw_command=openclaw_command,
        openclaw_agent_id=openclaw_agent_id,
        openclaw_profile=openclaw_profile,
        timeout_seconds=timeout_seconds,
        local=local,
        artifact_root=artifact_root,
        memory_modes=memory_mode,
        isolated_openclaw=isolated_openclaw,
        base_config_path=openclaw_base_config,
        prompt_limit=prompt_limit,
        debug_commands=debug_commands,
        raw_stream=raw_stream,
        raw_stream_path=raw_stream_path,
        public_summary_path=public_summary_path,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("serve")
def serve_command(
    host: str = typer.Option("127.0.0.1", help="Host interface for the Memoria sidecar."),
    port: int = typer.Option(18733, help="Port for the Memoria sidecar."),
    db: str = typer.Option("sqlite:///memoria.db", help="SQLite database URL."),
) -> None:
    engine = _engine(db)
    server = create_sidecar_server(engine, host=host, port=port)
    typer.echo(f"Memoria sidecar listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("Memoria sidecar stopped.")
    finally:
        server.server_close()
