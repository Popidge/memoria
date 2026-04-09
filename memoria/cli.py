from __future__ import annotations

from pathlib import Path
import json

from sqlalchemy import select
import typer

from memoria.demo_data import demo_catalog, load_demo
from memoria.engine import MemoryEngine
from memoria.models import Episode

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
