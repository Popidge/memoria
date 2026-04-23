from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select

from memoria.db import session_scope
from memoria.engine import MemoryEngine
from memoria.memoryarena import load_memoryarena_corpus
from memoria.models import ActivationState, Entity, Episode, ExperimentRun, ExperimentTurn, Fact, GraphEdge, SummaryNode, WorkingMemoryItem
from memoria.providers import ProviderConfig, ProviderResult, build_provider
from memoria.utils import word_count


DEFAULT_WORKBENCH_FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "memoryarena"


@dataclass
class RunSpec:
    title: str = "Untitled run"
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    namespace_id: str = "workbench.default"
    user_id: str | None = "local-user"
    agent_id: str | None = "assistant"
    session_id: str | None = None
    system_prompt: str = "You are a helpful assistant."
    prompt_limit: int = 4
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkbenchService:
    def __init__(self, engine: MemoryEngine, memoryarena_data_root: Path | None = None):
        self.engine = engine
        self.memoryarena_data_root = memoryarena_data_root or (
            DEFAULT_WORKBENCH_FIXTURES if DEFAULT_WORKBENCH_FIXTURES.exists() else None
        )

    def create_run(self, spec: RunSpec) -> dict[str, Any]:
        session_id = spec.session_id or f"workbench-{uuid4()}"
        run_context = self.engine.start_run(
            namespace_id=spec.namespace_id,
            user_id=spec.user_id,
            agent_id=spec.agent_id,
            session_id=session_id,
        )
        experiment = ExperimentRun(
            id=str(uuid4()),
            title=spec.title,
            status="active",
            provider_type=spec.provider.provider_type,
            model_name=spec.provider.model_name,
            api_base_url=spec.provider.api_base_url,
            api_key_env=spec.provider.api_key_env,
            namespace_id=spec.namespace_id,
            user_id=spec.user_id,
            agent_id=spec.agent_id,
            session_id=session_id,
            memory_run_id=run_context["run_id"],
            system_prompt=spec.system_prompt,
            config_json={
                "provider": asdict(spec.provider),
                "prompt_limit": spec.prompt_limit,
                "metadata": spec.metadata,
            },
            metrics_json={},
        )
        with session_scope(self.engine.session_factory) as session:
            session.add(experiment)
        return self.get_run(experiment.id)

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with session_scope(self.engine.session_factory) as session:
            rows = session.scalars(
                select(ExperimentRun)
                .order_by(ExperimentRun.updated_at.desc())
                .limit(limit)
            ).all()
            return [self._run_summary(session, row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any]:
        with session_scope(self.engine.session_factory) as session:
            row = session.get(ExperimentRun, run_id)
            if row is None:
                raise KeyError(f"Unknown run_id: {run_id}")
            return self._run_detail(session, row)

    def list_turns(self, run_id: str) -> list[dict[str, Any]]:
        with session_scope(self.engine.session_factory) as session:
            run = session.get(ExperimentRun, run_id)
            if run is None:
                raise KeyError(f"Unknown run_id: {run_id}")
            turns = session.scalars(
                select(ExperimentTurn)
                .where(ExperimentTurn.experiment_run_id == run_id)
                .order_by(ExperimentTurn.turn_index)
            ).all()
            return [self._turn_dict(turn) for turn in turns]

    def send_user_message(self, run_id: str, message: str) -> dict[str, Any]:
        message = message.strip()
        if not message:
            raise ValueError("message must not be empty")
        with session_scope(self.engine.session_factory) as session:
            run = session.get(ExperimentRun, run_id)
            if run is None:
                raise KeyError(f"Unknown run_id: {run_id}")
            prior_turns = session.scalars(
                select(ExperimentTurn)
                .where(ExperimentTurn.experiment_run_id == run_id)
                .order_by(ExperimentTurn.turn_index)
            ).all()
            turn_index = len(prior_turns)
            provider_config = ProviderConfig(**dict(run.config_json.get("provider") or {}))
            prompt_limit = int(run.config_json.get("prompt_limit", 4))
            self._ensure_memory_run_attached(session, run)
        user_step = self.engine.process_step_result(
            run.memory_run_id,
            step_type="user_input",
            text=message,
            metadata={"phase": "chat", "turn_index": turn_index},
        )
        prompt_addition = _build_prompt_addition(user_step["working_memory"], limit=prompt_limit)
        request_messages = self._request_messages(run_id, run.system_prompt, prompt_addition, message)
        provider = build_provider(provider_config)
        provider_result = provider.generate(request_messages)
        assistant_text = provider_result.text.strip()
        assistant_step = self.engine.process_step_result(
            run.memory_run_id,
            step_type="assistant_message",
            text=assistant_text,
            metadata={"phase": "chat", "turn_index": turn_index},
        )
        self.engine.add_messages(
            [
                {
                    "role": "user",
                    "source_type": "user_message",
                    "content": message,
                    "metadata_json": {"turn_index": turn_index, "phase": "chat"},
                },
                {
                    "role": "assistant",
                    "source_type": "agent_message",
                    "content": assistant_text,
                    "metadata_json": {"turn_index": turn_index, "phase": "chat"},
                },
            ],
            namespace_id=run.namespace_id,
            user_id=run.user_id,
            agent_id=run.agent_id,
            session_id=run.session_id,
        )
        self.engine.consolidate(namespace_id=run.namespace_id)
        turn = ExperimentTurn(
            experiment_run_id=run_id,
            turn_index=turn_index,
            status="ok",
            user_message=message,
            assistant_message=assistant_text,
            prompt_addition=prompt_addition,
            request_messages_json=request_messages,
            provider_payload_json=provider_result.payload,
            usage_json=provider_result.usage,
            metadata_json={
                "prompt_word_count": word_count(prompt_addition),
                "working_memory_count": len(user_step["working_memory"]),
                "assistant_working_memory_count": len(assistant_step["working_memory"]),
            },
            latency_ms=provider_result.latency_ms,
            user_step_index=user_step["step_index"],
            assistant_step_index=assistant_step["step_index"],
        )
        with session_scope(self.engine.session_factory) as session:
            persisted_run = session.get(ExperimentRun, run_id)
            if persisted_run is None:
                raise KeyError(f"Unknown run_id: {run_id}")
            session.add(turn)
            persisted_run.metrics_json = {
                "turn_count": turn_index + 1,
                "last_prompt_word_count": word_count(prompt_addition),
                "last_input_tokens": provider_result.usage.get("input_tokens", 0),
                "last_output_tokens": provider_result.usage.get("output_tokens", 0),
            }
            session.add(persisted_run)
        latest_trace = self.trace(run_id, step_index=assistant_step["step_index"])
        return {
            "run": self.get_run(run_id),
            "turn": self.list_turns(run_id)[-1],
            "latest_trace": latest_trace,
        }

    def trace(self, run_id: str, step_index: int | None = None) -> dict[str, Any]:
        with session_scope(self.engine.session_factory) as session:
            run = session.get(ExperimentRun, run_id)
            if run is None:
                raise KeyError(f"Unknown run_id: {run_id}")
            self._ensure_memory_run_attached(session, run)
        trace = self.engine.get_debug_trace(run.memory_run_id)
        if step_index is not None:
            steps = [step for step in trace["steps"] if step["step_index"] == step_index]
            return {"run_id": run_id, "memory_run_id": run.memory_run_id, "steps": steps}
        return {"run_id": run_id, "memory_run_id": run.memory_run_id, "steps": trace["steps"]}

    def graph_snapshot(self, run_id: str) -> dict[str, Any]:
        with session_scope(self.engine.session_factory) as session:
            run = session.get(ExperimentRun, run_id)
            if run is None:
                raise KeyError(f"Unknown run_id: {run_id}")
        graph = self.engine.export_graph(namespace_id=run.namespace_id)
        latest_trace = self.trace(run_id)
        latest_step = latest_trace["steps"][-1] if latest_trace["steps"] else None
        activation_by_node = {
            item["node_key"]: item["activation_value"]
            for item in (latest_step.get("activation") if latest_step else [])
        }
        return {
            "run_id": run_id,
            "namespace_id": run.namespace_id,
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "latest_step_index": latest_step["step_index"] if latest_step else None,
            "nodes": [
                {**node, "latest_activation": activation_by_node.get(node["id"])}
                for node in graph["nodes"][:200]
            ],
            "edges": graph["edges"][:400],
        }

    def corpus_snapshot(self, run_id: str, limit: int = 20) -> dict[str, Any]:
        with session_scope(self.engine.session_factory) as session:
            run = session.get(ExperimentRun, run_id)
            if run is None:
                raise KeyError(f"Unknown run_id: {run_id}")
            namespace_id = run.namespace_id
            recent_episodes = session.scalars(
                select(Episode)
                .where(Episode.namespace_id == namespace_id)
                .order_by(Episode.created_at.desc())
                .limit(limit)
            ).all()
            entities = session.scalars(
                select(Entity)
                .where(Entity.namespace_id == namespace_id)
                .order_by(Entity.updated_at.desc())
                .limit(limit)
            ).all()
            facts = session.scalars(
                select(Fact)
                .where(Fact.namespace_id == namespace_id)
                .order_by(Fact.updated_at.desc())
                .limit(limit)
            ).all()
            summaries = session.scalars(
                select(SummaryNode)
                .where(SummaryNode.namespace_id == namespace_id)
                .order_by(SummaryNode.updated_at.desc())
                .limit(limit)
            ).all()
            counts = {
                "episodes": session.scalar(select(func.count(Episode.id)).where(Episode.namespace_id == namespace_id)) or 0,
                "entities": session.scalar(select(func.count(Entity.id)).where(Entity.namespace_id == namespace_id)) or 0,
                "facts": session.scalar(select(func.count(Fact.id)).where(Fact.namespace_id == namespace_id)) or 0,
                "summaries": session.scalar(select(func.count(SummaryNode.id)).where(SummaryNode.namespace_id == namespace_id)) or 0,
                "edges": session.scalar(select(func.count(GraphEdge.id)).where(GraphEdge.namespace_id == namespace_id)) or 0,
            }
        return {
            "run_id": run_id,
            "namespace_id": namespace_id,
            "counts": counts,
            "episodes": [
                {
                    "id": item.id,
                    "role": item.role,
                    "source_type": item.source_type,
                    "summary": item.content_summary,
                    "content": item.content_raw,
                    "created_at": item.created_at.isoformat(),
                }
                for item in recent_episodes
            ],
            "entities": [
                {
                    "id": item.id,
                    "name": item.canonical_name,
                    "summary": item.summary,
                    "entity_type": item.entity_type,
                    "updated_at": item.updated_at.isoformat(),
                }
                for item in entities
            ],
            "facts": [
                {
                    "id": item.id,
                    "summary": item.summary,
                    "confidence": item.confidence,
                    "updated_at": item.updated_at.isoformat(),
                }
                for item in facts
            ],
            "summaries": [
                {
                    "id": item.id,
                    "title": item.title,
                    "summary": item.summary,
                    "summary_type": item.summary_type,
                    "updated_at": item.updated_at.isoformat(),
                }
                for item in summaries
            ],
        }

    def memoryarena_suites(self) -> dict[str, Any]:
        if self.memoryarena_data_root is None:
            return {"source": None, "suites": []}
        corpus = load_memoryarena_corpus(data_root=self.memoryarena_data_root)
        return {
            "source": str(self.memoryarena_data_root),
            "dataset_name": corpus.dataset_name,
            "revision": corpus.revision,
            "counts": corpus.counts,
            "suites": corpus.suites,
        }

    def memoryarena_tasks(self, suite: str, limit: int = 20) -> dict[str, Any]:
        if self.memoryarena_data_root is None:
            raise FileNotFoundError("No local MemoryArena fixture root is configured")
        corpus = load_memoryarena_corpus(data_root=self.memoryarena_data_root, suites=[suite], limit_per_suite=limit)
        return {
            "suite": suite,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "turn_count": len(task.turns),
                    "background_items": task.background_items,
                    "metadata": task.metadata,
                }
                for task in corpus.tasks
            ],
        }

    def memoryarena_task(self, suite: str, task_id: str) -> dict[str, Any]:
        if self.memoryarena_data_root is None:
            raise FileNotFoundError("No local MemoryArena fixture root is configured")
        corpus = load_memoryarena_corpus(data_root=self.memoryarena_data_root, suites=[suite])
        task = next((item for item in corpus.tasks if item.task_id == task_id), None)
        if task is None:
            raise KeyError(f"Unknown MemoryArena task: {suite}:{task_id}")
        return {
            "suite": suite,
            "task_id": task_id,
            "background_items": task.background_items,
            "metadata": task.metadata,
            "turns": [
                {
                    "turn_index": turn.turn_index,
                    "question": turn.question,
                    "gold_answer_text": turn.gold_answer_text,
                    "background_items": turn.background_items,
                    "metadata": turn.metadata,
                }
                for turn in task.turns
            ],
        }

    def _request_messages(
        self,
        run_id: str,
        system_prompt: str,
        prompt_addition: str,
        user_message: str,
    ) -> list[dict[str, str]]:
        turns = self.list_turns(run_id)
        messages: list[dict[str, str]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        if prompt_addition.strip():
            messages.append({"role": "system", "content": prompt_addition.strip()})
        for turn in turns:
            messages.append({"role": "user", "content": turn["user_message"]})
            if turn["assistant_message"]:
                messages.append({"role": "assistant", "content": turn["assistant_message"]})
        messages.append({"role": "user", "content": user_message})
        return messages

    def _ensure_memory_run_attached(self, session, run: ExperimentRun) -> None:
        try:
            self.engine.get_run_context(run.memory_run_id)
            return
        except ValueError:
            pass
        max_step = session.scalar(
            select(func.max(ActivationState.step_index)).where(ActivationState.run_id == run.memory_run_id)
        )
        self.engine.attach_run(
            run.memory_run_id,
            namespace_id=run.namespace_id,
            user_id=run.user_id,
            agent_id=run.agent_id,
            session_id=run.session_id,
            step_index=(int(max_step) + 1) if max_step is not None else 0,
        )

    def _run_summary(self, session, run: ExperimentRun) -> dict[str, Any]:
        turn_count = session.scalar(
            select(func.count(ExperimentTurn.id)).where(ExperimentTurn.experiment_run_id == run.id)
        ) or 0
        return {
            "id": run.id,
            "title": run.title,
            "status": run.status,
            "provider_type": run.provider_type,
            "model_name": run.model_name,
            "namespace_id": run.namespace_id,
            "session_id": run.session_id,
            "turn_count": int(turn_count),
            "created_at": run.created_at.isoformat(),
            "updated_at": run.updated_at.isoformat(),
        }

    def _run_detail(self, session, run: ExperimentRun) -> dict[str, Any]:
        summary = self._run_summary(session, run)
        summary.update(
            {
                "api_base_url": run.api_base_url,
                "api_key_env": run.api_key_env,
                "user_id": run.user_id,
                "agent_id": run.agent_id,
                "memory_run_id": run.memory_run_id,
                "system_prompt": run.system_prompt,
                "config": run.config_json,
                "metrics": run.metrics_json,
            }
        )
        return summary

    @staticmethod
    def _turn_dict(turn: ExperimentTurn) -> dict[str, Any]:
        return {
            "id": turn.id,
            "run_id": turn.experiment_run_id,
            "turn_index": turn.turn_index,
            "status": turn.status,
            "user_message": turn.user_message,
            "assistant_message": turn.assistant_message,
            "prompt_addition": turn.prompt_addition,
            "usage": turn.usage_json,
            "metadata": turn.metadata_json,
            "latency_ms": turn.latency_ms,
            "user_step_index": turn.user_step_index,
            "assistant_step_index": turn.assistant_step_index,
            "created_at": turn.created_at.isoformat(),
            "updated_at": turn.updated_at.isoformat(),
        }


def _build_prompt_addition(working_memory: list[dict[str, Any]], limit: int = 4) -> str:
    lines = ["Relevant Memoria recall for this turn:"]
    for item in working_memory[: max(limit, 0)]:
        label = item["content_type"].replace("_", " ")
        lines.append(f"- [{label}] {item['content']}")
    if len(lines) == 1:
        return ""
    lines.append("Use only the items that are directly relevant to the current task.")
    return "\n".join(lines)
