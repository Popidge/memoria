from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4
import json

from sqlalchemy import select

from memoria.activation import ActivationService
from memoria.config import DEFAULT_CONFIG, EngineConfig
from memoria.context import build_memory_context_packet, render_memory_context_packet
from memoria.consolidation import ConsolidationService
from memoria.db import create_db_engine, init_db, make_session_factory, session_scope
from memoria.entity_resolution import EntityResolver
from memoria.ingestion import EpisodeInput, IngestionService
from memoria.models import (
    ActivationState,
    EdgeDescriptor,
    Entity,
    Episode,
    EpisodeChunk,
    Fact,
    GraphEdge,
    NodeDescriptor,
    NodeType,
    ProvenanceLink,
    SummaryNode,
    WorkingMemoryItem,
)
from memoria.retrieval import RetrievalService
from memoria.utils import TextEmbedder


@dataclass
class RunContext:
    run_id: str
    namespace_id: str
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    step_index: int = 0


class MemoryEngine:
    def __init__(self, config: EngineConfig | None = None, database_url: str | None = None):
        self.config = config or DEFAULT_CONFIG
        if database_url:
            self.config.database_url = database_url
        self.embedder = TextEmbedder(
            model_name=self.config.embedding_model_name,
            dimensions=self.config.embedding_dimensions,
            enabled=self.config.use_sentence_transformers,
        )
        self.engine = create_db_engine(self.config.database_url)
        init_db(self.engine)
        self.session_factory = make_session_factory(self.engine)
        self.ingestion = IngestionService(self.embedder)
        self.retrieval = RetrievalService(self.embedder, self.config)
        self.resolver = EntityResolver(self.embedder, self.config)
        self.consolidation_service = ConsolidationService(self.embedder, self.resolver, self.config)
        self.activation_service = ActivationService(self.retrieval, self.config)
        self._runs: dict[str, RunContext] = {}
        self._runs_lock = RLock()
        self._run_locks: dict[str, RLock] = {}

    def add_episode(self, **kwargs) -> dict[str, Any]:
        with session_scope(self.session_factory) as session:
            episode = self.ingestion.add_episode(session, EpisodeInput(**kwargs))
            return self._episode_dict(episode)

    def add_messages(
        self,
        messages: list[dict[str, Any]],
        namespace_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        namespace_id = namespace_id or self.config.namespace_defaults.namespace_id
        with session_scope(self.session_factory) as session:
            episodes = self.ingestion.add_messages(
                session,
                messages,
                namespace_id=namespace_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
            )
            return [self._episode_dict(episode) for episode in episodes]

    def update_episode(self, episode_id: int, new_content: str) -> dict[str, Any]:
        with session_scope(self.session_factory) as session:
            episode = self.ingestion.update_episode(session, episode_id, new_content)
            episode.metadata_json = {k: v for k, v in episode.metadata_json.items() if k != "consolidated_at"}
            session.add(episode)
            return self._episode_dict(episode)

    def delete_episode(self, episode_id: int) -> None:
        with session_scope(self.session_factory) as session:
            self.ingestion.delete_episode(session, episode_id)

    def search(
        self,
        text: str,
        namespace_id: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        namespace_id = namespace_id or self.config.namespace_defaults.namespace_id
        with session_scope(self.session_factory) as session:
            results = self.retrieval.search(session, text, namespace_id, filters=filters, limit=limit)
            return [
                {
                    "node_key": item.node_key.as_string(),
                    "text": item.text,
                    "semantic_score": item.semantic_score,
                    "keyword_score": item.keyword_score,
                    "scope_score": item.scope_score,
                    "direct_score": item.direct_score,
                }
                for item in results
            ]

    def start_run(
        self,
        namespace_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        namespace_id = namespace_id or self.config.namespace_defaults.namespace_id
        run = RunContext(
            run_id=str(uuid4()),
            namespace_id=namespace_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        with self._runs_lock:
            self._runs[run.run_id] = run
            self._run_locks[run.run_id] = RLock()
        return run.__dict__.copy()

    def attach_run(
        self,
        run_id: str,
        *,
        namespace_id: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        step_index: int = 0,
    ) -> dict[str, Any]:
        with self._runs_lock:
            existing = self._runs.get(run_id)
            if existing is not None:
                return existing.__dict__.copy()
            run = RunContext(
                run_id=run_id,
                namespace_id=namespace_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                step_index=step_index,
            )
            self._runs[run_id] = run
            self._run_locks[run_id] = RLock()
            return run.__dict__.copy()

    def process_step(
        self,
        run_id: str,
        step_type: str,
        text: str,
        linked_tool_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        result = self.process_step_result(
            run_id,
            step_type=step_type,
            text=text,
            linked_tool_name=linked_tool_name,
            metadata=metadata,
        )
        return result["working_memory"]

    def process_step_result(
        self,
        run_id: str,
        step_type: str,
        text: str,
        linked_tool_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_lock = self._require_run_lock(run_id)
        with run_lock:
            run = self._require_run(run_id)
            with session_scope(self.session_factory) as session:
                result = self.activation_service.process_step(
                    session,
                    run_id=run.run_id,
                    step_index=run.step_index,
                    namespace_id=run.namespace_id,
                    step_type=step_type,
                    text=text,
                    linked_tool_name=linked_tool_name,
                    metadata=metadata,
                    filters={
                        "user_id": run.user_id,
                        "agent_id": run.agent_id,
                        "session_id": run.session_id,
                    },
                )
            run.step_index += 1
            packet = build_memory_context_packet(
                run_id=result.run_id,
                step_index=result.step_index,
                query=text,
                working_memory=result.working_memory,
                metadata={"step_type": step_type, "linked_tool_name": linked_tool_name, **dict(metadata or {})},
                limit=len(result.working_memory),
            )
            return {
                "run_id": result.run_id,
                "step_index": result.step_index,
                "working_memory": result.working_memory,
                "memory_context_packet": packet,
                "prompt_addition": render_memory_context_packet(packet),
                "debug": result.debug,
            }

    def get_run_context(self, run_id: str) -> dict[str, Any]:
        run = self._require_run(run_id)
        return run.__dict__.copy()

    def get_working_memory(self, run_id: str, step_index: int | None = None) -> list[dict[str, Any]]:
        with session_scope(self.session_factory) as session:
            query = select(WorkingMemoryItem).where(WorkingMemoryItem.run_id == run_id)
            if step_index is not None:
                query = query.where(WorkingMemoryItem.step_index == step_index)
            items = session.scalars(query.order_by(WorkingMemoryItem.step_index, WorkingMemoryItem.score.desc())).all()
            return [
                {
                    "step_index": item.step_index,
                    "node_key": f"{item.node_type}:{item.node_id}",
                    "content_type": item.content_type,
                    "content": item.content,
                    "score": item.score,
                    "source": item.source_json,
                    "slot": item.source_json.get("slot", "primary"),
                    "reason": item.source_json.get("reason", ""),
                    "evidence_ids": item.source_json.get("evidence_ids", []),
                    "node_class": item.source_json.get("node_class"),
                }
                for item in items
            ]

    def consolidate(self, namespace_id: str | None = None, force: bool = False) -> dict[str, int]:
        namespace_id = namespace_id or self.config.namespace_defaults.namespace_id
        with session_scope(self.session_factory) as session:
            return self.consolidation_service.consolidate(session, namespace_id, force=force)

    def get_debug_trace(self, run_id: str) -> dict[str, Any]:
        with session_scope(self.session_factory) as session:
            states = session.scalars(
                select(ActivationState)
                .where(ActivationState.run_id == run_id)
                .order_by(ActivationState.step_index, ActivationState.activation_value.desc())
            ).all()
            working_items = session.scalars(
                select(WorkingMemoryItem)
                .where(WorkingMemoryItem.run_id == run_id)
                .order_by(WorkingMemoryItem.step_index, WorkingMemoryItem.score.desc())
            ).all()
            grouped_states: dict[int, list[dict[str, Any]]] = defaultdict(list)
            grouped_items: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for state in states:
                grouped_states[state.step_index].append(
                    {
                        "node_key": f"{state.node_type}:{state.node_id}",
                        "activation_value": state.activation_value,
                        "direct_score": state.direct_score,
                        "spread_score": state.spread_score,
                        "recency_score": state.recency_score,
                        "importance_score": state.importance_score,
                        "inhibition_penalty": state.inhibition_penalty,
                        "surfaced": state.surfaced,
                        "reason": state.reason_json,
                    }
                )
            for item in working_items:
                grouped_items[item.step_index].append(
                    {
                        "node_key": f"{item.node_type}:{item.node_id}",
                        "content_type": item.content_type,
                        "content": item.content,
                        "score": item.score,
                        "source": item.source_json,
                        "slot": item.source_json.get("slot", "primary"),
                        "reason": item.source_json.get("reason", ""),
                        "evidence_ids": item.source_json.get("evidence_ids", []),
                        "node_class": item.source_json.get("node_class"),
                    }
                )
            steps = sorted(set(grouped_states) | set(grouped_items))
            return {
                "run_id": run_id,
                "steps": [
                    {
                        "step_index": step,
                        "activation": grouped_states.get(step, []),
                        "working_memory": grouped_items.get(step, []),
                    }
                    for step in steps
                ],
            }

    def export_graph(self, namespace_id: str | None = None, output_path: str | Path | None = None) -> dict[str, Any]:
        namespace_id = namespace_id or self.config.namespace_defaults.namespace_id
        with session_scope(self.session_factory) as session:
            nodes = []
            descriptors = {
                (descriptor.node_type, descriptor.node_id): descriptor
                for descriptor in session.scalars(select(NodeDescriptor).where(NodeDescriptor.namespace_id == namespace_id)).all()
            }
            for model, node_type in (
                (Entity, NodeType.ENTITY.value),
                (Fact, NodeType.FACT.value),
                (SummaryNode, NodeType.SUMMARY.value),
                (EpisodeChunk, NodeType.EPISODE_CHUNK.value),
                (Episode, NodeType.EPISODE.value),
            ):
                for row in session.scalars(select(model).where(model.namespace_id == namespace_id)).all():
                    label = getattr(row, "canonical_name", None) or getattr(row, "title", None) or getattr(row, "summary", None) or getattr(row, "content_summary", None) or getattr(row, "text", None) or str(row.id)
                    descriptor = descriptors.get((node_type, row.id))
                    nodes.append(
                        {
                            "id": f"{node_type}:{row.id}",
                            "type": node_type,
                            "label": label,
                            "node_class": descriptor.node_class if descriptor else None,
                            "facets": descriptor.facets_json if descriptor else {},
                            "evidence_count": descriptor.evidence_count if descriptor else 0,
                        }
                    )
            edge_descriptors = {
                descriptor.edge_id: descriptor
                for descriptor in session.scalars(select(EdgeDescriptor)).all()
            }
            edges = [
                {
                    "source": f"{edge.source_node_type}:{edge.source_node_id}",
                    "target": f"{edge.target_node_type}:{edge.target_node_id}",
                    "edge_type": edge.edge_type,
                    "weight": edge.weight,
                    "relation_class": edge_descriptors[edge.id].relation_class if edge.id in edge_descriptors else None,
                    "confidence": edge_descriptors[edge.id].confidence if edge.id in edge_descriptors else None,
                    "evidence_count": edge_descriptors[edge.id].evidence_count if edge.id in edge_descriptors else 0,
                }
                for edge in session.scalars(select(GraphEdge).where(GraphEdge.namespace_id == namespace_id)).all()
            ]
            provenance = [
                {"fact_id": link.fact_id, "episode_id": link.episode_id}
                for link in session.scalars(
                    select(ProvenanceLink).join(Fact, Fact.id == ProvenanceLink.fact_id).where(Fact.namespace_id == namespace_id)
                ).all()
            ]
        payload = {"namespace_id": namespace_id, "nodes": nodes, "edges": edges, "provenance": provenance}
        if output_path:
            Path(output_path).write_text(json.dumps(payload, indent=2))
        return payload

    def _require_run(self, run_id: str) -> RunContext:
        with self._runs_lock:
            run = self._runs.get(run_id)
            if run is None:
                raise ValueError(f"Unknown run_id: {run_id}")
            return run

    def _require_run_lock(self, run_id: str) -> RLock:
        with self._runs_lock:
            run_lock = self._run_locks.get(run_id)
            if run_lock is None:
                raise ValueError(f"Unknown run_id: {run_id}")
            return run_lock

    def _episode_dict(self, episode: Episode) -> dict[str, Any]:
        return {
            "id": episode.id,
            "namespace_id": episode.namespace_id,
            "user_id": episode.user_id,
            "agent_id": episode.agent_id,
            "session_id": episode.session_id,
            "source_type": episode.source_type,
            "content_summary": episode.content_summary,
            "created_at": episode.created_at.isoformat(),
            "metadata_json": episode.metadata_json,
        }
