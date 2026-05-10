from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class SourceType(str, Enum):
    USER_MESSAGE = "user_message"
    AGENT_MESSAGE = "agent_message"
    TOOL_RESULT = "tool_result"
    SYSTEM_NOTE = "system_note"
    IMPORTED_DOC = "imported_doc"


class SummaryType(str, Enum):
    TOPIC = "topic"
    PROCEDURE = "procedure"
    PROJECT = "project"
    USER_PROFILE = "user_profile"
    SESSION_ROLLUP = "session_rollup"
    COMMUNITY = "community"


class NodeType(str, Enum):
    ENTITY = "Entity"
    FACT = "Fact"
    SUMMARY = "SummaryNode"
    EPISODE = "Episode"
    EPISODE_CHUNK = "EpisodeChunk"


class WorkingMemoryContentType(str, Enum):
    ENTITY_SUMMARY = "entity_summary"
    FACT_SUMMARY = "fact_summary"
    SUMMARY_NODE = "summary_node"
    EPISODE_SNIPPET = "episode_snippet"
    MEMORY_ATOM = "memory_atom"


@dataclass(frozen=True)
class NodeKey:
    node_type: NodeType
    node_id: int

    def as_string(self) -> str:
        return f"{self.node_type.value}:{self.node_id}"


class Episode(Base):
    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    namespace_id: Mapped[str] = mapped_column(String(120), index=True)
    user_id: Mapped[str | None] = mapped_column(String(120), index=True, nullable=True)
    agent_id: Mapped[str | None] = mapped_column(String(120), index=True, nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(120), index=True, nullable=True)
    source_type: Mapped[str] = mapped_column(String(40), index=True)
    role: Mapped[str | None] = mapped_column(String(40), nullable=True)
    content_raw: Mapped[str] = mapped_column(Text)
    content_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class EpisodeChunk(Base):
    __tablename__ = "episode_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    namespace_id: Mapped[str] = mapped_column(String(120), index=True)
    episode_id: Mapped[int] = mapped_column(ForeignKey("episodes.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, index=True)
    chunk_type: Mapped[str] = mapped_column(String(60), index=True)
    text: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text, default="")
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    salience_seed: Mapped[float] = mapped_column(Float, default=0.0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    namespace_id: Mapped[str] = mapped_column(String(120), index=True)
    canonical_name: Mapped[str] = mapped_column(String(255), index=True)
    aliases_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    entity_type: Mapped[str] = mapped_column(String(80), default="unknown", index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    importance_prior: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class Fact(Base):
    __tablename__ = "facts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    namespace_id: Mapped[str] = mapped_column(String(120), index=True)
    subject_entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), index=True)
    predicate: Mapped[str] = mapped_column(String(120), index=True)
    object_entity_id: Mapped[int | None] = mapped_column(ForeignKey("entities.id"), nullable=True, index=True)
    object_literal: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.5, index=True)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    superseded_by_fact_id: Mapped[int | None] = mapped_column(ForeignKey("facts.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class ProvenanceLink(Base):
    __tablename__ = "provenance_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fact_id: Mapped[int] = mapped_column(ForeignKey("facts.id"), index=True)
    episode_id: Mapped[int] = mapped_column(ForeignKey("episodes.id"), index=True)


class SummaryNode(Base):
    __tablename__ = "summary_nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    namespace_id: Mapped[str] = mapped_column(String(120), index=True)
    summary_type: Mapped[str] = mapped_column(String(60), index=True)
    title: Mapped[str] = mapped_column(String(255), index=True)
    summary: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    importance_prior: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class GraphEdge(Base):
    __tablename__ = "graph_edges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    namespace_id: Mapped[str] = mapped_column(String(120), index=True)
    source_node_type: Mapped[str] = mapped_column(String(40), index=True)
    source_node_id: Mapped[int] = mapped_column(Integer, index=True)
    target_node_type: Mapped[str] = mapped_column(String(40), index=True)
    target_node_id: Mapped[int] = mapped_column(Integer, index=True)
    edge_type: Mapped[str] = mapped_column(String(80), index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class NodeDescriptor(Base):
    __tablename__ = "node_descriptors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    namespace_id: Mapped[str] = mapped_column(String(120), index=True)
    node_type: Mapped[str] = mapped_column(String(40), index=True)
    node_id: Mapped[int] = mapped_column(Integer, index=True)
    node_class: Mapped[str] = mapped_column(String(60), index=True)
    facets_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class EdgeDescriptor(Base):
    __tablename__ = "edge_descriptors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    edge_id: Mapped[int] = mapped_column(ForeignKey("graph_edges.id"), index=True)
    relation_class: Mapped[str] = mapped_column(String(60), index=True)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class ChunkEvidenceLink(Base):
    __tablename__ = "chunk_evidence_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    namespace_id: Mapped[str] = mapped_column(String(120), index=True)
    target_node_type: Mapped[str] = mapped_column(String(40), index=True)
    target_node_id: Mapped[int] = mapped_column(Integer, index=True)
    episode_chunk_id: Mapped[int] = mapped_column(ForeignKey("episode_chunks.id"), index=True)
    evidence_role: Mapped[str] = mapped_column(String(60), default="supports", index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.6)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ActivationState(Base):
    __tablename__ = "activation_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(120), index=True)
    step_index: Mapped[int] = mapped_column(Integer, index=True)
    node_type: Mapped[str] = mapped_column(String(40), index=True)
    node_id: Mapped[int] = mapped_column(Integer, index=True)
    activation_value: Mapped[float] = mapped_column(Float, default=0.0)
    direct_score: Mapped[float] = mapped_column(Float, default=0.0)
    spread_score: Mapped[float] = mapped_column(Float, default=0.0)
    recency_score: Mapped[float] = mapped_column(Float, default=0.0)
    importance_score: Mapped[float] = mapped_column(Float, default=0.0)
    inhibition_penalty: Mapped[float] = mapped_column(Float, default=0.0)
    surfaced: Mapped[bool] = mapped_column(default=False)
    reason_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class WorkingMemoryItem(Base):
    __tablename__ = "working_memory_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(120), index=True)
    step_index: Mapped[int] = mapped_column(Integer, index=True)
    node_type: Mapped[str] = mapped_column(String(40), index=True)
    node_id: Mapped[int] = mapped_column(Integer, index=True)
    content_type: Mapped[str] = mapped_column(String(40), index=True)
    content: Mapped[str] = mapped_column(Text)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    source_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), default="Untitled run")
    status: Mapped[str] = mapped_column(String(40), index=True, default="active")
    provider_type: Mapped[str] = mapped_column(String(60), index=True, default="replay")
    model_name: Mapped[str] = mapped_column(String(120), default="replay")
    api_base_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    api_key_env: Mapped[str | None] = mapped_column(String(120), nullable=True)
    namespace_id: Mapped[str] = mapped_column(String(120), index=True)
    user_id: Mapped[str | None] = mapped_column(String(120), index=True, nullable=True)
    agent_id: Mapped[str | None] = mapped_column(String(120), index=True, nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(120), index=True, nullable=True)
    memory_run_id: Mapped[str] = mapped_column(String(120), index=True, unique=True)
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, index=True)


class ExperimentTurn(Base):
    __tablename__ = "experiment_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_run_id: Mapped[str] = mapped_column(ForeignKey("experiment_runs.id"), index=True)
    turn_index: Mapped[int] = mapped_column(Integer, index=True)
    status: Mapped[str] = mapped_column(String(40), index=True, default="ok")
    user_message: Mapped[str] = mapped_column(Text)
    assistant_message: Mapped[str] = mapped_column(Text, default="")
    prompt_addition: Mapped[str] = mapped_column(Text, default="")
    request_messages_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    provider_payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    usage_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    user_step_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    assistant_step_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, index=True)


Index("ix_entity_namespace_name", Entity.namespace_id, Entity.canonical_name)
Index("ix_fact_namespace_predicate", Fact.namespace_id, Fact.predicate)
Index("ix_episode_chunk_episode_index", EpisodeChunk.episode_id, EpisodeChunk.chunk_index, unique=True)
Index("ix_node_descriptor_lookup", NodeDescriptor.namespace_id, NodeDescriptor.node_type, NodeDescriptor.node_id, unique=True)
Index("ix_edge_descriptor_lookup", EdgeDescriptor.edge_id, unique=True)
Index(
    "ix_chunk_evidence_lookup",
    ChunkEvidenceLink.namespace_id,
    ChunkEvidenceLink.target_node_type,
    ChunkEvidenceLink.target_node_id,
    ChunkEvidenceLink.episode_chunk_id,
    unique=True,
)
Index(
    "ix_graph_edges_lookup",
    GraphEdge.namespace_id,
    GraphEdge.source_node_type,
    GraphEdge.source_node_id,
    GraphEdge.target_node_type,
    GraphEdge.target_node_id,
    unique=True,
)
Index("ix_experiment_turn_unique", ExperimentTurn.experiment_run_id, ExperimentTurn.turn_index, unique=True)
