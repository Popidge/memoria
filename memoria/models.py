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


class WorkingMemoryContentType(str, Enum):
    ENTITY_SUMMARY = "entity_summary"
    FACT_SUMMARY = "fact_summary"
    SUMMARY_NODE = "summary_node"
    EPISODE_SNIPPET = "episode_snippet"


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


Index("ix_entity_namespace_name", Entity.namespace_id, Entity.canonical_name)
Index("ix_fact_namespace_predicate", Fact.namespace_id, Fact.predicate)
Index(
    "ix_graph_edges_lookup",
    GraphEdge.namespace_id,
    GraphEdge.source_node_type,
    GraphEdge.source_node_id,
    GraphEdge.target_node_type,
    GraphEdge.target_node_id,
    unique=True,
)
