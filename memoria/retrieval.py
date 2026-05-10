from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from memoria.config import EngineConfig
from memoria.models import (
    ActivationState,
    ChunkEvidenceLink,
    Entity,
    Episode,
    EpisodeChunk,
    Fact,
    GraphEdge,
    NodeKey,
    NodeDescriptor,
    NodeType,
    ProvenanceLink,
    SummaryNode,
)
from memoria.utils import TextEmbedder, build_text_signal, cosine_similarity, keyword_overlap, recency_decay


@dataclass
class RetrievedNode:
    node_key: NodeKey
    text: str
    semantic_score: float
    keyword_score: float
    scope_score: float
    confidence: float
    metadata: dict[str, Any]

    @property
    def direct_score(self) -> float:
        return min(1.0, self.semantic_score * 0.7 + self.keyword_score * 0.3 + self.scope_score)


class RetrievalService:
    def __init__(self, embedder: TextEmbedder, config: EngineConfig):
        self.embedder = embedder
        self.config = config

    def search(
        self,
        session: Session,
        text: str,
        namespace_id: str,
        filters: dict[str, Any] | None = None,
        limit: int = 8,
    ) -> list[RetrievedNode]:
        signal = build_text_signal(text, self.embedder)
        candidates: dict[str, RetrievedNode] = {}
        filters = filters or {}

        for model, node_type in (
            (SummaryNode, NodeType.SUMMARY),
            (Fact, NodeType.FACT),
            (Entity, NodeType.ENTITY),
            (EpisodeChunk, NodeType.EPISODE_CHUNK),
            (Episode, NodeType.EPISODE),
        ):
            for row in self._query_model(session, model, node_type, namespace_id, signal, filters):
                key = row.node_key.as_string()
                current = candidates.get(key)
                if current is None or row.direct_score > current.direct_score:
                    candidates[key] = row

        ranked = sorted(
            candidates.values(),
            key=lambda item: (item.direct_score, item.confidence, item.metadata.get("recency", 0.0)),
            reverse=True,
        )
        return ranked[:limit]

    def neighbors_for_hot_nodes(
        self,
        session: Session,
        namespace_id: str,
        run_id: str,
        previous_step: int,
        limit: int,
    ) -> list[NodeKey]:
        if previous_step < 0:
            return []
        hot_states = session.scalars(
            select(ActivationState)
            .where(ActivationState.run_id == run_id, ActivationState.step_index == previous_step)
            .order_by(desc(ActivationState.activation_value))
        ).all()
        primary_threshold = self.config.thresholds.summary
        hot_keys = {
            (state.node_type, state.node_id)
            for state in hot_states
            if state.activation_value >= primary_threshold
        }
        hot_keys = set(list(hot_keys)[: limit or 8])
        neighbors: list[NodeKey] = []
        for edge in session.scalars(select(GraphEdge).where(GraphEdge.namespace_id == namespace_id)).all():
            if (edge.source_node_type, edge.source_node_id) in hot_keys:
                neighbors.append(NodeKey(NodeType(edge.target_node_type), edge.target_node_id))
            if (edge.target_node_type, edge.target_node_id) in hot_keys:
                neighbors.append(NodeKey(NodeType(edge.source_node_type), edge.source_node_id))
        deduped: dict[str, NodeKey] = {neighbor.as_string(): neighbor for neighbor in neighbors}
        return list(deduped.values())[:limit]

    def fetch_node_text(self, session: Session, node_key: NodeKey) -> str:
        if node_key.node_type == NodeType.ENTITY:
            entity = session.get(Entity, node_key.node_id)
            return f"{entity.canonical_name}. {entity.summary}" if entity else ""
        if node_key.node_type == NodeType.FACT:
            fact = session.get(Fact, node_key.node_id)
            return fact.summary if fact else ""
        if node_key.node_type == NodeType.SUMMARY:
            summary = session.get(SummaryNode, node_key.node_id)
            return f"{summary.title}. {summary.summary}" if summary else ""
        if node_key.node_type == NodeType.EPISODE_CHUNK:
            chunk = session.get(EpisodeChunk, node_key.node_id)
            return chunk.summary or chunk.text if chunk else ""
        episode = session.get(Episode, node_key.node_id)
        return episode.content_summary or episode.content_raw if episode else ""

    def _query_model(self, session: Session, model, node_type: NodeType, namespace_id: str, signal, filters):
        rows = session.scalars(select(model).where(model.namespace_id == namespace_id)).all()
        results: list[RetrievedNode] = []
        for row in rows:
            text = self._row_text(row, node_type)
            semantic = cosine_similarity(signal.embedding, getattr(row, "embedding", None))
            keyword = keyword_overlap(signal.text, text)
            scope_score = self._scope_score(session, row, node_type, filters)
            entity_bonus = self._entity_bonus(signal.linked_entities, text)
            scope_score += entity_bonus
            confidence = getattr(row, "confidence", 1.0)
            recency = recency_decay(getattr(row, "updated_at", None) or getattr(row, "created_at", None))
            if max(semantic, keyword, scope_score) < self.config.retrieval.confidence_threshold:
                continue
            results.append(
                RetrievedNode(
                    node_key=NodeKey(node_type, row.id),
                    text=text,
                    semantic_score=semantic,
                    keyword_score=keyword,
                    scope_score=scope_score,
                    confidence=confidence,
                    metadata={"recency": recency},
                )
            )
        return results

    def _row_text(self, row, node_type: NodeType) -> str:
        if node_type == NodeType.ENTITY:
            return f"{row.canonical_name}. {row.summary}"
        if node_type == NodeType.FACT:
            return row.summary
        if node_type == NodeType.SUMMARY:
            return f"{row.title}. {row.summary}"
        if node_type == NodeType.EPISODE_CHUNK:
            return row.summary or row.text
        return row.content_summary or row.content_raw

    def _scope_score(self, session: Session, row, node_type: NodeType, filters: dict[str, Any]) -> float:
        if not filters:
            return 0.0
        matches = 0.0
        if node_type == NodeType.EPISODE:
            for field in ("user_id", "agent_id", "session_id"):
                if filters.get(field) and getattr(row, field, None) == filters[field]:
                    matches += self.config.scope_match_bonus
        if node_type == NodeType.EPISODE_CHUNK:
            # Chunk rows keep source scope on their parent episode; loading it keeps the
            # scope behavior consistent with episode retrieval without denormalizing ids.
            parent = session.get(Episode, row.episode_id)
            if parent is not None:
                for field in ("user_id", "agent_id", "session_id"):
                    if filters.get(field) and getattr(parent, field, None) == filters[field]:
                        matches += self.config.scope_match_bonus
            matches += min(0.12, getattr(row, "salience_seed", 0.0) * 0.15)
        return matches

    def _entity_bonus(self, linked_entities: list[str], text: str) -> float:
        lowered_text = text.lower()
        for entity in linked_entities:
            if entity.lower() in lowered_text:
                return 0.2
        return 0.0

    def provenance_episode_ids(self, session: Session, fact_id: int) -> list[int]:
        links = session.scalars(select(ProvenanceLink).where(ProvenanceLink.fact_id == fact_id)).all()
        return [link.episode_id for link in links]

    def evidence_chunk_ids(self, session: Session, node_type: str, node_id: int) -> list[int]:
        links = session.scalars(
            select(ChunkEvidenceLink).where(
                ChunkEvidenceLink.target_node_type == node_type,
                ChunkEvidenceLink.target_node_id == node_id,
            )
        ).all()
        return [link.episode_chunk_id for link in links]

    def node_descriptor(self, session: Session, node_type: str, node_id: int, namespace_id: str | None = None) -> NodeDescriptor | None:
        query = select(NodeDescriptor).where(
            NodeDescriptor.node_type == node_type,
            NodeDescriptor.node_id == node_id,
        )
        if namespace_id is not None:
            query = query.where(NodeDescriptor.namespace_id == namespace_id)
        return session.scalar(query)
