from __future__ import annotations

from collections import Counter
from datetime import datetime

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from memoria.config import EngineConfig
from memoria.entity_resolution import EntityResolver
from memoria.extraction import build_summary_titles, extract_entities, extract_fact_candidates
from memoria.models import (
    ChunkEvidenceLink,
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
    SummaryType,
)
from memoria.utils import TextEmbedder, summarise_text, utc_now


class ConsolidationService:
    def __init__(self, embedder: TextEmbedder, resolver: EntityResolver, config: EngineConfig):
        self.embedder = embedder
        self.resolver = resolver
        self.config = config

    def consolidate(self, session: Session, namespace_id: str, force: bool = False) -> dict[str, int]:
        episodes = session.scalars(
            select(Episode).where(Episode.namespace_id == namespace_id).order_by(Episode.created_at)
        ).all()
        processed = 0
        facts_created = 0
        entities_created = 0
        summaries_updated = 0

        for episode in episodes:
            if not force and episode.metadata_json.get("consolidated_at"):
                continue

            known_entity_ids: list[int] = []
            for candidate in extract_entities(episode):
                before = session.scalar(select(Entity).where(
                    Entity.namespace_id == namespace_id,
                    Entity.canonical_name == candidate.name,
                ))
                entity = self.resolver.resolve_or_create(session, namespace_id, candidate)
                known_entity_ids.append(entity.id)
                if before is None:
                    entities_created += 1
                self._upsert_node_descriptor(
                    session,
                    namespace_id,
                    NodeType.ENTITY.value,
                    entity.id,
                    _node_class_for_entity(entity),
                    {"entity_type": entity.entity_type, "aliases": entity.aliases_json},
                    evidence_count=1,
                    confidence=0.65,
                )

            for fact_candidate in extract_fact_candidates(episode):
                fact = self._upsert_fact(session, namespace_id, fact_candidate, episode.created_at)
                if fact.created_at == fact.updated_at:
                    facts_created += 1
                self._ensure_provenance(session, fact.id, episode.id)
                evidence_chunks = self._evidence_chunks_for_fact(session, episode, fact.summary)
                for chunk in evidence_chunks:
                    self._ensure_chunk_evidence(
                        session,
                        namespace_id,
                        NodeType.FACT.value,
                        fact.id,
                        chunk.id,
                        evidence_role="supports",
                        confidence=max(0.55, fact.confidence),
                    )
                self._upsert_node_descriptor(
                    session,
                    namespace_id,
                    NodeType.FACT.value,
                    fact.id,
                    _node_class_for_fact(fact.predicate, fact.summary),
                    {"predicate": fact.predicate, "valid_to": fact.valid_to.isoformat() if fact.valid_to else None},
                    evidence_count=max(1, len(evidence_chunks)),
                    confidence=fact.confidence,
                )
                self._ensure_edge(
                    session,
                    namespace_id,
                    NodeType.FACT.value,
                    fact.id,
                    NodeType.EPISODE.value,
                    episode.id,
                    "provenance",
                    1.0,
                )
                self._ensure_edge(
                    session,
                    namespace_id,
                    NodeType.ENTITY.value,
                    fact.subject_entity_id,
                    NodeType.FACT.value,
                    fact.id,
                    "subject_of",
                    0.8,
                )
                if fact.object_entity_id:
                    self._ensure_edge(
                        session,
                        namespace_id,
                        NodeType.FACT.value,
                        fact.id,
                        NodeType.ENTITY.value,
                        fact.object_entity_id,
                        "object_of",
                        0.8,
                    )
                for entity_id in known_entity_ids:
                    self._ensure_edge(
                        session,
                        namespace_id,
                        NodeType.EPISODE.value,
                        episode.id,
                        NodeType.ENTITY.value,
                        entity_id,
                        "mentions",
                        0.5,
                    )

            for chunk in session.scalars(select(EpisodeChunk).where(EpisodeChunk.episode_id == episode.id)).all():
                self._upsert_node_descriptor(
                    session,
                    namespace_id,
                    NodeType.EPISODE_CHUNK.value,
                    chunk.id,
                    _node_class_for_chunk(chunk.chunk_type),
                    {"chunk_type": chunk.chunk_type, "episode_id": chunk.episode_id},
                    evidence_count=1,
                    confidence=0.6 + min(chunk.salience_seed, 0.3),
                )

            episode.metadata_json = {**episode.metadata_json, "consolidated_at": utc_now().isoformat()}
            session.add(episode)
            self._upsert_node_descriptor(
                session,
                namespace_id,
                NodeType.EPISODE.value,
                episode.id,
                _node_class_for_episode(episode),
                {"source_type": episode.source_type, "role": episode.role},
                evidence_count=1,
                confidence=0.55,
            )
            processed += 1

        summary_counts = self._refresh_summary_nodes(session, namespace_id)
        summaries_updated += summary_counts
        self._link_summary_edges(session, namespace_id)
        self._refresh_descriptor_evidence_counts(session, namespace_id)
        return {
            "episodes_processed": processed,
            "facts_created_or_updated": facts_created,
            "entities_created": entities_created,
            "summary_nodes_updated": summaries_updated,
        }

    def _upsert_fact(self, session: Session, namespace_id: str, fact_candidate, valid_from: datetime | None) -> Fact:
        subject_entity = self.resolver.resolve_or_create(
            session,
            namespace_id,
            extract_entities_for_subject(fact_candidate.subject_name),
        )
        object_entity_id = None
        if fact_candidate.object_name:
            object_entity = self.resolver.resolve_or_create(
                session,
                namespace_id,
                extract_entities_for_subject(fact_candidate.object_name),
            )
            object_entity_id = object_entity.id

        existing = session.scalars(
            select(Fact).where(
                Fact.namespace_id == namespace_id,
                Fact.subject_entity_id == subject_entity.id,
                Fact.predicate == fact_candidate.predicate,
                Fact.object_entity_id == object_entity_id,
                Fact.object_literal == fact_candidate.object_literal,
                Fact.valid_to.is_(None),
            )
        ).all()
        if existing:
            fact = existing[0]
            fact.confidence = min(1.0, fact.confidence + 0.05)
            fact.updated_at = utc_now()
            session.add(fact)
            session.flush()
            return fact

        if fact_candidate.conflict_key:
            conflict_facts = session.scalars(
                select(Fact).where(
                    Fact.namespace_id == namespace_id,
                    Fact.subject_entity_id == subject_entity.id,
                    Fact.predicate == fact_candidate.predicate,
                    Fact.valid_to.is_(None),
                )
            ).all()
            for conflict in conflict_facts:
                conflict.valid_to = valid_from or utc_now()
                session.add(conflict)

        fact = Fact(
            namespace_id=namespace_id,
            subject_entity_id=subject_entity.id,
            predicate=fact_candidate.predicate,
            object_entity_id=object_entity_id,
            object_literal=fact_candidate.object_literal,
            summary=fact_candidate.summary,
            embedding=self.embedder.embed(fact_candidate.summary),
            confidence=fact_candidate.confidence,
            valid_from=valid_from,
        )
        session.add(fact)
        session.flush()

        if fact_candidate.conflict_key:
            conflict_facts = session.scalars(
                select(Fact).where(
                    Fact.namespace_id == namespace_id,
                    Fact.subject_entity_id == subject_entity.id,
                    Fact.predicate == fact_candidate.predicate,
                    Fact.id != fact.id,
                    Fact.valid_to == fact.valid_from,
                )
            ).all()
            for conflict in conflict_facts:
                conflict.superseded_by_fact_id = fact.id
                session.add(conflict)

        return fact

    def _ensure_provenance(self, session: Session, fact_id: int, episode_id: int) -> None:
        link = session.scalar(
            select(ProvenanceLink).where(
                ProvenanceLink.fact_id == fact_id,
                ProvenanceLink.episode_id == episode_id,
            )
        )
        if link is None:
            session.add(ProvenanceLink(fact_id=fact_id, episode_id=episode_id))
            session.flush()

    def _refresh_summary_nodes(self, session: Session, namespace_id: str) -> int:
        count = 0
        episodes = session.scalars(
            select(Episode).where(Episode.namespace_id == namespace_id).order_by(desc(Episode.created_at))
        ).all()
        facts = session.scalars(select(Fact).where(Fact.namespace_id == namespace_id, Fact.valid_to.is_(None))).all()
        entities = session.scalars(select(Entity).where(Entity.namespace_id == namespace_id)).all()
        if not episodes:
            return 0

        title_map = build_summary_titles(namespace_id, episodes[0].user_id, episodes[0].session_id)

        session_title = title_map.get("session_rollup")
        if session_title:
            session_summary = summarise_text(" ".join((episode.content_summary or "") for episode in episodes[:6]), 48)
            self._upsert_summary_node(
                session,
                namespace_id,
                SummaryType.SESSION_ROLLUP.value,
                session_title,
                session_summary,
                importance_prior=0.7,
            )
            count += 1

        user_title = title_map.get("user_profile")
        if user_title:
            user_fact_summaries = [fact.summary for fact in facts if any(token in fact.summary for token in ["user:", "identity", "prefers", "dislikes", "working on"])]
            user_summary = summarise_text(" ".join(user_fact_summaries[:6]) or "No user profile facts yet.", 48)
            self._upsert_summary_node(
                session,
                namespace_id,
                SummaryType.USER_PROFILE.value,
                user_title,
                user_summary,
                importance_prior=0.8,
            )
            count += 1

        top_entities = ", ".join(entity.canonical_name for entity in entities[:5]) or "No salient entities"
        self._upsert_summary_node(
            session,
            namespace_id,
            SummaryType.PROJECT.value,
            title_map["project"],
            summarise_text(f"Current graph centers around {top_entities}.", 36),
            importance_prior=0.5,
        )
        count += 1

        keyword_counter = Counter()
        for episode in episodes[:8]:
            for token in (episode.content_summary or episode.content_raw).split():
                cleaned = token.strip(".,!?").lower()
                if len(cleaned) > 3:
                    keyword_counter[cleaned] += 1
        common_keywords = ", ".join(keyword for keyword, _ in keyword_counter.most_common(6))
        self._upsert_summary_node(
            session,
            namespace_id,
            SummaryType.TOPIC.value,
            title_map["topic"],
            f"Frequently discussed terms: {common_keywords}",
            importance_prior=0.45,
        )
        count += 1
        return count

    def _upsert_summary_node(
        self,
        session: Session,
        namespace_id: str,
        summary_type: str,
        title: str,
        summary: str,
        importance_prior: float,
    ) -> SummaryNode:
        node = session.scalar(
            select(SummaryNode).where(
                SummaryNode.namespace_id == namespace_id,
                SummaryNode.summary_type == summary_type,
                SummaryNode.title == title,
            )
        )
        if node is None:
            node = SummaryNode(
                namespace_id=namespace_id,
                summary_type=summary_type,
                title=title,
                summary=summary,
                embedding=self.embedder.embed(f"{title}. {summary}"),
                importance_prior=importance_prior,
            )
        else:
            node.summary = summary
            node.embedding = self.embedder.embed(f"{title}. {summary}")
            node.importance_prior = importance_prior
        session.add(node)
        session.flush()
        self._upsert_node_descriptor(
            session,
            namespace_id,
            NodeType.SUMMARY.value,
            node.id,
            _node_class_for_summary(summary_type),
            {"summary_type": summary_type, "title": title},
            evidence_count=0,
            confidence=0.7,
        )
        return node

    def _link_summary_edges(self, session: Session, namespace_id: str) -> None:
        summaries = session.scalars(select(SummaryNode).where(SummaryNode.namespace_id == namespace_id)).all()
        facts = session.scalars(select(Fact).where(Fact.namespace_id == namespace_id, Fact.valid_to.is_(None))).all()
        entities = session.scalars(select(Entity).where(Entity.namespace_id == namespace_id)).all()
        for summary in summaries:
            for fact in facts[:6]:
                if any(token in fact.summary.lower() for token in summary.summary.lower().split()[:5]):
                    self._ensure_edge(
                        session,
                        namespace_id,
                        NodeType.SUMMARY.value,
                        summary.id,
                        NodeType.FACT.value,
                        fact.id,
                        "summarises",
                        0.7,
                    )
            for entity in entities[:6]:
                if entity.canonical_name.lower() in summary.summary.lower():
                    self._ensure_edge(
                        session,
                        namespace_id,
                        NodeType.SUMMARY.value,
                        summary.id,
                        NodeType.ENTITY.value,
                        entity.id,
                        "references",
                        0.6,
                    )

    def _ensure_edge(
        self,
        session: Session,
        namespace_id: str,
        source_type: str,
        source_id: int,
        target_type: str,
        target_id: int,
        edge_type: str,
        weight: float,
    ) -> GraphEdge:
        edge = session.scalar(
            select(GraphEdge).where(
                GraphEdge.namespace_id == namespace_id,
                GraphEdge.source_node_type == source_type,
                GraphEdge.source_node_id == source_id,
                GraphEdge.target_node_type == target_type,
                GraphEdge.target_node_id == target_id,
            )
        )
        if edge is None:
            edge = GraphEdge(
                namespace_id=namespace_id,
                source_node_type=source_type,
                source_node_id=source_id,
                target_node_type=target_type,
                target_node_id=target_id,
                edge_type=edge_type,
                weight=weight,
            )
        else:
            edge.edge_type = edge_type
            edge.weight = weight
        session.add(edge)
        session.flush()
        self._upsert_edge_descriptor(
            session,
            edge,
            relation_class=_relation_class_for_edge(edge_type),
            confidence=max(0.35, min(1.0, weight)),
            evidence_count=1 if edge_type in {"provenance", "mentions", "summarises"} else 0,
        )
        return edge

    def _evidence_chunks_for_fact(self, session: Session, episode: Episode, fact_summary: str) -> list[EpisodeChunk]:
        chunks = session.scalars(
            select(EpisodeChunk).where(EpisodeChunk.episode_id == episode.id).order_by(EpisodeChunk.chunk_index)
        ).all()
        if not chunks:
            return []
        fact_tokens = {token.strip(".,!?").lower() for token in fact_summary.split() if len(token) > 3}
        ranked = []
        for chunk in chunks:
            chunk_tokens = {token.strip(".,!?").lower() for token in chunk.text.split() if len(token) > 3}
            ranked.append((len(fact_tokens & chunk_tokens), chunk))
        ranked.sort(key=lambda item: (-item[0], item[1].chunk_index))
        selected = [chunk for overlap, chunk in ranked if overlap > 0][:2]
        return selected or [chunks[0]]

    def _ensure_chunk_evidence(
        self,
        session: Session,
        namespace_id: str,
        target_node_type: str,
        target_node_id: int,
        episode_chunk_id: int,
        evidence_role: str,
        confidence: float,
    ) -> None:
        link = session.scalar(
            select(ChunkEvidenceLink).where(
                ChunkEvidenceLink.namespace_id == namespace_id,
                ChunkEvidenceLink.target_node_type == target_node_type,
                ChunkEvidenceLink.target_node_id == target_node_id,
                ChunkEvidenceLink.episode_chunk_id == episode_chunk_id,
            )
        )
        if link is None:
            link = ChunkEvidenceLink(
                namespace_id=namespace_id,
                target_node_type=target_node_type,
                target_node_id=target_node_id,
                episode_chunk_id=episode_chunk_id,
                evidence_role=evidence_role,
                confidence=confidence,
            )
        else:
            link.evidence_role = evidence_role
            link.confidence = max(link.confidence, confidence)
        session.add(link)
        session.flush()

    def _upsert_node_descriptor(
        self,
        session: Session,
        namespace_id: str,
        node_type: str,
        node_id: int,
        node_class: str,
        facets: dict,
        evidence_count: int,
        confidence: float,
    ) -> NodeDescriptor:
        descriptor = session.scalar(
            select(NodeDescriptor).where(
                NodeDescriptor.namespace_id == namespace_id,
                NodeDescriptor.node_type == node_type,
                NodeDescriptor.node_id == node_id,
            )
        )
        if descriptor is None:
            descriptor = NodeDescriptor(
                namespace_id=namespace_id,
                node_type=node_type,
                node_id=node_id,
                node_class=node_class,
                facets_json=facets,
                evidence_count=evidence_count,
                confidence=confidence,
            )
        else:
            descriptor.node_class = node_class
            descriptor.facets_json = {**dict(descriptor.facets_json or {}), **facets}
            descriptor.evidence_count = max(descriptor.evidence_count, evidence_count)
            descriptor.confidence = max(descriptor.confidence, confidence)
            descriptor.updated_at = utc_now()
        session.add(descriptor)
        session.flush()
        return descriptor

    def _upsert_edge_descriptor(
        self,
        session: Session,
        edge: GraphEdge,
        relation_class: str,
        confidence: float,
        evidence_count: int,
    ) -> EdgeDescriptor:
        descriptor = session.scalar(select(EdgeDescriptor).where(EdgeDescriptor.edge_id == edge.id))
        if descriptor is None:
            descriptor = EdgeDescriptor(
                edge_id=edge.id,
                relation_class=relation_class,
                confidence=confidence,
                evidence_count=evidence_count,
                metadata_json={"edge_type": edge.edge_type},
            )
        else:
            descriptor.relation_class = relation_class
            descriptor.confidence = max(descriptor.confidence, confidence)
            descriptor.evidence_count = max(descriptor.evidence_count, evidence_count)
            descriptor.metadata_json = {**dict(descriptor.metadata_json or {}), "edge_type": edge.edge_type}
            descriptor.updated_at = utc_now()
        session.add(descriptor)
        session.flush()
        return descriptor

    def _refresh_descriptor_evidence_counts(self, session: Session, namespace_id: str) -> None:
        descriptors = session.scalars(select(NodeDescriptor).where(NodeDescriptor.namespace_id == namespace_id)).all()
        for descriptor in descriptors:
            count = session.scalar(
                select(func.count(ChunkEvidenceLink.id)).where(
                    ChunkEvidenceLink.namespace_id == namespace_id,
                    ChunkEvidenceLink.target_node_type == descriptor.node_type,
                    ChunkEvidenceLink.target_node_id == descriptor.node_id,
                )
            )
            if count:
                descriptor.evidence_count = max(descriptor.evidence_count, int(count))
                session.add(descriptor)
        session.flush()


def extract_entities_for_subject(name: str):
    from memoria.extraction import EntityCandidate

    entity_type = "person" if name.startswith("user:") else "topic"
    if name.startswith("agent:"):
        entity_type = "agent"
    if name.startswith("episode:"):
        entity_type = "episode_anchor"
    return EntityCandidate(name=name, entity_type=entity_type, summary=summarise_text(name))


def _node_class_for_entity(entity: Entity) -> str:
    if entity.entity_type in {"person", "agent", "episode_anchor"}:
        return "identity"
    if entity.entity_type in {"resource", "organization"}:
        return "resource"
    return "claim"


def _node_class_for_fact(predicate: str, summary: str) -> str:
    lowered = f"{predicate} {summary}".lower()
    if predicate.startswith("identity"):
        return "identity"
    if predicate in {"prefers", "dislikes"}:
        return "preference"
    if predicate in {"working_on", "researching"}:
        return "task_state"
    if any(token in lowered for token in ("must", "only", "cannot", "constraint")):
        return "constraint"
    if predicate == "tool_result":
        return "event"
    return "claim"


def _node_class_for_episode(episode: Episode) -> str:
    if episode.source_type == "tool_result":
        return "event"
    if episode.source_type == "imported_doc":
        return "resource"
    return "claim"


def _node_class_for_chunk(chunk_type: str) -> str:
    mapping = {
        "preference": "preference",
        "constraint": "constraint",
        "decision": "task_state",
        "task_state": "task_state",
        "tool_result": "event",
        "result": "summary",
        "reasoning_note": "procedure",
    }
    return mapping.get(chunk_type, "claim")


def _node_class_for_summary(summary_type: str) -> str:
    if summary_type in {SummaryType.PROCEDURE.value, SummaryType.COMMUNITY.value}:
        return "procedure"
    if summary_type in {SummaryType.PROJECT.value, SummaryType.SESSION_ROLLUP.value}:
        return "task_state"
    if summary_type == SummaryType.USER_PROFILE.value:
        return "identity"
    return "summary"


def _relation_class_for_edge(edge_type: str) -> str:
    mapping = {
        "provenance": "supports",
        "subject_of": "mentions",
        "object_of": "mentions",
        "mentions": "mentions",
        "summarises": "summarises",
        "references": "same_topic",
        "supersedes": "updates",
    }
    return mapping.get(edge_type, "same_topic")
