from __future__ import annotations

from collections import Counter
from datetime import datetime

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from tame.config import EngineConfig
from tame.entity_resolution import EntityResolver
from tame.extraction import build_summary_titles, extract_entities, extract_fact_candidates
from tame.models import Entity, Episode, Fact, GraphEdge, NodeType, ProvenanceLink, SummaryNode, SummaryType
from tame.utils import TextEmbedder, summarise_text, utc_now


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

            for fact_candidate in extract_fact_candidates(episode):
                fact = self._upsert_fact(session, namespace_id, fact_candidate, episode.created_at)
                if fact.created_at == fact.updated_at:
                    facts_created += 1
                self._ensure_provenance(session, fact.id, episode.id)
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

            episode.metadata_json = {**episode.metadata_json, "consolidated_at": utc_now().isoformat()}
            session.add(episode)
            processed += 1

        summary_counts = self._refresh_summary_nodes(session, namespace_id)
        summaries_updated += summary_counts
        self._link_summary_edges(session, namespace_id)
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
    ) -> None:
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


def extract_entities_for_subject(name: str):
    from tame.extraction import EntityCandidate

    entity_type = "person" if name.startswith("user:") else "topic"
    if name.startswith("agent:"):
        entity_type = "agent"
    if name.startswith("episode:"):
        entity_type = "episode_anchor"
    return EntityCandidate(name=name, entity_type=entity_type, summary=summarise_text(name))
