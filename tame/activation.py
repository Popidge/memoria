from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from tame.config import EngineConfig
from tame.models import ActivationState, Entity, Episode, Fact, GraphEdge, NodeKey, NodeType, SummaryNode, WorkingMemoryContentType, WorkingMemoryItem
from tame.retrieval import RetrievalService
from tame.utils import build_text_signal, clip, cosine_similarity, episode_snippet, keyword_overlap, recency_decay


@dataclass
class StepResult:
    run_id: str
    step_index: int
    working_memory: list[dict[str, Any]]
    debug: list[dict[str, Any]]


class ActivationService:
    def __init__(self, retrieval: RetrievalService, config: EngineConfig):
        self.retrieval = retrieval
        self.config = config

    def process_step(
        self,
        session: Session,
        *,
        run_id: str,
        step_index: int,
        namespace_id: str,
        step_type: str,
        text: str,
        filters: dict[str, Any] | None = None,
        linked_tool_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StepResult:
        filters = filters or {}
        signal = build_text_signal(text, self.retrieval.embedder)
        search_candidates = self.retrieval.search(
            session,
            text=text,
            namespace_id=namespace_id,
            filters=filters,
            limit=self.config.top_k.semantic_candidates + self.config.top_k.keyword_candidates,
        )
        neighbor_keys = self.retrieval.neighbors_for_hot_nodes(
            session,
            namespace_id=namespace_id,
            run_id=run_id,
            previous_step=step_index - 1,
            limit=self.config.top_k.neighbor_candidates,
        )

        candidates: dict[str, dict[str, Any]] = {
            item.node_key.as_string(): {
                "node_key": item.node_key,
                "text": item.text,
                "semantic_score": item.semantic_score,
                "keyword_score": item.keyword_score,
                "scope_score": item.scope_score,
                "confidence": item.confidence,
            }
            for item in search_candidates
        }
        for node_key in neighbor_keys:
            candidates.setdefault(
                node_key.as_string(),
                {
                    "node_key": node_key,
                    "text": self.retrieval.fetch_node_text(session, node_key),
                    "semantic_score": cosine_similarity(signal.embedding, self._fetch_embedding(session, node_key)),
                    "keyword_score": keyword_overlap(text, self.retrieval.fetch_node_text(session, node_key)),
                    "scope_score": 0.0,
                    "confidence": 1.0,
                },
            )

        previous_states = {
            NodeKey(NodeType(row.node_type), row.node_id).as_string(): row
            for row in session.scalars(
                select(ActivationState).where(
                    ActivationState.run_id == run_id,
                    ActivationState.step_index == step_index - 1,
                )
            ).all()
        }
        recent_surface_keys = {
            f"{item.node_type}:{item.node_id}"
            for item in session.scalars(
                select(WorkingMemoryItem).where(
                    WorkingMemoryItem.run_id == run_id,
                    WorkingMemoryItem.step_index >= max(0, step_index - self.config.inhibition.recent_surface_steps),
                )
            ).all()
        }

        scored: list[ActivationState] = []
        for candidate in candidates.values():
            node_key: NodeKey = candidate["node_key"]
            spread_score = self._spread_score(session, namespace_id, node_key, previous_states)
            recency_score = self._recency_score(session, node_key)
            importance_score = self._importance_score(session, node_key)
            duplicate_penalty = self._duplicate_penalty(candidate["text"], scored)
            inhibition_penalty = duplicate_penalty + self._generic_penalty(session, namespace_id, node_key)
            if node_key.as_string() in recent_surface_keys:
                inhibition_penalty += self.config.inhibition.recently_surfaced_penalty
            previous_activation = previous_states.get(node_key.as_string())
            decayed = previous_activation.activation_value * self.config.decay_factor if previous_activation else 0.0
            direct_component = clip(
                candidate["semantic_score"] * 0.7
                + candidate["keyword_score"] * 0.3
                + candidate["scope_score"]
            )
            activation_value = clip(
                self.config.activation_weights.semantic * candidate["semantic_score"]
                + self.config.activation_weights.keyword * candidate["keyword_score"]
                + self.config.activation_weights.spread * spread_score
                + self.config.activation_weights.recency * recency_score
                + self.config.activation_weights.importance * importance_score
                + decayed
                - inhibition_penalty
            )
            state = ActivationState(
                run_id=run_id,
                step_index=step_index,
                node_type=node_key.node_type.value,
                node_id=node_key.node_id,
                activation_value=activation_value,
                direct_score=direct_component,
                spread_score=spread_score,
                recency_score=recency_score,
                importance_score=importance_score,
                inhibition_penalty=inhibition_penalty,
                surfaced=False,
                reason_json={
                    "step_type": step_type,
                    "keywords": signal.keywords,
                    "linked_entities": signal.linked_entities,
                    "linked_tool_name": linked_tool_name,
                    "metadata": metadata or {},
                    "text": candidate["text"],
                    "semantic_score": candidate["semantic_score"],
                    "keyword_score": candidate["keyword_score"],
                    "scope_score": candidate["scope_score"],
                    "decayed_previous_activation": decayed,
                },
            )
            session.add(state)
            scored.append(state)
        session.flush()

        promoted_items = self._promote(session, run_id, step_index, scored, signal.keywords, metadata or {})
        promoted_keys = {(item.node_type, item.node_id) for item in promoted_items}
        for state in scored:
            if (state.node_type, state.node_id) in promoted_keys:
                state.surfaced = True
                session.add(state)
        session.flush()

        working_memory = [
            {
                "node_key": f"{item.node_type}:{item.node_id}",
                "content_type": item.content_type,
                "content": item.content,
                "score": item.score,
                "source": item.source_json,
            }
            for item in promoted_items
        ]
        debug = [
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
            for state in sorted(scored, key=lambda item: item.activation_value, reverse=True)
        ]
        return StepResult(run_id=run_id, step_index=step_index, working_memory=working_memory, debug=debug)

    def _spread_score(self, session: Session, namespace_id: str, node_key: NodeKey, previous_states: dict[str, ActivationState]) -> float:
        if not previous_states:
            return 0.0
        total = 0.0
        edges = session.scalars(select(GraphEdge).where(GraphEdge.namespace_id == namespace_id)).all()
        for edge in edges:
            if edge.source_node_type == node_key.node_type.value and edge.source_node_id == node_key.node_id:
                other = f"{edge.target_node_type}:{edge.target_node_id}"
            elif edge.target_node_type == node_key.node_type.value and edge.target_node_id == node_key.node_id:
                other = f"{edge.source_node_type}:{edge.source_node_id}"
            else:
                continue
            previous = previous_states.get(other)
            if previous:
                total += previous.activation_value * edge.weight
        return clip(total)

    def _recency_score(self, session: Session, node_key: NodeKey) -> float:
        row = self._fetch_row(session, node_key)
        timestamp = getattr(row, "updated_at", None) or getattr(row, "created_at", None)
        return recency_decay(timestamp)

    def _importance_score(self, session: Session, node_key: NodeKey) -> float:
        row = self._fetch_row(session, node_key)
        return clip(getattr(row, "importance_prior", 0.0))

    def _generic_penalty(self, session: Session, namespace_id: str, node_key: NodeKey) -> float:
        degree = 0
        for edge in session.scalars(select(GraphEdge).where(GraphEdge.namespace_id == namespace_id)).all():
            if (
                edge.source_node_type == node_key.node_type.value
                and edge.source_node_id == node_key.node_id
            ) or (
                edge.target_node_type == node_key.node_type.value
                and edge.target_node_id == node_key.node_id
            ):
                degree += 1
        return min(self.config.inhibition.generic_degree_penalty, degree * 0.01)

    def _duplicate_penalty(self, text: str, states: list[ActivationState]) -> float:
        existing = [state.reason_json.get("text", "") for state in states]
        for prior_text in existing:
            if keyword_overlap(text, prior_text) >= self.config.duplicate_similarity_threshold:
                return self.config.inhibition.duplicate_summary_penalty
        return 0.0

    def _promote(
        self,
        session: Session,
        run_id: str,
        step_index: int,
        states: list[ActivationState],
        keywords: list[str],
        metadata: dict[str, Any],
    ) -> list[WorkingMemoryItem]:
        selected: list[WorkingMemoryItem] = []
        counts_by_type: defaultdict[str, int] = defaultdict(int)
        provenance_requested = bool(metadata.get("include_provenance"))
        priority = {
            NodeType.SUMMARY.value: 0,
            NodeType.FACT.value: 1,
            NodeType.ENTITY.value: 2,
            NodeType.EPISODE.value: 3,
        }
        for state in sorted(states, key=lambda item: (priority.get(item.node_type, 9), -item.activation_value)):
            if len(selected) >= self.config.top_k.working_memory_total:
                break
            if counts_by_type[state.node_type] >= self.config.top_k.per_type:
                continue
            content_type, content, source_json = self._materialize_item(session, state, keywords, provenance_requested)
            if not content_type:
                continue
            minimum = self._promotion_threshold(state.node_type)
            if state.activation_value < minimum:
                continue
            item = WorkingMemoryItem(
                run_id=run_id,
                step_index=step_index,
                node_type=state.node_type,
                node_id=state.node_id,
                content_type=content_type,
                content=content,
                score=state.activation_value,
                source_json=source_json,
            )
            session.add(item)
            selected.append(item)
            counts_by_type[state.node_type] += 1
        session.flush()
        return selected

    def _materialize_item(
        self,
        session: Session,
        state: ActivationState,
        keywords: list[str],
        provenance_requested: bool,
    ) -> tuple[str | None, str, dict[str, Any]]:
        node_key = NodeKey(NodeType(state.node_type), state.node_id)
        row = self._fetch_row(session, node_key)
        if row is None:
            return None, "", {}
        if state.node_type == NodeType.SUMMARY.value:
            content = f"{row.title}: {row.summary}"
            return WorkingMemoryContentType.SUMMARY_NODE.value, content, {"title": row.title}
        if state.node_type == NodeType.FACT.value:
            source_json = {"confidence": row.confidence}
            if provenance_requested or state.activation_value >= self.config.thresholds.fact:
                source_json["provenance_episode_ids"] = self.retrieval.provenance_episode_ids(session, row.id)
            return WorkingMemoryContentType.FACT_SUMMARY.value, row.summary, source_json
        if state.node_type == NodeType.ENTITY.value:
            content = f"{row.canonical_name}: {row.summary}"
            return WorkingMemoryContentType.ENTITY_SUMMARY.value, content, {"entity_type": row.entity_type}
        if state.activation_value >= self.config.thresholds.snippet or provenance_requested:
            snippet = episode_snippet(row.content_raw, keywords, self.config.retrieval.episode_snippet_words)
            return WorkingMemoryContentType.EPISODE_SNIPPET.value, snippet, {"episode_id": row.id}
        return None, "", {}

    def _promotion_threshold(self, node_type: str) -> float:
        if node_type == NodeType.FACT.value:
            return self.config.thresholds.fact
        if node_type == NodeType.EPISODE.value:
            return self.config.thresholds.snippet
        return self.config.thresholds.summary

    def _fetch_row(self, session: Session, node_key: NodeKey):
        if node_key.node_type == NodeType.ENTITY:
            return session.get(Entity, node_key.node_id)
        if node_key.node_type == NodeType.FACT:
            return session.get(Fact, node_key.node_id)
        if node_key.node_type == NodeType.SUMMARY:
            return session.get(SummaryNode, node_key.node_id)
        return session.get(Episode, node_key.node_id)

    def _fetch_embedding(self, session: Session, node_key: NodeKey) -> list[float] | None:
        row = self._fetch_row(session, node_key)
        return getattr(row, "embedding", None)
