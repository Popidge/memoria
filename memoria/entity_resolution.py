from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from memoria.config import EngineConfig
from memoria.extraction import EntityCandidate
from memoria.models import Entity
from memoria.utils import TextEmbedder, cosine_similarity, normalise_name, summarise_text


class EntityResolver:
    def __init__(self, embedder: TextEmbedder, config: EngineConfig):
        self.embedder = embedder
        self.config = config

    def resolve_or_create(self, session: Session, namespace_id: str, candidate: EntityCandidate) -> Entity:
        lowered = normalise_name(candidate.name)
        namespace_entities = session.scalars(select(Entity).where(Entity.namespace_id == namespace_id)).all()
        entities = [
            entity
            for entity in namespace_entities
            if normalise_name(entity.canonical_name) == lowered
            or lowered in {normalise_name(alias) for alias in entity.aliases_json}
        ]
        for entity in entities:
            if normalise_name(entity.canonical_name) == lowered:
                return entity
            aliases = {normalise_name(alias) for alias in entity.aliases_json}
            if lowered in aliases:
                return entity

        desired_embedding = self.embedder.embed(candidate.name)
        for entity in namespace_entities:
            similarity = cosine_similarity(desired_embedding, entity.embedding)
            if similarity >= self.config.entity_similarity_threshold:
                if candidate.name not in entity.aliases_json and candidate.name != entity.canonical_name:
                    entity.aliases_json = sorted(set(entity.aliases_json + [candidate.name]))
                    session.add(entity)
                    session.flush()
                return entity

        entity = Entity(
            namespace_id=namespace_id,
            canonical_name=candidate.name,
            aliases_json=sorted(set(candidate.aliases or [])),
            entity_type=candidate.entity_type,
            summary=candidate.summary or summarise_text(candidate.name),
            embedding=desired_embedding,
            importance_prior=candidate.importance_prior,
        )
        session.add(entity)
        session.flush()
        return entity

    def merge_duplicates(self, session: Session, namespace_id: str) -> int:
        entities = session.scalars(select(Entity).where(Entity.namespace_id == namespace_id)).all()
        merged = 0
        survivors: dict[int, Entity] = {}
        for entity in entities:
            if entity.id in survivors:
                continue
            survivors[entity.id] = entity
            for other in entities:
                if other.id <= entity.id or other.id in survivors:
                    continue
                similarity = cosine_similarity(entity.embedding, other.embedding)
                if similarity >= self.config.entity_similarity_threshold:
                    entity.aliases_json = sorted(set(entity.aliases_json + [other.canonical_name] + other.aliases_json))
                    merged += 1
                    session.delete(other)
        return merged
