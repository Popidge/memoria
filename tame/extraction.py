from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from tame.models import Episode, SourceType
from tame.utils import capitalised_phrases, summarise_text


@dataclass
class EntityCandidate:
    name: str
    entity_type: str = "unknown"
    summary: str = ""
    aliases: list[str] | None = None
    importance_prior: float = 0.0


@dataclass
class FactCandidate:
    subject_name: str
    predicate: str
    summary: str
    object_name: str | None = None
    object_literal: str | None = None
    confidence: float = 0.6
    valid_from: Any = None
    conflict_key: str | None = None


PREFERENCE_PATTERNS = [
    (re.compile(r"\bI (?:really )?(?:like|love|prefer|enjoy)\s+(.+?)(?:[.!?]|$)", re.I), "prefers"),
    (re.compile(r"\bI (?:really )?(?:dislike|hate)\s+(.+?)(?:[.!?]|$)", re.I), "dislikes"),
]
IDENTITY_PATTERNS = [
    (re.compile(r"\bmy name is ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.I), "identity_name"),
    (re.compile(r"\bI am a[n]?\s+(.+?)(?:[.!?]|$)", re.I), "identity_role"),
]
PROJECT_PATTERNS = [
    (re.compile(r"\bI(?:'m| am)? working on\s+(.+?)(?:[.!?]|$)", re.I), "working_on"),
    (re.compile(r"\bWe(?:'re| are)? researching\s+(.+?)(?:[.!?]|$)", re.I), "researching"),
]
TOOL_RESULT_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z0-9 ]+?)\s+(?:finished|were|was|is|are)\s+(.+?)(?:[.!?]|$)"
)


def summarise_episode(content: str) -> str:
    return summarise_text(content)


def extract_entities(episode: Episode) -> list[EntityCandidate]:
    candidates: list[EntityCandidate] = []
    for phrase in capitalised_phrases(episode.content_raw):
        entity_type = "organization" if any(token in phrase for token in ["Maple Leafs", "NHL", "Lab"]) else "topic"
        importance = 0.6 if entity_type == "organization" else 0.45
        candidates.append(
            EntityCandidate(
                name=phrase,
                entity_type=entity_type,
                summary=f"Referenced in episode {episode.id}",
                importance_prior=importance,
            )
        )
    if episode.user_id:
        candidates.append(
            EntityCandidate(
                name=f"user:{episode.user_id}",
                entity_type="person",
                summary=f"Local profile entity for user {episode.user_id}",
                importance_prior=0.6,
            )
        )
    if episode.agent_id:
        candidates.append(
            EntityCandidate(
                name=f"agent:{episode.agent_id}",
                entity_type="agent",
                summary=f"Local profile entity for agent {episode.agent_id}",
                importance_prior=0.4,
            )
        )
    deduped: dict[str, EntityCandidate] = {}
    for candidate in candidates:
        deduped[candidate.name.lower()] = candidate
    return list(deduped.values())


def extract_fact_candidates(episode: Episode) -> list[FactCandidate]:
    content = episode.content_raw
    facts: list[FactCandidate] = []
    subject_name = f"user:{episode.user_id}" if episode.user_id else "speaker"

    for pattern, predicate in PREFERENCE_PATTERNS:
        match = pattern.search(content)
        if match:
            obj = match.group(1).strip()
            facts.append(
                FactCandidate(
                    subject_name=subject_name,
                    predicate=predicate,
                    object_literal=obj,
                    summary=f"{subject_name} {predicate} {obj}",
                    confidence=0.75,
                    valid_from=episode.created_at,
                    conflict_key=predicate,
                )
            )

    for pattern, predicate in IDENTITY_PATTERNS:
        match = pattern.search(content)
        if match:
            value = match.group(1).strip()
            facts.append(
                FactCandidate(
                    subject_name=subject_name,
                    predicate=predicate,
                    object_literal=value,
                    summary=f"{subject_name} {predicate.replace('_', ' ')} {value}",
                    confidence=0.78,
                    valid_from=episode.created_at,
                    conflict_key=predicate,
                )
            )

    for pattern, predicate in PROJECT_PATTERNS:
        match = pattern.search(content)
        if match:
            value = match.group(1).strip()
            facts.append(
                FactCandidate(
                    subject_name=subject_name,
                    predicate=predicate,
                    object_literal=value,
                    summary=f"{subject_name} {predicate.replace('_', ' ')} {value}",
                    confidence=0.7,
                    valid_from=episode.created_at,
                    conflict_key=predicate,
                )
            )

    if episode.source_type == SourceType.TOOL_RESULT.value:
        for subject, result in TOOL_RESULT_PATTERN.findall(content):
            cleaned_subject = subject.strip()
            cleaned_result = result.strip()
            facts.append(
                FactCandidate(
                    subject_name=cleaned_subject,
                    predicate="tool_result",
                    object_literal=cleaned_result,
                    summary=f"{cleaned_subject} tool result: {cleaned_result}",
                    confidence=0.82,
                    valid_from=episode.created_at,
                )
            )

    episode_summary = summarise_episode(content)
    facts.append(
        FactCandidate(
            subject_name=f"episode:{episode.id}",
            predicate="topic_discussed",
            object_literal=episode_summary,
            summary=f"Episode {episode.id} discussed: {episode_summary}",
            confidence=0.45,
            valid_from=episode.created_at,
        )
    )
    return facts


def build_summary_titles(namespace_id: str, user_id: str | None, session_id: str | None) -> dict[str, str]:
    titles = {
        "project": f"{namespace_id} project rollup",
        "topic": f"{namespace_id} topic rollup",
    }
    if user_id:
        titles["user_profile"] = f"user:{user_id} profile"
    if session_id:
        titles["session_rollup"] = f"session:{session_id} rollup"
    return titles
