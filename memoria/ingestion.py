from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re

from sqlalchemy import delete
from sqlalchemy.orm import Session

from memoria.models import Episode, EpisodeChunk, SourceType
from memoria.utils import TextEmbedder, summarise_text


@dataclass
class EpisodeInput:
    content_raw: str
    source_type: str = SourceType.USER_MESSAGE.value
    namespace_id: str = "default"
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    role: str | None = None
    metadata_json: dict[str, Any] = field(default_factory=dict)


class IngestionService:
    def __init__(self, embedder: TextEmbedder):
        self.embedder = embedder

    def add_episode(self, session: Session, episode_input: EpisodeInput) -> Episode:
        summary = summarise_text(episode_input.content_raw)
        episode = Episode(
            namespace_id=episode_input.namespace_id,
            user_id=episode_input.user_id,
            agent_id=episode_input.agent_id,
            session_id=episode_input.session_id,
            source_type=episode_input.source_type,
            role=episode_input.role,
            content_raw=episode_input.content_raw,
            content_summary=summary,
            embedding=self.embedder.embed(summary or episode_input.content_raw),
            metadata_json=episode_input.metadata_json,
        )
        session.add(episode)
        session.flush()
        self._replace_chunks(session, episode)
        return episode

    def add_messages(
        self,
        session: Session,
        messages: list[dict[str, Any]],
        namespace_id: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        default_source_type: str = SourceType.USER_MESSAGE.value,
    ) -> list[Episode]:
        episodes: list[Episode] = []
        for message in messages:
            episodes.append(
                self.add_episode(
                    session,
                    EpisodeInput(
                        content_raw=message["content"],
                        source_type=message.get("source_type", default_source_type),
                        namespace_id=namespace_id,
                        user_id=message.get("user_id", user_id),
                        agent_id=message.get("agent_id", agent_id),
                        session_id=message.get("session_id", session_id),
                        role=message.get("role"),
                        metadata_json=message.get("metadata_json", {}),
                    ),
                )
            )
        return episodes

    def update_episode(self, session: Session, episode_id: int, new_content: str) -> Episode:
        episode = session.get(Episode, episode_id)
        if episode is None:
            raise ValueError(f"Episode {episode_id} not found")
        episode.content_raw = new_content
        episode.content_summary = summarise_text(new_content)
        episode.embedding = self.embedder.embed(episode.content_summary or new_content)
        session.add(episode)
        session.flush()
        self._replace_chunks(session, episode)
        return episode

    def delete_episode(self, session: Session, episode_id: int) -> None:
        episode = session.get(Episode, episode_id)
        if episode is None:
            return
        session.execute(delete(EpisodeChunk).where(EpisodeChunk.episode_id == episode_id))
        session.delete(episode)

    def _replace_chunks(self, session: Session, episode: Episode) -> None:
        session.execute(delete(EpisodeChunk).where(EpisodeChunk.episode_id == episode.id))
        chunks = _chunk_episode(episode)
        for index, chunk in enumerate(chunks):
            summary = summarise_text(chunk["text"], max_words=18)
            session.add(
                EpisodeChunk(
                    namespace_id=episode.namespace_id,
                    episode_id=episode.id,
                    chunk_index=index,
                    chunk_type=chunk["chunk_type"],
                    text=chunk["text"],
                    summary=summary,
                    embedding=self.embedder.embed(summary or chunk["text"]),
                    salience_seed=chunk["salience_seed"],
                    metadata_json={
                        **dict(episode.metadata_json or {}),
                        "episode_role": episode.role,
                        "source_type": episode.source_type,
                    },
                )
            )
        session.flush()


def _chunk_episode(episode: Episode) -> list[dict[str, Any]]:
    segments = _split_segments(episode.content_raw)
    if not segments:
        segments = [episode.content_raw.strip()]
    return [
        {
            "text": segment,
            "chunk_type": _infer_chunk_type(episode, segment),
            "salience_seed": _salience_seed(episode, segment),
        }
        for segment in segments
        if segment.strip()
    ]


def _split_segments(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", cleaned) if part.strip()]
    segments: list[str] = []
    for paragraph in paragraphs:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", paragraph) if part.strip()]
        if len(sentences) > 1:
            segments.extend(sentences)
            continue
        if len(paragraph.split()) <= 42:
            segments.append(paragraph)
            continue
        segments.extend(sentences or [paragraph])
    return segments


def _infer_chunk_type(episode: Episode, text: str) -> str:
    metadata_kind = str((episode.metadata_json or {}).get("artifact_kind") or "").lower()
    lowered = text.lower()
    if episode.source_type == SourceType.TOOL_RESULT.value:
        return "tool_result"
    if episode.source_type == SourceType.IMPORTED_DOC.value:
        return "reasoning_note"
    if any(token in lowered for token in ("prefer", "like", "love", "dislike", "hate")):
        return "preference"
    if any(token in lowered for token in ("must", "cannot", "can't", "only", "constraint", "require")):
        return "constraint"
    if any(token in lowered for token in ("decided", "decision", "choose", "selected", "plan:")) or metadata_kind:
        return "decision"
    if any(token in lowered for token in ("working on", "status", "done", "next", "pending")):
        return "task_state"
    if episode.role in {"assistant", "agent"}:
        return "result"
    return "utterance"


def _salience_seed(episode: Episode, text: str) -> float:
    if episode.source_type == SourceType.TOOL_RESULT.value:
        return 0.55
    chunk_type = _infer_chunk_type(episode, text)
    if chunk_type in {"preference", "constraint", "decision", "task_state"}:
        return 0.65
    if chunk_type == "result":
        return 0.45
    return 0.35
