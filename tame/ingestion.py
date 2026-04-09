from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from tame.models import Episode, SourceType
from tame.utils import TextEmbedder, summarise_text


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
        return episode

    def delete_episode(self, session: Session, episode_id: int) -> None:
        episode = session.get(Episode, episode_id)
        if episode is None:
            return
        session.delete(episode)
