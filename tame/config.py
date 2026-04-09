from __future__ import annotations

from pathlib import Path
from typing import Any
import tomllib

from pydantic import BaseModel, Field


class ActivationWeights(BaseModel):
    semantic: float = 0.45
    keyword: float = 0.20
    spread: float = 0.15
    recency: float = 0.10
    importance: float = 0.10


class ThresholdConfig(BaseModel):
    summary: float = 0.24
    fact: float = 0.28
    snippet: float = 0.52


class TopKConfig(BaseModel):
    semantic_candidates: int = 8
    keyword_candidates: int = 8
    neighbor_candidates: int = 8
    working_memory_total: int = 8
    per_type: int = 3


class InhibitionConfig(BaseModel):
    recent_surface_steps: int = 2
    recently_surfaced_penalty: float = 0.12
    generic_degree_penalty: float = 0.08
    duplicate_summary_penalty: float = 0.10


class NamespaceDefaults(BaseModel):
    namespace_id: str = "default"
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None


class RetrievalConfig(BaseModel):
    confidence_threshold: float = 0.2
    episode_snippet_words: int = 28


class EngineConfig(BaseModel):
    database_url: str = "sqlite:///memoria.db"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 128
    use_sentence_transformers: bool = False
    activation_weights: ActivationWeights = Field(default_factory=ActivationWeights)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    top_k: TopKConfig = Field(default_factory=TopKConfig)
    inhibition: InhibitionConfig = Field(default_factory=InhibitionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    decay_factor: float = 0.65
    scope_match_bonus: float = 0.08
    entity_similarity_threshold: float = 0.84
    duplicate_similarity_threshold: float = 0.86
    namespace_defaults: NamespaceDefaults = Field(default_factory=NamespaceDefaults)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "EngineConfig":
        if path is None:
            return cls()
        raw = tomllib.loads(Path(path).read_text())
        return cls.model_validate(raw)

    def dump(self) -> dict[str, Any]:
        return self.model_dump()


DEFAULT_CONFIG = EngineConfig()
