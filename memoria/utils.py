from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
import re

import numpy as np


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "we",
    "with",
    "you",
    "your",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-/]+", text.lower())


def extract_keywords(text: str, limit: int = 8) -> list[str]:
    counts = Counter(token for token in tokenize(text) if token not in STOPWORDS)
    return [token for token, _ in counts.most_common(limit)]


def summarise_text(text: str, max_words: int = 24) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + " ..."


def episode_snippet(text: str, keywords: list[str], max_words: int = 28) -> str:
    words = text.split()
    lowered = [word.lower().strip(".,!?") for word in words]
    for keyword in keywords:
        if keyword in lowered:
            idx = lowered.index(keyword)
            start = max(0, idx - 6)
            end = min(len(words), idx + max_words - 6)
            return " ".join(words[start:end])
    return " ".join(words[:max_words])


def hash_embedding(text: str, dimensions: int = 128) -> list[float]:
    vector = np.zeros(dimensions, dtype=float)
    for token in tokenize(text):
        vector[hash(token) % dimensions] += 1.0
    norm = np.linalg.norm(vector)
    if norm:
        vector = vector / norm
    return vector.tolist()


class TextEmbedder:
    def __init__(self, model_name: str, dimensions: int = 128, enabled: bool = False):
        self.model_name = model_name
        self.dimensions = dimensions
        self.enabled = enabled
        self._model = None

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            return [0.0] * self.dimensions
        if self.enabled:
            try:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name)
                vector = self._model.encode(text, normalize_embeddings=True)
                return np.asarray(vector, dtype=float).tolist()
            except Exception:
                self.enabled = False
        return hash_embedding(text, self.dimensions)


def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return 0.0
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def keyword_overlap(left: str, right: str) -> float:
    left_tokens = set(extract_keywords(left))
    right_tokens = set(extract_keywords(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def normalise_name(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


ENTITY_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z0-9][a-zA-Z0-9&./-]+){0,3}|[A-Z]{2,}(?:\s+[A-Z0-9]{2,}){0,2})\b"
)


def capitalised_phrases(text: str) -> list[str]:
    seen: list[str] = []
    for match in ENTITY_PATTERN.findall(text):
        cleaned = match.strip(" ,.;:!?")
        if len(cleaned) >= 2 and cleaned not in seen:
            seen.append(cleaned)
    return seen


def hours_since(when: datetime | None, now: datetime | None = None) -> float:
    if when is None:
        return 999999.0
    now = now or utc_now()
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    delta = now - when
    return max(delta.total_seconds() / 3600.0, 0.0)


def recency_decay(when: datetime | None, half_life_hours: float = 72.0) -> float:
    age = hours_since(when)
    return 1.0 / (1.0 + (age / half_life_hours))


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class TextSignal:
    text: str
    keywords: list[str]
    embedding: list[float]
    linked_entities: list[str]


def build_text_signal(text: str, embedder: TextEmbedder) -> TextSignal:
    return TextSignal(
        text=text,
        keywords=extract_keywords(text),
        embedding=embedder.embed(text),
        linked_entities=capitalised_phrases(text),
    )


def word_count(text: str) -> int:
    return len(text.split())


def l2_norm(values: list[float]) -> float:
    return sqrt(sum(value * value for value in values))
