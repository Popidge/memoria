from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json
import os


@dataclass
class ProviderConfig:
    provider_type: str = "replay"
    model_name: str = "replay"
    api_base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = None
    timeout_seconds: int = 120
    replay_responses: list[str] = field(default_factory=list)


@dataclass
class ProviderResult:
    text: str
    latency_ms: float
    payload: dict[str, Any]
    usage: dict[str, Any]


class ChatProvider(Protocol):
    def generate(self, messages: list[dict[str, Any]]) -> ProviderResult:
        ...


class ReplayProvider:
    def __init__(self, config: ProviderConfig):
        self.config = config
        self._index = 0

    def generate(self, messages: list[dict[str, Any]]) -> ProviderResult:
        started_at = perf_counter()
        if self._index < len(self.config.replay_responses):
            text = self.config.replay_responses[self._index]
        else:
            last_user = next((item["content"] for item in reversed(messages) if item["role"] == "user"), "")
            text = f"[replay] {last_user}".strip()
        self._index += 1
        latency_ms = (perf_counter() - started_at) * 1000.0
        return ProviderResult(
            text=text,
            latency_ms=latency_ms,
            payload={
                "provider": "replay",
                "model": self.config.model_name,
                "choices": [{"message": {"role": "assistant", "content": text}}],
            },
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )


class OpenAICompatibleProvider:
    def __init__(self, config: ProviderConfig):
        if not config.api_base_url:
            raise ValueError("api_base_url is required for the openai-compatible provider")
        self.config = config

    def generate(self, messages: list[dict[str, Any]]) -> ProviderResult:
        api_key = self._api_key()
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens
        request = Request(
            self._chat_completions_url(),
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
            },
        )
        started_at = perf_counter()
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Provider request failed: {exc.code} {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach provider endpoint: {exc.reason}") from exc
        latency_ms = (perf_counter() - started_at) * 1000.0
        parsed = json.loads(raw)
        text = _extract_chat_completion_text(parsed)
        usage = _normalise_usage(parsed.get("usage"))
        return ProviderResult(text=text, latency_ms=latency_ms, payload=parsed, usage=usage)

    def _api_key(self) -> str | None:
        if not self.config.api_key_env:
            return None
        value = os.environ.get(self.config.api_key_env, "").strip()
        if not value:
            raise RuntimeError(f"Environment variable {self.config.api_key_env} is not set")
        return value

    def _chat_completions_url(self) -> str:
        base = self.config.api_base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"


def build_provider(config: ProviderConfig) -> ChatProvider:
    if config.provider_type == "openai-compatible":
        return OpenAICompatibleProvider(config)
    if config.provider_type == "replay":
        return ReplayProvider(config)
    raise KeyError(f"Unknown provider_type: {config.provider_type}")


def _extract_chat_completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                item.get("text", "").strip()
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            return "\n".join(part for part in parts if part)
    text = choices[0].get("text") if isinstance(choices[0], dict) else ""
    return str(text or "").strip()


def _normalise_usage(raw_usage: Any) -> dict[str, Any]:
    if not isinstance(raw_usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = raw_usage.get("prompt_tokens")
    if input_tokens is None:
        input_tokens = raw_usage.get("input_tokens", 0)
    output_tokens = raw_usage.get("completion_tokens")
    if output_tokens is None:
        output_tokens = raw_usage.get("output_tokens", 0)
    total_tokens = raw_usage.get("total_tokens")
    if total_tokens is None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }
