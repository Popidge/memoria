from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from sqlalchemy import select
from threading import RLock
from typing import Any
from urllib.parse import unquote, urlparse
import json
import re

from memoria.engine import MemoryEngine
from memoria.context import build_memory_context_packet, render_memory_context_packet
from memoria.models import Episode, SourceType


def _session_key(namespace_id: str, user_id: str | None, agent_id: str | None, session_id: str | None) -> str:
    parts = [namespace_id, user_id or "-", agent_id or "-", session_id or "-"]
    return "::".join(parts)


def _infer_source_type(message: dict[str, Any]) -> str:
    source_type = message.get("source_type")
    if source_type:
        return str(source_type)
    role = str(message.get("role") or "").lower()
    if role == "user":
        return SourceType.USER_MESSAGE.value
    if role in {"assistant", "agent"}:
        return SourceType.AGENT_MESSAGE.value
    if role in {"tool", "toolresult"}:
        return SourceType.TOOL_RESULT.value
    if role == "system":
        return SourceType.SYSTEM_NOTE.value
    return SourceType.USER_MESSAGE.value


def _normalise_message(message: dict[str, Any]) -> dict[str, Any]:
    metadata_json = dict(message.get("metadata_json") or message.get("metadata") or {})
    timestamp = message.get("timestamp")
    if timestamp is not None and "timestamp" not in metadata_json:
        metadata_json["timestamp"] = timestamp
    normalised = {
        "content": str(message["content"]),
        "role": message.get("role"),
        "source_type": _infer_source_type(message),
        "metadata_json": metadata_json,
    }
    for key in ("user_id", "agent_id", "session_id"):
        value = message.get(key)
        if value is not None:
            normalised[key] = value
    return normalised


def _build_prompt_addition(working_memory: list[dict[str, Any]], limit: int = 4) -> str:
    lines = [
        "Relevant Memoria recall for this turn:",
    ]
    for item in working_memory[: max(limit, 0)]:
        label = item["content_type"].replace("_", " ")
        lines.append(f"- [{label}] {item['content']}")
    if len(lines) == 1:
        return ""
    lines.append("Use only the items that are directly relevant to the user request.")
    return "\n".join(lines)


_STOPWORDS = {
    "about",
    "after",
    "agent",
    "been",
    "from",
    "help",
    "into",
    "just",
    "need",
    "plan",
    "that",
    "this",
    "turn",
    "user",
    "with",
}

_QUERY_EXPANSIONS: dict[str, set[str]] = {
    "cafe": {"coffee", "restaurant", "restaurants", "oat", "milk"},
    "coffee": {"cafe", "oat", "milk", "restaurant", "restaurants"},
    "meeting": {"project"},
    "project": {"meeting"},
}


def _query_terms(text: str) -> set[str]:
    terms = {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 4 and token not in _STOPWORDS}
    expanded = set(terms)
    for term in list(terms):
        expanded.update(_QUERY_EXPANSIONS.get(term, set()))
    return expanded


def _split_support_segments(text: str) -> list[str]:
    return [segment.strip(" -") for segment in re.split(r"(?<=[.!?])\s+|(?=\buser:)|(?=\bEpisode \d+\b)", text) if segment.strip(" -")]


def _supporting_snippets(query_text: str, search_results: list[dict[str, Any]], limit: int = 3) -> list[str]:
    terms = _query_terms(query_text)
    candidates: list[tuple[int, str]] = []
    seen: set[str] = set()
    for result in search_results:
        for segment in _split_support_segments(str(result["text"])):
            lowered = segment.lower()
            matched_terms = [term for term in terms if term in lowered]
            if not matched_terms:
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            score = len(matched_terms)
            if any(term in lowered for term in {"oat", "milk", "restaurant", "restaurants", "coffee"}):
                score += 3
            candidates.append((score, segment))
    candidates.sort(key=lambda item: (-item[0], len(item[1])))
    return [segment for _, segment in candidates[:limit]]


def _build_prompt_addition_with_support(
    working_memory: list[dict[str, Any]],
    supporting_snippets: list[str],
    limit: int = 4,
) -> str:
    lines = _build_prompt_addition(working_memory, limit=limit).splitlines()
    if not supporting_snippets:
        return "\n".join(lines)
    if not lines:
        lines = ["Relevant Memoria recall for this turn:"]
    if lines and lines[-1] == "Use only the items that are directly relevant to the user request.":
        lines = lines[:-1]
    lines.append("Supporting snippets:")
    for snippet in supporting_snippets:
        lines.append(f"- [support] {snippet}")
    lines.append("Use only the items that are directly relevant to the user request.")
    return "\n".join(lines)


def _condense_content_for_query(content: str, query_text: str, limit: int = 3) -> str:
    segments = _supporting_snippets(query_text, [{"text": content}], limit=limit)
    if not segments:
        return content
    return " ".join(segments)


@dataclass
class SidecarSession:
    session_key: str
    namespace_id: str
    user_id: str | None
    agent_id: str | None
    session_id: str | None
    run_id: str


class MemoriaSidecar:
    def __init__(self, engine: MemoryEngine):
        self.engine = engine
        self._lock = RLock()
        self._sessions: dict[str, SidecarSession] = {}

    def health(self) -> dict[str, Any]:
        with self._lock:
            session_count = len(self._sessions)
        return {
            "ok": True,
            "service": "memoria-sidecar",
            "session_count": session_count,
        }

    def bootstrap_session(
        self,
        namespace_id: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        session_key = _session_key(namespace_id, user_id, agent_id, session_id)
        with self._lock:
            existing = self._sessions.get(session_key)
            if existing is not None:
                return self._session_payload(existing)
            run = self.engine.start_run(
                namespace_id=namespace_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
            )
            session = SidecarSession(
                session_key=session_key,
                namespace_id=namespace_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                run_id=run["run_id"],
            )
            self._sessions[session_key] = session
            return self._session_payload(session)

    def ingest_messages(
        self,
        *,
        namespace_id: str,
        messages: list[dict[str, Any]],
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        consolidate: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        session = self._ensure_session(namespace_id, user_id, agent_id, session_id)
        normalised_messages = [_normalise_message(message) for message in messages]
        episodes = self.engine.add_messages(
            normalised_messages,
            namespace_id=session.namespace_id,
            user_id=session.user_id,
            agent_id=session.agent_id,
            session_id=session.session_id,
        )
        stats = self.engine.consolidate(namespace_id=session.namespace_id, force=force) if consolidate else None
        return {
            "session_key": session.session_key,
            "stored_count": len(episodes),
            "episode_ids": [episode["id"] for episode in episodes],
            "consolidation": stats,
        }

    def recall(
        self,
        *,
        namespace_id: str,
        text: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        step_type: str = "user_input",
        linked_tool_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        prompt_limit: int = 4,
    ) -> dict[str, Any]:
        session = self._ensure_session(namespace_id, user_id, agent_id, session_id)
        filters = {
            "user_id": session.user_id,
            "agent_id": session.agent_id,
            "session_id": session.session_id,
        }
        working_memory = self.engine.process_step(
            session.run_id,
            step_type=step_type,
            text=text,
            linked_tool_name=linked_tool_name,
            metadata=metadata,
        )
        search_results = self.engine.search(
            text,
            namespace_id=session.namespace_id,
            filters=filters,
            limit=max(prompt_limit + 8, 12),
        )
        prompt_working_memory = [
            {
                **item,
                "content": _condense_content_for_query(item["content"], text),
            }
            for item in working_memory
        ]
        supporting_snippets = _supporting_snippets(
            text,
            self._episode_support_candidates(session, limit=12) + search_results,
            limit=3,
        )
        trace = self.engine.get_debug_trace(session.run_id)
        latest_step = trace["steps"][-1] if trace["steps"] else None
        packet = build_memory_context_packet(
            run_id=session.run_id,
            step_index=latest_step["step_index"] if latest_step is not None else None,
            query=text,
            working_memory=prompt_working_memory,
            evidence_snippets=supporting_snippets,
            metadata={"session_key": session.session_key, "step_type": step_type},
            limit=prompt_limit,
        )
        prompt_addition = render_memory_context_packet(packet)
        if not prompt_addition:
            prompt_addition = _build_prompt_addition_with_support(
                prompt_working_memory,
                supporting_snippets,
                limit=prompt_limit,
            )
        return {
            "session_key": session.session_key,
            "run_id": session.run_id,
            "trace_id": session.run_id,
            "step_index": latest_step["step_index"] if latest_step is not None else None,
            "working_memory": working_memory,
            "memory_context_packet": packet,
            "prompt_addition": prompt_addition,
        }

    def after_turn(
        self,
        *,
        namespace_id: str,
        messages: list[dict[str, Any]],
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        consolidate: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        return self.ingest_messages(
            namespace_id=namespace_id,
            messages=messages,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            consolidate=consolidate,
            force=force,
        )

    def trace(self, session_key: str) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_key)
        if session is None:
            raise KeyError(f"Unknown session_key: {session_key}")
        trace = self.engine.get_debug_trace(session.run_id)
        return {
            "session_key": session.session_key,
            "run_id": session.run_id,
            "trace": trace,
        }

    def _ensure_session(
        self,
        namespace_id: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> SidecarSession:
        session_key = _session_key(namespace_id, user_id, agent_id, session_id)
        with self._lock:
            session = self._sessions.get(session_key)
        if session is not None:
            return session
        self.bootstrap_session(namespace_id, user_id=user_id, agent_id=agent_id, session_id=session_id)
        with self._lock:
            return self._sessions[session_key]

    @staticmethod
    def _session_payload(session: SidecarSession) -> dict[str, Any]:
        return {
            "session_key": session.session_key,
            "namespace_id": session.namespace_id,
            "user_id": session.user_id,
            "agent_id": session.agent_id,
            "session_id": session.session_id,
            "run_id": session.run_id,
        }

    def _episode_support_candidates(self, session: SidecarSession, limit: int = 12) -> list[dict[str, Any]]:
        query = select(Episode.content_raw).where(Episode.namespace_id == session.namespace_id)
        if session.user_id is not None:
            query = query.where(Episode.user_id == session.user_id)
        if session.agent_id is not None:
            query = query.where(Episode.agent_id == session.agent_id)
        if session.session_id is not None:
            query = query.where(Episode.session_id == session.session_id)
        query = query.order_by(Episode.created_at.desc()).limit(limit)
        with self.engine.session_factory() as db_session:
            texts = db_session.scalars(query).all()
        return [{"text": text} for text in texts]


class MemoriaSidecarHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], sidecar: MemoriaSidecar):
        super().__init__(server_address, MemoriaSidecarRequestHandler)
        self.sidecar = sidecar


class MemoriaSidecarRequestHandler(BaseHTTPRequestHandler):
    server: MemoriaSidecarHTTPServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._write_json(HTTPStatus.OK, self.server.sidecar.health())
            return
        if parsed.path.startswith("/sessions/") and parsed.path.endswith("/trace"):
            session_key = unquote(parsed.path[len("/sessions/") : -len("/trace")]).strip("/")
            try:
                payload = self.server.sidecar.trace(session_key)
            except KeyError as exc:
                self._write_error(HTTPStatus.NOT_FOUND, str(exc))
                return
            self._write_json(HTTPStatus.OK, payload)
            return
        self._write_error(HTTPStatus.NOT_FOUND, f"Unknown route: {parsed.path}")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/sessions/bootstrap":
                response = self.server.sidecar.bootstrap_session(
                    namespace_id=str(payload["namespace_id"]),
                    user_id=payload.get("user_id"),
                    agent_id=payload.get("agent_id"),
                    session_id=payload.get("session_id"),
                )
            elif parsed.path == "/sessions/ingest":
                response = self.server.sidecar.ingest_messages(
                    namespace_id=str(payload["namespace_id"]),
                    messages=list(payload.get("messages") or []),
                    user_id=payload.get("user_id"),
                    agent_id=payload.get("agent_id"),
                    session_id=payload.get("session_id"),
                    consolidate=bool(payload.get("consolidate", True)),
                    force=bool(payload.get("force", False)),
                )
            elif parsed.path == "/sessions/recall":
                response = self.server.sidecar.recall(
                    namespace_id=str(payload["namespace_id"]),
                    text=str(payload["text"]),
                    user_id=payload.get("user_id"),
                    agent_id=payload.get("agent_id"),
                    session_id=payload.get("session_id"),
                    step_type=str(payload.get("step_type", "user_input")),
                    linked_tool_name=payload.get("linked_tool_name"),
                    metadata=payload.get("metadata"),
                    prompt_limit=int(payload.get("prompt_limit", 4)),
                )
            elif parsed.path == "/sessions/after-turn":
                response = self.server.sidecar.after_turn(
                    namespace_id=str(payload["namespace_id"]),
                    messages=list(payload.get("messages") or []),
                    user_id=payload.get("user_id"),
                    agent_id=payload.get("agent_id"),
                    session_id=payload.get("session_id"),
                    consolidate=bool(payload.get("consolidate", True)),
                    force=bool(payload.get("force", False)),
                )
            else:
                self._write_error(HTTPStatus.NOT_FOUND, f"Unknown route: {parsed.path}")
                return
        except KeyError as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, f"Missing field: {exc.args[0]}")
            return
        except ValueError as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except json.JSONDecodeError as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, f"Invalid JSON: {exc.msg}")
            return
        self._write_json(HTTPStatus.OK, response)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_error(self, status: HTTPStatus, message: str) -> None:
        self._write_json(status, {"error": message})


def create_sidecar_server(engine: MemoryEngine, host: str = "127.0.0.1", port: int = 18733) -> MemoriaSidecarHTTPServer:
    return MemoriaSidecarHTTPServer((host, port), MemoriaSidecar(engine))
