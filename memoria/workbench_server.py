from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
import json

from memoria.providers import ProviderConfig
from memoria.workbench import RunSpec, WorkbenchService


WEB_ROOT = Path(__file__).resolve().parent / "web"


class WorkbenchHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], service: WorkbenchService):
        super().__init__(server_address, WorkbenchRequestHandler)
        self.service = service


class WorkbenchRequestHandler(BaseHTTPRequestHandler):
    server: WorkbenchHTTPServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                self._write_json(HTTPStatus.OK, {"ok": True, "service": "memoria-workbench"})
                return
            if parsed.path == "/api/runs":
                limit = int(self._query(parsed).get("limit", ["50"])[0])
                self._write_json(HTTPStatus.OK, {"runs": self.server.service.list_runs(limit=limit)})
                return
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/turns"):
                run_id = parsed.path.split("/")[3]
                self._write_json(HTTPStatus.OK, {"turns": self.server.service.list_turns(run_id)})
                return
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/trace"):
                run_id = parsed.path.split("/")[3]
                raw_step = self._query(parsed).get("step_index", [None])[0]
                step_index = int(raw_step) if raw_step is not None else None
                self._write_json(HTTPStatus.OK, self.server.service.trace(run_id, step_index=step_index))
                return
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/graph"):
                run_id = parsed.path.split("/")[3]
                self._write_json(HTTPStatus.OK, self.server.service.graph_snapshot(run_id))
                return
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/corpus"):
                run_id = parsed.path.split("/")[3]
                limit = int(self._query(parsed).get("limit", ["20"])[0])
                self._write_json(HTTPStatus.OK, self.server.service.corpus_snapshot(run_id, limit=limit))
                return
            if parsed.path.startswith("/api/runs/"):
                run_id = parsed.path.split("/")[3]
                self._write_json(HTTPStatus.OK, self.server.service.get_run(run_id))
                return
            if parsed.path == "/api/memoryarena/suites":
                self._write_json(HTTPStatus.OK, self.server.service.memoryarena_suites())
                return
            if parsed.path == "/api/memoryarena/tasks":
                query = self._query(parsed)
                suite = query.get("suite", [""])[0]
                if not suite:
                    raise ValueError("suite is required")
                limit = int(query.get("limit", ["20"])[0])
                self._write_json(HTTPStatus.OK, self.server.service.memoryarena_tasks(suite, limit=limit))
                return
            if parsed.path == "/api/memoryarena/task":
                query = self._query(parsed)
                suite = query.get("suite", [""])[0]
                task_id = query.get("task_id", [""])[0]
                if not suite or not task_id:
                    raise ValueError("suite and task_id are required")
                self._write_json(HTTPStatus.OK, self.server.service.memoryarena_task(suite, task_id))
                return
            self._write_static(parsed.path)
        except KeyError as exc:
            self._write_error(HTTPStatus.NOT_FOUND, str(exc))
        except FileNotFoundError as exc:
            self._write_error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/runs":
                provider = ProviderConfig(**dict(payload.get("provider") or {}))
                spec = RunSpec(
                    title=str(payload.get("title") or "Untitled run"),
                    provider=provider,
                    namespace_id=str(payload.get("namespace_id") or "workbench.default"),
                    user_id=payload.get("user_id", "local-user"),
                    agent_id=payload.get("agent_id", "assistant"),
                    session_id=payload.get("session_id"),
                    system_prompt=str(payload.get("system_prompt") or "You are a helpful assistant."),
                    prompt_limit=int(payload.get("prompt_limit", 4)),
                    metadata=dict(payload.get("metadata") or {}),
                )
                self._write_json(HTTPStatus.CREATED, self.server.service.create_run(spec))
                return
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/turns"):
                run_id = parsed.path.split("/")[3]
                message = str(payload.get("message") or "")
                self._write_json(HTTPStatus.OK, self.server.service.send_user_message(run_id, message))
                return
            self._write_error(HTTPStatus.NOT_FOUND, f"Unknown route: {parsed.path}")
        except json.JSONDecodeError as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, f"Invalid JSON: {exc.msg}")
        except KeyError as exc:
            self._write_error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, str(exc))
        except RuntimeError as exc:
            self._write_error(HTTPStatus.BAD_GATEWAY, str(exc))

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _query(self, parsed) -> dict[str, list[str]]:
        return parse_qs(parsed.query)

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

    def _write_static(self, path: str) -> None:
        relative = "index.html" if path in {"", "/"} else path.lstrip("/")
        asset_path = (WEB_ROOT / relative).resolve()
        if WEB_ROOT.resolve() not in asset_path.parents and asset_path != WEB_ROOT.resolve():
            self._write_error(HTTPStatus.NOT_FOUND, f"Unknown route: {path}")
            return
        if not asset_path.exists() or not asset_path.is_file():
            self._write_error(HTTPStatus.NOT_FOUND, f"Unknown route: {path}")
            return
        body = asset_path.read_bytes()
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", _content_type(asset_path.suffix))
        self.end_headers()
        self.wfile.write(body)


def create_workbench_server(
    service: WorkbenchService,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> WorkbenchHTTPServer:
    return WorkbenchHTTPServer((host, port), service)


def _content_type(suffix: str) -> str:
    if suffix == ".js":
        return "text/javascript; charset=utf-8"
    if suffix == ".css":
        return "text/css; charset=utf-8"
    if suffix == ".json":
        return "application/json; charset=utf-8"
    return "text/html; charset=utf-8"
