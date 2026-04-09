from __future__ import annotations

from threading import Thread
from urllib.parse import quote
from urllib.request import Request, urlopen
import json

from memoria.demo_data import demo_catalog
from memoria.sidecar import MemoriaSidecar, create_sidecar_server


def _request_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, method=method, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def test_sidecar_personal_assistant_recall_flow(engine):
    sidecar = MemoriaSidecar(engine)
    demo = demo_catalog()["personal_assistant"]

    bootstrap = sidecar.bootstrap_session(
        namespace_id=demo["namespace_id"],
        user_id=demo["user_id"],
        agent_id=demo["agent_id"],
        session_id=demo["session_id"],
    )
    ingest = sidecar.ingest_messages(
        namespace_id=demo["namespace_id"],
        user_id=demo["user_id"],
        agent_id=demo["agent_id"],
        session_id=demo["session_id"],
        messages=demo["episodes"],
    )
    recall = sidecar.recall(
        namespace_id=demo["namespace_id"],
        user_id=demo["user_id"],
        agent_id=demo["agent_id"],
        session_id=demo["session_id"],
        text="Help me plan a cafe meeting for the Orion project.",
        metadata={"include_provenance": True},
    )
    after_turn = sidecar.after_turn(
        namespace_id=demo["namespace_id"],
        user_id=demo["user_id"],
        agent_id=demo["agent_id"],
        session_id=demo["session_id"],
        messages=[
            {
                "role": "assistant",
                "source_type": "agent_message",
                "content": "Let's choose a quiet cafe with oat milk options for the Orion project check-in.",
            }
        ],
    )
    trace = sidecar.trace(bootstrap["session_key"])

    assert ingest["stored_count"] == len(demo["episodes"])
    assert after_turn["stored_count"] == 1
    assert recall["session_key"] == bootstrap["session_key"]
    assert recall["trace_id"] == bootstrap["run_id"]
    assert "oat milk" in recall["prompt_addition"].lower()
    assert "orion" in recall["prompt_addition"].lower()
    assert "boutique hotel" not in recall["prompt_addition"].lower()
    assert trace["trace"]["steps"]


def test_sidecar_http_contract(engine):
    server = create_sidecar_server(engine, host="127.0.0.1", port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    demo = demo_catalog()["personal_assistant"]

    try:
        health = _request_json("GET", f"{base_url}/health")
        bootstrap = _request_json(
            "POST",
            f"{base_url}/sessions/bootstrap",
            {
                "namespace_id": demo["namespace_id"],
                "user_id": demo["user_id"],
                "agent_id": demo["agent_id"],
                "session_id": demo["session_id"],
            },
        )
        _request_json(
            "POST",
            f"{base_url}/sessions/ingest",
            {
                "namespace_id": demo["namespace_id"],
                "user_id": demo["user_id"],
                "agent_id": demo["agent_id"],
                "session_id": demo["session_id"],
                "messages": demo["episodes"],
            },
        )
        recall = _request_json(
            "POST",
            f"{base_url}/sessions/recall",
            {
                "namespace_id": demo["namespace_id"],
                "user_id": demo["user_id"],
                "agent_id": demo["agent_id"],
                "session_id": demo["session_id"],
                "text": "Help me plan a cafe meeting for the Orion project.",
                "step_type": "user_input",
            },
        )
        trace = _request_json(
            "GET",
            f"{base_url}/sessions/{quote(bootstrap['session_key'], safe='')}/trace",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert health["ok"] is True
    assert bootstrap["session_key"]
    assert recall["working_memory"]
    assert "oat milk" in recall["prompt_addition"].lower()
    assert trace["trace"]["run_id"] == bootstrap["run_id"]
