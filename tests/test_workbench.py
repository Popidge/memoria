from __future__ import annotations

from threading import Thread
from urllib.request import Request, urlopen
import json

from memoria.providers import ProviderConfig
from memoria.workbench import RunSpec, WorkbenchService
from memoria.workbench_server import create_workbench_server


def _request_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, method=method, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def test_workbench_service_persists_runs_and_recalls_memory(engine):
    service = WorkbenchService(engine)
    run = service.create_run(
        RunSpec(
            title="Replay test",
            provider=ProviderConfig(
                provider_type="replay",
                model_name="replay",
                replay_responses=[
                    "Noted. I'll remember your meeting preferences.",
                    "You prefer oat milk and quiet cafes for Orion check-ins.",
                ],
            ),
            prompt_limit=4,
        )
    )

    first_turn = service.send_user_message(
        run["id"],
        "Remember that I prefer oat milk and quiet cafes for Orion project check-ins.",
    )
    second_turn = service.send_user_message(
        run["id"],
        "Help me plan the Orion meeting.",
    )

    assert first_turn["turn"]["assistant_message"] == "Noted. I'll remember your meeting preferences."
    assert "oat milk" in second_turn["turn"]["prompt_addition"].lower()
    assert "orion" in second_turn["turn"]["prompt_addition"].lower()

    restarted_service = WorkbenchService(engine)
    turns = restarted_service.list_turns(run["id"])
    trace = restarted_service.trace(run["id"])
    corpus = restarted_service.corpus_snapshot(run["id"])

    assert len(turns) == 2
    assert trace["steps"]
    assert corpus["counts"]["episodes"] >= 4


def test_workbench_http_contract(engine):
    service = WorkbenchService(engine)
    server = create_workbench_server(service, host="127.0.0.1", port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"

    try:
        health = _request_json("GET", f"{base_url}/api/health")
        run = _request_json(
            "POST",
            f"{base_url}/api/runs",
            {
                "title": "HTTP replay run",
                "provider": {
                    "provider_type": "replay",
                    "model_name": "replay",
                    "replay_responses": ["Stored."],
                },
            },
        )
        turn = _request_json(
            "POST",
            f"{base_url}/api/runs/{run['id']}/turns",
            {"message": "Remember that I like quiet cafes."},
        )
        _request_json(
            "POST",
            f"{base_url}/api/runs/{run['id']}/turns",
            {"message": "What should I remember for the cafe meeting?"},
        )
        turns = _request_json("GET", f"{base_url}/api/runs/{run['id']}/turns")
        trace = _request_json("GET", f"{base_url}/api/runs/{run['id']}/trace")
        graph = _request_json("GET", f"{base_url}/api/runs/{run['id']}/graph")
        corpus = _request_json("GET", f"{base_url}/api/runs/{run['id']}/corpus")
        suites = _request_json("GET", f"{base_url}/api/memoryarena/suites")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

        assert health["ok"] is True
        assert turn["turn"]["assistant_message"] == "Stored."
        assert len(turns["turns"]) == 2
        assert trace["steps"]
    assert graph["node_count"] >= 0
    assert corpus["counts"]["episodes"] >= 2
    assert "suites" in suites
