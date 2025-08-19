import os
os.environ.setdefault("FAST_DEV", "1")
os.environ.setdefault("ENV", "development")

from fastapi.testclient import TestClient
from backend.main import app
from backend.services.ai_orchestrator import get_ai_orchestrator


def _fake_user():
    return {"user_id": "tester", "payload": {}}


try:
    from core.security import get_current_user  # type: ignore
    app.dependency_overrides[get_current_user] = _fake_user
except Exception:  # pragma: no cover
    pass

client = TestClient(app)


def test_projects_listing():
    import asyncio
    loop = asyncio.get_event_loop()
    orch = loop.run_until_complete(get_ai_orchestrator())
    pid = loop.run_until_complete(orch.create_project("TestProj", "tester", {}))
    resp = client.get("/api/v1/music/projects")
    assert resp.status_code == 200
    data = resp.json()
    assert any(p["id"] == pid for p in data.get("items", []))


def test_project_events_endpoint():
    import asyncio
    loop = asyncio.get_event_loop()
    orch = loop.run_until_complete(get_ai_orchestrator())
    pid = loop.run_until_complete(orch.create_project("TestProj2", "tester", {}))
    resp = client.get(f"/api/v1/music/project/{pid}/events")
    assert resp.status_code == 200
    events = resp.json()
    assert isinstance(events, list)
