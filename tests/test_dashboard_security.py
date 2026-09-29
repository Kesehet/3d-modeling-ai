import time

import pytest
from fastapi.testclient import TestClient

from app import main, security


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(security, "API_TOKEN", "test-workspace-access-key")
    monkeypatch.setattr(main, "JOBS_ROOT", tmp_path)
    security._LOGIN_ATTEMPTS.clear()
    with TestClient(main.app, base_url="https://models.test") as client:
        yield client


def test_public_dashboard_cannot_read_or_mutate_workspace(client):
    page = client.get("/")
    assert page.status_code == 200 and "Workspace access key" in page.text
    assert "test-workspace-access-key" not in page.text
    assert client.get("/dashboard/api").status_code == 401
    assert client.get("/dashboard/artifacts/private/exports/model.glb").status_code == 401
    assert client.post("/dashboard/jobs", json={"prompt": "example"}).status_code == 401
    assert client.delete("/dashboard/jobs/private").status_code == 401


def test_login_cookie_unlocks_workspace_and_logout_locks_it(client):
    response = client.post("/auth/login", data={"key": "test-workspace-access-key"}, follow_redirects=False)
    assert response.status_code == 303
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie and "Secure" in cookie and "SameSite=strict" in cookie
    assert "test-workspace-access-key" not in cookie
    assert client.get("/").status_code == 200
    assert "newJobBtn" in client.get("/").text
    assert client.post("/dashboard/jobs", json={"prompt": "example"}).status_code == 200
    assert client.post("/v1/jobs", json={"prompt": "example"}).status_code == 401
    client.post("/auth/logout", follow_redirects=False)
    assert client.get("/dashboard/api").status_code == 401


def test_cookie_mutations_reject_cross_site_requests(client):
    client.post("/auth/login", data={"key": "test-workspace-access-key"})
    for headers in ({"Origin": "https://unrelated.test"}, {"Sec-Fetch-Site": "cross-site"}, {"Origin": "https://["}):
        assert client.post("/dashboard/jobs", json={"prompt": "example"}, headers=headers).status_code == 403
        assert client.post("/auth/logout", headers=headers).status_code == 403
    assert client.post("/dashboard/jobs", json={"prompt": "example"},
                       headers={"Origin": "https://models.test"}).status_code == 200


def test_existing_bearer_clients_keep_access(client):
    response = client.post("/dashboard/jobs", json={"prompt": "example"},
                           headers={"Authorization": "Bearer test-workspace-access-key"})
    assert response.status_code == 200
    assert client.post("/v1/jobs", json={"prompt": "example"},
                       headers={"Authorization": "Bearer test-workspace-access-key"}).status_code == 200


def test_invalid_keys_are_rate_limited_without_returning_submitted_key(client):
    for _ in range(5):
        result = client.post("/auth/login", data={"key": "bad-key-that-must-not-be-reflected"})
        assert result.status_code == 401
        assert "bad-key-that-must-not-be-reflected" not in result.text
    assert client.post("/auth/login", data={"key": "bad"}).status_code == 429


def test_session_expiry_tampering_and_key_rotation_fail_closed(monkeypatch):
    monkeypatch.setattr(security, "API_TOKEN", "test-key")
    now = time.time()
    session = security._new_session()
    assert security._valid_session(session)
    assert not security._valid_session(session[:-1] + "!")
    assert not security._valid_session(f"{int(now)+100}.{'a'*32}.é")
    monkeypatch.setattr(security.time, "time", lambda: now + security.SESSION_SECONDS + 1)
    assert not security._valid_session(session)
    monkeypatch.setattr(security.time, "time", lambda: now)
    monkeypatch.setattr(security, "API_TOKEN", "rotated-key")
    assert not security._valid_session(session)
