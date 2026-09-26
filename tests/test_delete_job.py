from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_dashboard_exposes_delete_job_route():
    paths = {route.path for route in app.routes}
    assert "/dashboard/jobs/{job_id}" in paths


def test_dashboard_delete_button_is_present():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert 'id="deleteBtn"' in html
    assert 'method:"DELETE"' in html
    assert "permanently removes all renders" in html
