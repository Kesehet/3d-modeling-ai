import json
from datetime import UTC, datetime, timedelta

from app.dashboard import dashboard_page, reconcile_running_status


def test_dashboard_is_gallery_first():
    html = dashboard_page().body.decode("utf-8")
    assert 'id="jobGallery"' in html
    assert 'id="detailView"' in html
    assert 'id="fileList"' in html
    assert 'id="referenceGrid"' in html
    assert 'Reference images used' in html
    assert 'id="newJobBtn"' in html
    assert 'data-tab=' not in html


def test_dashboard_job_detail_can_download_and_improve():
    html = dashboard_page().body.decode("utf-8")
    assert '/dashboard/artifacts/' in html
    assert 'id="improveBtn"' in html
    assert '/auto-improve' in html
    assert 'Auto improve ×30' in html
    assert '/generate' in html


def test_dashboard_job_detail_exposes_reference_images():
    html = dashboard_page().body.decode("utf-8")
    assert '"references"' in html
    assert 'latest_vision_images' in html
    assert 'Used in latest vision pass' in html


def test_dashboard_surfaces_quality_gate_and_allows_vision_retry():
    html = dashboard_page().body.decode("utf-8")
    assert 'id="qualityBanner"' in html
    assert 'generic_needs_strategy_switch' in html
    assert 'generic_quality_unverified' in html
    assert 'Quality gate: current model is not recognizable enough.' in html
    assert 'diagnosis has been queued for automatic repair' in html
    assert 'Visual QA needs another pass.' in html


def test_dashboard_offers_mesh_fallback_instead_of_dead_end():
    html = dashboard_page().body.decode("utf-8")
    assert 'Auto improve ×30' in html
    assert 'adaptive_mesh_needs_refinement' in html
    assert 'Auto improve ×30' in html



def test_stale_running_job_is_reconciled_to_failed(tmp_path):
    root = tmp_path / "job-1"
    root.mkdir()
    now = datetime(2026, 9, 27, 10, 0, tzinfo=UTC)
    status = {
        "job_id": "job-1",
        "state": "running",
        "stage": "generic_build_v2",
        "updated_at": (now - timedelta(hours=3)).isoformat(),
    }
    (root / "status.json").write_text(json.dumps(status), encoding="utf-8")

    reconciled = reconcile_running_status(root, status, now=now)

    assert reconciled["state"] == "failed"
    assert reconciled["stage"] == "generic_build_v2"
    assert reconciled["interrupted"] is True
    assert reconciled["interrupted_stage"] == "generic_build_v2"
    persisted = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert persisted["state"] == "failed"
    history = json.loads((root / "history.json").read_text(encoding="utf-8"))
    assert history[-1]["event"] == "stale_running_recovered"


def test_recent_running_job_is_not_reconciled(tmp_path):
    root = tmp_path / "job-2"
    root.mkdir()
    now = datetime(2026, 9, 27, 10, 0, tzinfo=UTC)
    status = {
        "job_id": "job-2",
        "state": "running",
        "stage": "generic_build_v1",
        "updated_at": (now - timedelta(minutes=10)).isoformat(),
    }
    (root / "status.json").write_text(json.dumps(status), encoding="utf-8")

    reconciled = reconcile_running_status(root, status, now=now)

    assert reconciled["state"] == "running"
    assert json.loads((root / "status.json").read_text(encoding="utf-8"))["state"] == "running"


def test_restart_force_reconciles_even_recent_running_job(tmp_path):
    root = tmp_path / "job-3"
    root.mkdir()
    now = datetime(2026, 9, 27, 10, 0, tzinfo=UTC)
    status = {
        "job_id": "job-3",
        "state": "running",
        "stage": "agent_selected_procedural",
        "updated_at": (now - timedelta(seconds=5)).isoformat(),
    }
    (root / "status.json").write_text(json.dumps(status), encoding="utf-8")

    reconciled = reconcile_running_status(root, status, now=now, force=True)

    assert reconciled["state"] == "failed"
    assert "restarted" in reconciled["error"].lower()



def test_dashboard_exposes_feature_worker_state_and_disables_cache():
    response = dashboard_page()
    html = response.body.decode("utf-8")

    assert 'AI feature sub-jobs' in html
    assert 'Feature coordinator' in html
    assert 'CLIENT_UI_VERSION="feature-workers-v3"' in html
    assert 'data.ui_version' in html
    assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate"
