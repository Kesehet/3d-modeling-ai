"""Integration contract: regular jobs, CPU CAD worker, 3D viewer and exports."""
import asyncio
import json
import uuid

import httpx
from fastapi import HTTPException

from app import cad_jobs, main
from app.dashboard import dashboard_page


def create_job_in_root(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOBS_ROOT", tmp_path)
    job_id = str(uuid.uuid4())
    root = tmp_path / job_id
    for folder in main.ARTIFACT_CATEGORIES:
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "request.json").write_text(json.dumps({"prompt": "A mechanical assembly with three arms",
                                                   "intended_use": "rendering"}), encoding="utf-8")
    main._write_status(root, state="created", stage="waiting_for_input")
    return job_id, root


def test_dashboard_has_cpu_cad_choice_and_step_download():
    html = dashboard_page().body.decode()
    assert 'id="jobEngine"' in html
    assert 'value="cpu_cad"' in html
    assert '/cad' in html
    assert 'glb|obj|stl|step' in html
    assert 'CAD model ready for visual review' in html


def test_cpu_cad_job_status_and_viewer_exports(tmp_path, monkeypatch):
    ident, root = create_job_in_root(tmp_path, monkeypatch)
    worker_dir = tmp_path / ".cad-experiments" / uuid.uuid4().hex
    worker_dir.mkdir(parents=True)
    for filename in ("assembly.glb", "assembly.stl", "assembly.step", "manifest.json"):
        (worker_dir / filename).write_bytes(b"x" * 220)
    response_payload = {
        "id": worker_dir.name, "ok": True, "step_reimport_valid": True,
        "part_count": 3, "parts": [{"name": "core"}, {"name": "left"}, {"name": "right"}],
        "attempts": 1,
    }

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            assert url.endswith("/v1/cad-experiments")
            return httpx.Response(200, json=response_payload,
                                  request=httpx.Request("POST", url))

    monkeypatch.setattr(cad_jobs.httpx, "AsyncClient", FakeClient)
    asyncio.run(cad_jobs.run_cad_job(ident, 1))
    status = main._read_status(root)
    assert status["state"] == "ready"
    assert status["stage"] == "cad_needs_visual_review"
    assert status["generic_model"]["version"] == 1
    assert status["cad"]["part_count"] == 3
    assert status["quality_gate"]["recognizable"] is None
    for suffix in (".step", ".stl", ".glb", "-cad.json", "-qa.json"):
        assert (root / "exports" / f"model-v1{suffix}").is_file()
    assert not worker_dir.exists()


def test_cpu_cad_worker_failure_is_visible_and_existing_model_is_preserved(tmp_path, monkeypatch):
    ident, root = create_job_in_root(tmp_path, monkeypatch)
    original = root / "exports" / "model-v1.glb"
    original.write_bytes(b"previous successful model")
    main._write_status(root, state="ready", generic_model={"version": 1})

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            return httpx.Response(502, text="worker unavailable",
                                  request=httpx.Request("POST", url))

    monkeypatch.setattr(cad_jobs.httpx, "AsyncClient", FakeClient)
    asyncio.run(cad_jobs.run_cad_job(ident, 2))
    assert main._read_status(root)["state"] == "failed"
    assert main._read_status(root)["stage"] == "cad_failed"
    assert original.read_bytes() == b"previous successful model"


def test_cpu_cad_rejects_running_legacy_job(tmp_path, monkeypatch):
    ident, root = create_job_in_root(tmp_path, monkeypatch)
    main._write_status(root, state="running", stage="legacy_geometry")
    try:
        asyncio.run(cad_jobs.start_cad_generation(ident, cad_jobs.CadGenerateRequest()))
    except HTTPException as exc:
        assert exc.status_code == 409
    else:
        raise AssertionError("Concurrent Blender/CAD generations must be rejected")
