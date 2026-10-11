"""Integration contract: regular jobs, CPU CAD worker, 3D viewer and exports."""
import asyncio
import json
import uuid

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import cad_jobs, dashboard, main
from app.dashboard import dashboard_page


def create_job_in_root(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOBS_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "JOBS_ROOT", tmp_path)
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
    worker_id = uuid.uuid4().hex
    response_payload = {
        "id": worker_id, "ok": True, "step_reimport_valid": True,
        "part_count": 3, "parts": [{"name": "core"}, {"name": "left"}, {"name": "right"}],
        "attempts": 1,
    }

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def get(self, url, **kwargs):
            suffix = url.rsplit("/", 1)[-1]
            assert suffix in ("step", "stl", "glb", "json")
            return httpx.Response(200, content=b"x" * 220, request=httpx.Request("GET", url))

        async def delete(self, url, **kwargs):
            assert url.endswith(worker_id)
            return httpx.Response(200, json={"deleted": True}, request=httpx.Request("DELETE", url))

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

def test_real_cpu_geometry_flows_through_standard_job_exports(tmp_path, monkeypatch):
    """Actual OpenCascade STEP and GLB, simulated Ollama+network transport."""
    import pytest

    import_step = pytest.importorskip("build123d").import_step
    trimesh = pytest.importorskip("trimesh")
    from app.cad_engine import CadBlueprint, build_cad

    ident, root = create_job_in_root(tmp_path, monkeypatch)
    worker_id = uuid.uuid4().hex
    worker_dir = tmp_path / "worker" / worker_id
    blueprint = CadBlueprint.model_validate({
        "title": "Three-body benchmark",
        "parts": [
            {"name": "motor", "kind": "cylinder", "radius_mm": 25, "height_mm": 25},
            {"name": "shaft", "kind": "cylinder", "radius_mm": 7, "height_mm": 70,
             "position_mm": [0, 0, 45]},
            {"name": "side_arm", "kind": "box", "dimensions_mm": [60, 12, 4],
             "position_mm": [35, 0, 0]},
        ],
    })
    report = build_cad(blueprint, worker_dir)

    class LocalWorkerTransport:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            return httpx.Response(200, request=httpx.Request("POST", url),
                                  json={"id": worker_id, "ok": True,
                                        "step_reimport_valid": report["step_reimport_valid"],
                                        "part_count": report["part_count"],
                                        "parts": report["parts"], "attempts": 1})

        async def get(self, url, **kwargs):
            suffix = url.rsplit("/", 1)[-1]
            filename = {"step": "assembly.step", "stl": "assembly.stl",
                        "glb": "assembly.glb", "json": "manifest.json"}[suffix]
            return httpx.Response(200, request=httpx.Request("GET", url),
                                  content=(worker_dir / filename).read_bytes())

        async def delete(self, url, **kwargs):
            return httpx.Response(200, request=httpx.Request("DELETE", url),
                                  json={"deleted": True})

    monkeypatch.setattr(cad_jobs.httpx, "AsyncClient", LocalWorkerTransport)
    asyncio.run(cad_jobs.run_cad_job(ident, 1))
    stored = root / "exports"
    assert main._read_status(root)["stage"] == "cad_needs_visual_review"
    imported = import_step(str(stored / "model-v1.step"))
    assert imported.is_valid and imported.volume > 0
    glb = trimesh.load(stored / "model-v1.glb", force="scene")
    assert len(glb.geometry) == 3
    assert (stored / "model-v1-cad.json").stat().st_size > 100

def test_dashboard_cad_route_creates_trackable_job(tmp_path, monkeypatch):
    ident, root = create_job_in_root(tmp_path, monkeypatch)
    invocations = []

    async def fake_cad_runner(job_id, version, *, model=None, reference_mode="shape_guidance"):
        invocations.append((job_id, version))

    monkeypatch.setattr(cad_jobs, "run_cad_job", fake_cad_runner)
    with TestClient(main.app) as client:
        response = client.post(f"/dashboard/jobs/{ident}/cad", json={})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["started"] is True
        assert body["job_id"] == ident
        assert body["engine"] == "cpu_cad"
        assert client.get("/dashboard/api").json()["jobs"][0]["job_id"] == ident
    assert (root / "scene" / ".model-v1.reserve").is_file()
    assert invocations == [(ident, 1)]

def test_cad_sidecar_has_internal_api_and_external_ollama_networks():
    from pathlib import Path

    import yaml

    compose = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "compose.hostinger.yaml").read_text()
    )
    cad = compose["services"]["cad"]
    api = compose["services"]["api"]
    assert "worker-net" in cad["networks"] and "worker-net" in api["networks"]
    assert "traefik-proxy" in cad["networks"]
    assert "ports" not in cad
    assert "traefik.enable=false" in cad["labels"]
    assert cad["environment"]["CAD_EXPERIMENT_ENABLED"] == "1"
    assert compose["networks"]["worker-net"]["internal"] is True
    assert cad["mem_limit"]


def test_cad_reads_uploaded_images_and_normalizes_for_proxy(tmp_path, monkeypatch):
    import base64
    from PIL import Image

    ident, root = create_job_in_root(tmp_path, monkeypatch)
    source = root / "references" / "a-test.png"
    Image.new("RGB", (1300, 750), "navy").save(source)
    (root / "references.json").write_text(json.dumps([
        {"stored_name": "a-test.png", "uploaded_at": "2026-10-11T00:00:00Z"}
    ]))
    refs = cad_jobs._load_reference_images(root)
    assert len(refs) == 1
    raw = base64.b64decode(refs[0])
    assert raw[:2] == bytes.fromhex("ffd8")
    assert len(raw) < 425_000


def test_dashboard_reference_route_and_input_are_present():
    html = dashboard_page().body.decode()
    assert 'id="jobReferences"' in html
    assert 'id="jobReferenceMode"' in html
    assert "/references" in html
