"""Real CPU BRep + API integration tests; never claim fake LLM output is real inference."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import cad_experiment
from app.cad_engine import CadBlueprint, build_cad
from app.cad_service import app
from app.cad_spatial import evaluate_spatial_contract

# Base CI installs only [dev]; the dedicated CAD workflow installs [cad,dev].
pytest.importorskip("build123d")
trimesh = pytest.importorskip("trimesh")

BLUEPRINT = {
    "title": "Three-part generic assembly",
    "units": "mm",
    "parts": [
        {"name": "core", "kind": "cylinder", "radius_mm": 12, "height_mm": 14},
        {"name": "arm_left", "kind": "box", "dimensions_mm": [75, 13, 4],
         "position_mm": [-44, 0, 0]},
        {"name": "arm_right", "kind": "box", "dimensions_mm": [75, 13, 4],
         "position_mm": [44, 0, 0]},
    ],
}


class FakeOllama:
    def __init__(self, responses=None):
        self.responses = list(responses or [BLUEPRINT])
        self.calls = []

    async def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=self.responses[min(len(self.calls) - 1, len(self.responses) - 1)])



def test_spatial_quality_gate_detects_assembly_errors():
    data = {
        "title": "Radial assembly QA",
        "parts": [
            {"name": "motor_housing", "kind": "cylinder", "radius_mm": 30, "height_mm": 40},
            {"name": "rotor_hub", "kind": "cylinder", "radius_mm": 15, "height_mm": 20},
            {"name": "downrod", "kind": "cylinder", "radius_mm": 10, "height_mm": 200,
             "position_mm": [0, 0, 120]},
            {"name": "blade_1", "kind": "box", "dimensions_mm": [300, 30, 5],
             "position_mm": [165, 0, 0]},
            {"name": "blade_2", "kind": "box", "dimensions_mm": [300, 30, 5],
             "position_mm": [-82.5, 142.894, 0], "rotation_deg": [0, 0, 120]},
            {"name": "blade_3", "kind": "box", "dimensions_mm": [300, 30, 5],
             "position_mm": [-82.5, -142.894, 0], "rotation_deg": [0, 0, 240]},
        ],
    }
    kwargs = {
        "coaxial_terms": ("motor", "hub", "rod"),
        "vertical_support_term": "rod",
        "supported_term": "motor",
        "radial_term": "blade",
        "radial_count": 3,
    }
    assert evaluate_spatial_contract(CadBlueprint.model_validate(data), **kwargs)["pass"]
    data["parts"][1]["position_mm"] = [0, 55, 0]
    data["parts"][2]["position_mm"] = [0, 0, -115]
    verdict = evaluate_spatial_contract(CadBlueprint.model_validate(data), **kwargs)
    assert not verdict["pass"]
    assert any("central axis" in issue for issue in verdict["failures"])
    assert any("upward" in issue for issue in verdict["failures"])


def test_cpu_build_creates_real_step_stl_and_named_glb(tmp_path):
    plan = CadBlueprint.model_validate(BLUEPRINT)
    report = build_cad(plan, tmp_path)
    assert report["step_reimport_valid"]
    assert report["part_count"] == 3
    assert all(item["mesh_watertight"] for item in report["parts"])
    for ext in ("step", "stl", "glb"):
        assert (tmp_path / f"assembly.{ext}").stat().st_size > 100
    mesh_scene = trimesh.load(tmp_path / "assembly.glb", force="scene")
    assert len(mesh_scene.geometry) == 3
    assert (tmp_path / "manifest.json").exists()


def test_parametric_tube_export(tmp_path):
    plan = CadBlueprint.model_validate({
        "title": "Hollow spacer", "parts": [
            {"name": "spacer", "kind": "tube", "radius_mm": 25,
             "inner_radius_mm": 12, "height_mm": 40},
        ],
    })
    assert build_cad(plan, tmp_path)["step_reimport_valid"]


@pytest.mark.parametrize("bad", [
    {"title": "missing parts", "parts": []},
    {"title": "duplicate", "parts": [
        {"name": "a", "kind": "sphere", "radius_mm": 8},
        {"name": "a", "kind": "sphere", "radius_mm": 8},
    ]},
    {"title": "bad hollow", "parts": [
        {"name": "tube", "kind": "tube", "radius_mm": 3,
         "inner_radius_mm": 6, "height_mm": 10},
    ]},
    {"title": "code injection", "parts": [
        {"name": "../bad", "kind": "box", "dimensions_mm": [1, 1, 1]},
    ]},
])
def test_rejects_invalid_blueprints(bad):
    with pytest.raises(ValueError):
        CadBlueprint.model_validate(bad)


def test_llm_schema_retry_then_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(cad_experiment, "JOBS_ROOT", tmp_path)
    fake = FakeOllama([{"title": "Invalid", "parts": []}, BLUEPRINT])
    plan, report, out = asyncio.run(cad_experiment.generate_cad(
        "Make an assembly with one cylinder and two arms", "mock-model", client=fake,
    ))
    assert plan.title == BLUEPRINT["title"]
    assert report["attempts"] == 2
    assert out.is_dir()
    assert "Previous attempt failed" in fake.calls[1]["prompt"]
    assert fake.calls[0]["schema"]["type"] == "object"


def test_api_end_to_end_with_fake_ollama(tmp_path, monkeypatch):
    monkeypatch.setenv("CAD_EXPERIMENT_ENABLED", "1")
    monkeypatch.setattr(cad_experiment, "JOBS_ROOT", tmp_path)
    fake = FakeOllama()
    monkeypatch.setattr(cad_experiment, "OllamaProxyClient", lambda: fake)
    with TestClient(app) as client:
        resp = client.post("/v1/cad-experiments", json={
            "prompt": "Create a spindle with two flat side attachments.",
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["part_count"] == 3 and body["step_reimport_valid"]
        for suffix in ("step", "stl", "glb", "json"):
            reply = client.get(body["files"][suffix])
            assert reply.status_code == 200 and len(reply.content) > 100


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CAD_EXPERIMENT_ENABLED", raising=False)
    with TestClient(app) as client:
        assert client.post("/v1/cad-experiments", json={"prompt": "A cylindrical spacer"}).status_code == 404


def test_failed_blueprint_cleans_temp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cad_experiment, "JOBS_ROOT", tmp_path)
    with pytest.raises(ValueError, match="No valid CAD"):
        asyncio.run(cad_experiment.generate_cad(
            "Create a fully described complex test geometry", "mock",
            client=FakeOllama([{"title": "bad", "parts": []}]),
        ))
    root = tmp_path / ".cad-experiments"
    assert list(root.iterdir()) == []
