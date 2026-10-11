"""Real CPU BRep + API integration tests; never claim fake LLM output is real inference."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import cad_experiment
from app.cad_assembly_qa import evaluate_assembly_solids
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


def test_measured_disconnection_retries_and_preserves_good_export(tmp_path, monkeypatch):
    monkeypatch.setattr(cad_experiment, "JOBS_ROOT", tmp_path)
    floating = {
        "title": "Bad candidate",
        "parts": [
            {"name": "object_A", "kind": "box", "dimensions_mm": [10, 10, 10]},
            {"name": "object_B", "kind": "box", "dimensions_mm": [10, 10, 10],
             "position_mm": [130, 0, 0]},
        ],
    }
    repaired = {
        **floating,
        "title": "Repaired candidate",
        "parts": [
            floating["parts"][0],
            {**floating["parts"][1], "position_mm": [9, 0, 0]},
        ],
    }
    fake = FakeOllama([floating, repaired])
    plan, report, out = asyncio.run(cad_experiment.generate_cad(
        "Two parts that must touch", "mock", client=fake,
    ))
    assert report["attempts"] == 2
    assert report["assembly_qa"]["pass"]
    assert "Disconnected solid group" in fake.calls[1]["prompt"]
    assert plan.title == "Repaired candidate"
    assert (out / "assembly-front-left.png").stat().st_size > 100
    assert (out / "assembly-top.png").stat().st_size > 100


def test_multimodal_reference_compares_real_render_with_original(tmp_path, monkeypatch):
    monkeypatch.setattr(cad_experiment, "JOBS_ROOT", tmp_path)
    brief = {
        "subject_description": "Solid mechanical object with radial arms",
        "silhouette_geometry": ["long horizontal pieces"],
        "required_components": ["cylinder", "two arms"],
        "spatial_relationships": ["arms attach to core"],
        "relative_proportions": ["arm length exceeds core diameter"],
        "uncertain_details": [],
    }
    fake = FakeOllama([
        brief,
        BLUEPRINT,
        {"passed": False, "subject_recognizable": False,
         "reference_match_score": 0.1, "findings": ["wrong silhouette"]},
        BLUEPRINT,
        {"passed": True, "subject_recognizable": True,
         "reference_match_score": 0.92, "findings": []},
    ])
    _, report, out = asyncio.run(cad_experiment.generate_cad(
        "A spindle with two long arms", "mock", client=fake,
        reference_images=["sample-original-b64"], reference_mode="exact_match",
    ))
    assert len(fake.calls) == 5
    assert fake.calls[0]["images"] == ["sample-original-b64"]
    assert fake.calls[0]["model"] == cad_experiment.VISION_MODEL
    assert fake.calls[1]["images"] is None
    assert fake.calls[1]["model"] == "mock"
    assert "Independently inspected" in fake.calls[1]["prompt"]
    assert len(fake.calls[2]["images"]) == 6
    assert fake.calls[2]["images"][0] == "sample-original-b64"
    assert "wrong silhouette" in fake.calls[3]["prompt"]
    assert report["reference_visual_qa"]["status"] == "passed"
    assert report["reference_geometry_brief"]["status"] == "verified"
    assert not report["repair_exhausted"]
    assert (out / "manifest.json").is_file()


def test_photo_slope_requires_nonprimitive_outline(tmp_path, monkeypatch):
    monkeypatch.setattr(cad_experiment, "JOBS_ROOT", tmp_path)
    simple_brief = {
        "subject_description": "A tapered multi-part structure",
        "silhouette_geometry": ["Trapezoidal roof with sloped side edges"],
        "required_components": ["roof", "base"],
        "spatial_relationships": ["roof rests on base"],
        "relative_proportions": ["roof narrower at top"],
        "uncertain_details": [],
    }
    box_only = {
        "title": "Bad simplification",
        "parts": [{"name": "roof", "kind": "box", "dimensions_mm": [80, 20, 20]}],
    }
    profiled = {
        "title": "Reference silhouette respected",
        "parts": [{
            "name": "roof", "kind": "profile_extrusion", "profile_plane": "xz",
            "profile_points_mm": [[-40, 0], [40, 0], [24, 25], [-20, 25]],
            "height_mm": 20,
        }],
    }
    fake = FakeOllama([
        simple_brief, box_only, profiled,
        {"passed": True, "subject_recognizable": True,
         "reference_match_score": 0.9, "findings": []},
    ])
    _, report, _ = asyncio.run(cad_experiment.generate_cad(
        "Tapered structural form", "mock", client=fake,
        reference_images=["mock-reference"], reference_mode="shape_guidance",
    ))
    assert report["attempts"] == 2
    assert report["parts"][0]["kind"] == "profile_extrusion"
    assert "nonrectangular" in fake.calls[2]["prompt"].lower()


def test_exhausted_disconnected_model_is_never_certified(tmp_path, monkeypatch):
    monkeypatch.setattr(cad_experiment, "JOBS_ROOT", tmp_path)
    disconnected = {
        "title": "Disconnected",
        "parts": [
            {"name": "first", "kind": "box", "dimensions_mm": [10, 10, 10]},
            {"name": "second", "kind": "box", "dimensions_mm": [10, 10, 10],
             "position_mm": [300, 0, 0]},
        ],
    }
    _, report, out = asyncio.run(cad_experiment.generate_cad(
        "Two touching spheres", "mock", client=FakeOllama([disconnected]),
    ))
    assert report["attempts"] == 3
    assert report["repair_exhausted"]
    assert not report["assembly_qa"]["pass"]
    assert (out / "assembly.step").is_file()


@pytest.mark.parametrize("kind,details", [
    ("profile_extrusion", {
        "profile_points_mm": [[-20, -10], [30, -10], [20, 8], [0, 15], [-15, 12]],
        "height_mm": 15,
    }),
    ("profile_extrusion", {
        "profile_points_mm": [[-45, 0], [45, 0], [30, 28], [-22, 35]],
        "profile_plane": "xz",
        "height_mm": 30,
    }),
    ("section_loft", {
        "loft_sections_mm": [
            [-30, 50, 28, 0, 0],
            [0, 60, 32, 8, 0],
            [35, 28, 16, 0, 0],
        ],
    }),
])
def test_nonprimitive_generic_shapes_export_true_cad(kind, details, tmp_path):
    blueprint = CadBlueprint.model_validate({
        "title": "Independent shape", "parts": [
            {"name": "model", "kind": kind, **details}
        ],
    })
    report = build_cad(blueprint, tmp_path)
    assert report["step_reimport_valid"]
    assert report["assembly_qa"]["pass"]
    assert report["renders"]["top"] == "assembly-top.png"
    assert report["renders"]["left"] == "assembly-left.png"
    assert report["renders"]["back-right"] == "assembly-back-right.png"
    assert (tmp_path / "assembly.glb").stat().st_size > 100


@pytest.mark.parametrize("bad", [
    {"kind": "profile_extrusion", "height_mm": 20, "profile_points_mm": [[0, 0], [2, 2], [4, 4]]},
    {"kind": "section_loft", "loft_sections_mm": [[2, 5, 6, 0, 0], [1, 5, 6, 0, 0]]},
    {"kind": "section_loft", "loft_sections_mm": [[0, 5, 6, 0, 0], [1, -5, 6, 0, 0]]},
])
def test_nonprimitive_shapes_reject_degenerate_profiles(bad):
    with pytest.raises(ValueError):
        CadBlueprint.model_validate({"title": "Bad", "parts": [{"name": "bad", **bad}]})


def test_brep_distance_rejects_false_aabb_contact():
    """Two spheres can have overlapping AABBs while remaining physically apart."""
    from build123d import Pos, Sphere

    first = Sphere(10)
    second = Pos(18, 18, 0) * Sphere(10)
    report = evaluate_assembly_solids(
        {"first": first, "second": second},
        {"first": [[-10, -10, -10], [10, 10, 10]],
         "second": [[8, 8, -10], [28, 28, 10]]},
    )
    assert report["measurement"] == "brep_closest_solid_distance"
    assert not report["pass"]
    assert len(report["groups"]) == 2
    assert not report["contact_verified"]
    assert report["nearest_noncontact_pairs"][0]["gap_mm"] > 5


def test_brep_distance_accepts_true_contact_for_unrelated_solids():
    from build123d import Box, Pos

    a = Box(20, 20, 20)
    b = Pos(15, 0, 0) * Box(20, 20, 20)
    report = evaluate_assembly_solids(
        {"a": a, "b": b},
        {"a": [[-10, -10, -10], [10, 10, 10]],
         "b": [[5, -10, -10], [25, 10, 10]]},
    )
    assert report["pass"]
    assert report["solid_proximity_verified"] is True
    assert report["physical_fit_verified"] is False
