"""Semantic tests for research-first planning, parallel component jobs and measured repairs."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from app import cad_agents
from app.cad_agents import (
    ComponentJob,
    DesignPlan,
    assembly_report,
    part_aabb,
    topological_levels,
)
from app.cad_engine import CadBlueprint
from app.cad_visual import VisualAssessment
from app.research import _try_search_json


def _plan():
    return DesignPlan.model_validate({
        "title": "Generic multi-component assembly",
        "identity": "A freestanding mechanical device",
        "silhouette": "A broad primary body with a smaller attachment near its right end.",
        "dimensions_mm": [500, 300, 300],
        "components": [
            {"id": "base", "purpose": "Main foundation and structural mass",
             "center_mm": [0, 0, 0], "size_mm": [220, 100, 80]},
            {"id": "aux", "purpose": "Attached end module with physical contact",
             "center_mm": [110, 0, 0], "size_mm": [300, 100, 80],
             "depends_on": ["base"]},
        ],
        "relationships": [
            {"a": "base", "b": "aux", "relationship": "attached", "tolerance_mm": 3},
        ],
    })


def test_graph_orders_dependencies_before_parallel_jobs():
    plan = _plan()
    assert topological_levels(plan.components) == [["base"], ["aux"]]
    plan.components.append(ComponentJob(
        id="cover", purpose="Independent outer cover module",
        center_mm=(0, 0, 50), size_mm=(200, 100, 20),
    ))
    assert topological_levels(plan.components) == [["base", "cover"], ["aux"]]


def test_invalid_cycle_or_foreign_relation_rejected():
    data = _plan().model_dump()
    data["components"][0]["depends_on"] = ["aux"]
    with pytest.raises(ValueError, match="Cyclic"):
        DesignPlan.model_validate(data)
    data = _plan().model_dump()
    data["relationships"][0]["b"] = "not_in_plan"
    with pytest.raises(ValueError, match="missing"):
        DesignPlan.model_validate(data)
    data = _plan().model_dump()
    data["relationships"] = []
    with pytest.raises(ValueError, match="explicit assembly relationship"):
        DesignPlan.model_validate(data)


def test_detects_realistic_disconnected_attachment_instead_of_export_only():
    plan = _plan()
    bp = CadBlueprint.model_validate({
        "title": "Bad layout",
        "parts": [
            {"name": "base_body", "kind": "box", "dimensions_mm": [200, 80, 50]},
            {"name": "aux_body", "kind": "box", "dimensions_mm": [20, 20, 20],
             "position_mm": [300, 0, 0]},
        ],
    })
    groups = {"base": [bp.parts[0]], "aux": [bp.parts[1]]}
    result = assembly_report(plan, groups)
    assert not result["pass"]
    assert result["measurements"][0]["gap_mm"] == pytest.approx(190)
    assert "190.0 mm" in result["failures"][0]
    groups["aux"][0].position_mm = (110, 0, 0)
    assert assembly_report(plan, groups)["pass"]


def test_rotations_affect_world_space_bounds():
    bp = CadBlueprint.model_validate({
        "title": "Rotated bar",
        "parts": [{"name": "beam", "kind": "box",
                   "dimensions_mm": [200, 20, 10],
                   "rotation_deg": [0, 0, 90]}],
    })
    lo, hi = part_aabb(bp.parts[0])
    assert hi[0] - lo[0] == pytest.approx(20)
    assert hi[1] - lo[1] == pytest.approx(200)


def test_loft_sections_contribute_to_world_space_spatial_qa():
    part = CadBlueprint.model_validate({
        "title": "Loft body",
        "parts": [{
            "name": "shell", "kind": "loft", "loft_axis": "x",
            "loft_sections": [
                {"at_mm": -100, "width_mm": 20, "depth_mm": 15},
                {"at_mm": 100, "width_mm": 80, "depth_mm": 40},
            ], "rotation_deg": [0, 0, 90],
        }],
    }).parts[0]
    lower, upper = part_aabb(part)
    assert upper[0] - lower[0] == pytest.approx(80)
    assert upper[1] - lower[1] == pytest.approx(200)
    assert upper[2] - lower[2] == pytest.approx(40)


def test_research_precedes_component_agents_and_repair_preserves_other_work(tmp_path, monkeypatch):
    calls = []

    async def fake_research(query, directory, max_images):
        directory.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.new("RGB", (500, 350), "white").save(directory / "sample.jpg")
        return {
            "pages": [{"title": "Object specifications", "url": "https://example.test/object",
                       "extract": "Evidence describing an object."}],
            "candidate_count": 1,
            "references": [{"stored_name": "sample.jpg",
                            "source_url": "https://example.test/object",
                            "image_url": "https://example.test/object.jpg"}],
        }

    def fake_builder(blueprint, root):
        calls.append(("export", [p.name for p in blueprint.parts]))
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text(json.dumps({
            "blueprint": blueprint.model_dump(), "validation": {"ok": True},
        }))
        for name in ("assembly.step", "assembly.stl", "assembly.glb"):
            (root / name).write_bytes(b"fake-geometry-for-unit-test")
        return {"ok": True, "step_reimport_valid": True,
                "part_count": len(blueprint.parts), "parts": [],
                "quality_scope": "geometry-only"}

    class FakeClient:
        async def chat_json(self, *, system, prompt, **kwargs):
            if "Judge the actual image pixels" in system:
                calls.append(("vision", None))
                return SimpleNamespace(data={
                    "matches_subject": True, "confidence": 0.99,
                    "viewpoint": "side", "visible_shape": "A compact base with an end attachment",
                    "identifiable_features": ["main base", "end module"],
                })
            if "chief CAD designer" in system:
                calls.append(("plan", None))
                return SimpleNamespace(data=_plan().model_dump())
            task = json.loads(prompt)["component"]["id"]
            correction = json.loads(prompt)["repair_feedback"]
            calls.append(("part", task, bool(correction)))
            if task == "base":
                parts = [{"name": "COMPONENTID_base_body", "kind": "box",
                          "dimensions_mm": [200, 80, 50]}]
            else:
                parts = [{"name": "COMPONENTID_aux_body", "kind": "box",
                          "dimensions_mm": [20, 20, 20],
                          "position_mm": [110 if correction else 220, 0, 0]}]
            return SimpleNamespace(data={"title": task, "parts": parts})

    async def fake_inspect_candidate(**kwargs):
        calls.append(("visual", kwargs["round_number"]))
        return VisualAssessment(
            recognizable=True, confidence=0.92, subject_match_score=0.9,
            silhouette_match_score=0.86, missing_features=[], defects=[],
            repair_component_ids=[], explanation="All required structural parts present",
        )

    monkeypatch.setattr(cad_agents, "research_web_references", fake_research)
    monkeypatch.setattr(cad_agents, "build_cad", fake_builder)
    monkeypatch.setattr(cad_agents, "inspect_candidate", fake_inspect_candidate)
    bp, result, path = asyncio.run(cad_agents.generate_agent_cad(
        "A generic mechanical device", root=tmp_path, client=FakeClient(),
    ))
    assert bp.title == _plan().title
    assert result["ok"] and result["assembly_validation"]["pass"]
    assert result["visual_acceptance_verified"]
    assert calls == [
        ("vision", None), ("plan", None), ("part", "base", False),
        ("part", "aux", False), ("part", "aux", True),
        ("export", ["base_body", "aux_body"]),
        ("visual", 1),
    ]
    assert (path / "design-plan.json").is_file()
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["agent_assessment"]["quality_status"] == "vision_model_accepted"


def test_research_failure_stops_agent_geometry_generation(tmp_path, monkeypatch):
    async def no_references(*args, **kwargs):
        return {"candidate_count": 0, "references": [], "pages": []}

    class NoPlanningClient:
        async def chat_json(self, **kwargs):
            raise AssertionError("No planning without verified research")

    monkeypatch.setattr(cad_agents, "research_web_references", no_references)
    with pytest.raises(ValueError, match="No visually confirmed"):
        asyncio.run(cad_agents.generate_agent_cad(
            "An unknown named object", root=tmp_path, client=NoPlanningClient(),
        ))


def test_visual_review_repairs_only_named_component_and_stops_when_accepted(tmp_path, monkeypatch):
    """Two real workflow rounds; all unrelated geometry stays unchanged."""
    calls = []

    async def known_research(prompt, root, client):
        calls.append("research")
        return {"verified_visual_references": [{"stored_name": "known.jpg"}], "text_sources": []}

    async def inspected(**kwargs):
        calls.append(("visual", kwargs["round_number"]))
        if kwargs["round_number"] == 1:
            return VisualAssessment(
                recognizable=False, confidence=0.92, subject_match_score=0.22,
                silhouette_match_score=0.3, missing_features=["top silhouette"],
                defects=["Main body shape is too small"],
                repair_component_ids=["base"],
                explanation="Base shape is too short relative to the object references",
            )
        return VisualAssessment(
            recognizable=True, confidence=0.9, subject_match_score=0.89,
            silhouette_match_score=0.85, missing_features=[], defects=[],
            repair_component_ids=[], explanation="Good match to documented geometry",
        )

    def fake_builder(blueprint, root):
        root.mkdir(parents=True, exist_ok=True)
        calls.append(("export", {p.name: p.model_dump() for p in blueprint.parts}))
        for name in ("assembly.step", "assembly.stl", "assembly.glb"):
            (root / name).write_bytes(b"test-model-placeholder")
        (root / "manifest.json").write_text(json.dumps({
            "blueprint": blueprint.model_dump(), "validation": {"ok": True},
        }))
        return {"ok": True, "step_reimport_valid": True,
                "part_count": len(blueprint.parts), "parts": []}

    class DesignClient:
        async def chat_json(self, *, system, prompt, **kwargs):
            if "chief CAD designer" in system:
                calls.append("chief")
                return SimpleNamespace(data=_plan().model_dump())
            info = json.loads(prompt)
            task = info["component"]["id"]
            repair = bool(info["repair_feedback"])
            calls.append(("component", task, repair))
            if task == "base":
                response = {"name": "base_body", "kind": "box",
                            "dimensions_mm": [210 if repair else 200, 80, 50]}
            else:
                response = {"name": "aux_body", "kind": "box",
                            "dimensions_mm": [20, 20, 20],
                            "position_mm": [110, 0, 0]}
            return SimpleNamespace(data={"title": task, "parts": [response]})

    monkeypatch.setattr(cad_agents, "gather_research", known_research)
    monkeypatch.setattr(cad_agents, "build_cad", fake_builder)
    monkeypatch.setattr(cad_agents, "inspect_candidate", inspected)
    plan, result, output = asyncio.run(cad_agents.generate_agent_cad(
        "A generic mechanical device", root=tmp_path, client=DesignClient(),
    ))
    assert result["ok"] and result["visual_acceptance_verified"]
    assert result["agent_workflow"]["rounds"] == 2
    assert [event for event in calls if event[:1] == ("component",)] == [
        ("component", "base", False), ("component", "aux", False),
        ("component", "base", True),
    ]
    assert [p.name for p in plan.parts] == ["base_body", "aux_body"]
    assert plan.parts[0].dimensions_mm[0] == 210
    assert plan.parts[1].position_mm == (110, 0, 0)
    assert (output / "assembly.glb").exists()
    assert (output / "candidates" / "round-1" / "assembly.glb").exists()
    assert (output / "candidates" / "round-2" / "assembly.glb").exists()


def test_reference_research_handles_html_gateway_block_without_crashing():
    """An HTML bot block must not become an uncaught JSONDecodeError."""
    async def check():
        def handler(req):
            if req.url.path.endswith("/api.php"):
                return httpx.Response(200, text="<html>bot block</html>")
            return httpx.Response(200, json={"pages": [{"title": "Sample"}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            legacy, error = await _try_search_json(client, "https://en.wikipedia.org/w/api.php")
            assert legacy == {} and "ValueError" in error
            rest, error = await _try_search_json(
                client, "https://en.wikipedia.org/w/rest.php/v1/search/page"
            )
            assert not error and rest["pages"][0]["title"] == "Sample"

    asyncio.run(check())
