import asyncio
import json

import httpx
import pytest
from PIL import Image
from pydantic import ValidationError

from app import main
from app.artifacts import require_unused_version, reserve_model_version
from app.feature_tasks import FeatureEvaluation, FeatureTask
from app.ollama import OllamaProxyClient


def job(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOBS_ROOT", tmp_path)
    root = tmp_path / "abc123"
    for name in ("scene", "renders", "exports", "references", "logs"):
        (root / name).mkdir(parents=True)
    (root / "request.json").write_text(json.dumps({"prompt": "A simple object"}))
    return root


def test_versions_survive_gaps_partial_builds_and_concurrent_reservations(tmp_path):
    (tmp_path / "scene").mkdir()
    (tmp_path / "exports").mkdir()
    baseline = tmp_path / "scene/model-v9.blend"
    baseline.write_bytes(b"keep this exact baseline")
    (tmp_path / "exports/model-v12.glb").write_bytes(b"interrupted export")
    assert reserve_model_version(tmp_path) == 13
    assert reserve_model_version(tmp_path) == 14
    with pytest.raises(ValueError, match="refusing to overwrite"):
        require_unused_version(tmp_path, 9)
    assert baseline.read_bytes() == b"keep this exact baseline"


def test_replan_never_overwrites_or_accepts_a_regressing_candidate(tmp_path, monkeypatch):
    root = job(tmp_path, monkeypatch)
    spec = main.HardSurfaceCageSpec(title="Test", stations=[
        {"position": p, "profile": [[0, 0], [1, 0], [1, 1], [0, 1]]}
        for p in (-3, -1, 1, 3)
    ])
    (root / "scene/model-v7.blend").write_bytes(b"baseline")
    (root / "cage-spec-v7.json").write_text(json.dumps({"spec": spec.model_dump()}))
    main._write_status(root, generic_model={"version": 7}, working_cage_version=7)
    versions = []

    async def plan(*a, **kw):
        return spec

    async def build(*a, version, activate_status, **kw):
        assert activate_status is False
        versions.append(version)
        (root / f"scene/model-v{version}.blend").write_bytes(b"candidate")
        return {"candidate_model": {"version": version}}

    async def compare(*a, baseline_version, candidate_version):
        assert baseline_version == 7
        assert candidate_version != 7
        return {"candidate_is_better": False, "summary": "Lost protected geometry"}

    async def evaluate(*a, **kw):
        return {"passed": True, "visible": True, "criteria_satisfied": True,
                "confidence": 0.99, "reference_match_score": 0.99, "subject_recognizable": True}

    monkeypatch.setattr(main, "_build_hard_surface_cage_spec", plan)
    monkeypatch.setattr(main, "_execute_hard_surface_cage", build)
    monkeypatch.setattr(main, "_compare_generic_versions", compare)
    monkeypatch.setattr(main, "_evaluate_feature_candidate", evaluate)
    feature = FeatureTask(id="body", name="body", strategy="base_mesh_region")
    result = asyncio.run(main._generate_hard_surface_cage("abc123", reason="test", feature_task=feature))
    assert versions == [8]
    assert result["accepted"] is False
    assert main._read_status(root)["generic_model"]["version"] == 7
    assert (root / "scene/model-v7.blend").read_bytes() == b"baseline"


def test_visual_replan_switches_out_of_the_failed_cage_representation(tmp_path, monkeypatch):
    root = job(tmp_path, monkeypatch)
    spec = main.HardSurfaceCageSpec(title="Test", stations=[
        {"position": p, "profile": [[0, 0], [1, 0], [1, 1], [0, 1]]}
        for p in (-3, -1, 1, 3)
    ])
    (root / "cage-spec-v1.json").write_text(json.dumps({"spec": spec.model_dump()}))
    main._write_status(
        root,
        modeling_strategy="hard_surface_cage",
        working_cage_version=1,
        cage_edit_stall_count=0,
    )
    feature = FeatureTask(id="body", name="body", strategy="base_mesh_region")
    calls = []

    async def decide(*a, **kw):
        return main.CageEditAction(
            operation="replan_representation",
            reason="The current representation cannot express the visible shape.",
            expected_visual_effect="Use a different topology.",
        )

    async def adaptive(*a, **kw):
        calls.append(kw)
        return {"strategy": "adaptive_loft", "switched": True}

    async def cage(*a, **kw):
        raise AssertionError("replan_representation must not regenerate the same cage")

    monkeypatch.setattr(main, "_decide_hard_surface_cage_edit", decide)
    monkeypatch.setattr(main, "_generate_adaptive_mesh_fallback", adaptive)
    monkeypatch.setattr(main, "_generate_hard_surface_cage", cage)

    result = asyncio.run(
        main._refine_hard_surface_cage_incrementally("abc123", feature_task=feature)
    )

    assert result["switched"] is True
    assert calls and calls[0]["feature_task"] is feature
    assert "Do not generate another cage" in calls[0]["reason"]


def test_adaptive_representation_keeps_clear_partial_progress(tmp_path, monkeypatch):
    root = job(tmp_path, monkeypatch)
    (root / "scene/model-v7.blend").write_bytes(b"preserved cage")
    main._write_status(
        root,
        state="ready",
        stage="hard_surface_cage_needs_refinement",
        modeling_strategy="hard_surface_cage",
        generic_model=None,
        working_cage_version=7,
        cage_edit_stall_count=2,
        quality_gate={"recognizable": False, "subject_match_score": 0.2},
    )
    feature = FeatureTask(id="body", name="body", strategy="base_mesh_region")
    progress = []

    async def plan(*a, **kw):
        return object()

    async def build(*a, version, **kw):
        assert version == 8
        return {
            "status": {
                "generic_model": {
                    "version": version,
                    "strategy": "adaptive_loft",
                    "blend": f"model-v{version}.blend",
                    "renders": [],
                }
            }
        }

    async def compare(*a, baseline_version, candidate_version):
        assert baseline_version == 7
        assert candidate_version == 8
        return {
            "candidate_is_better": True,
            "summary": "The new representation has a much closer silhouette.",
        }

    async def evaluate(*a, baseline_version, candidate_version, **kw):
        assert baseline_version == 7
        assert candidate_version == 8
        return {
            "feature_id": "body",
            "passed": False,
            "visible": True,
            "criteria_satisfied": False,
            "subject_recognizable": False,
            "confidence": 0.99,
            "reference_match_score": 0.58,
            "regression_detected": False,
            "summary": "Clearly improved, but not finished.",
            "problems": ["needs another refinement"],
            "protected_geometry_notes": [],
            "model": "vision-test",
        }

    def record(_root, feature_id, *, version, summary):
        progress.append((feature_id, version, summary))

    def finish(*a, **kw):
        raise AssertionError("partial visual progress must not finish or fail the active feature")

    monkeypatch.setattr(main, "_build_adaptive_loft_spec", plan)
    monkeypatch.setattr(main, "_execute_adaptive_loft", build)
    monkeypatch.setattr(main, "_compare_generic_versions", compare)
    monkeypatch.setattr(main, "_evaluate_feature_candidate", evaluate)
    monkeypatch.setattr(main, "record_feature_progress", record)
    monkeypatch.setattr(main, "finish_feature", finish)
    monkeypatch.setattr(main, "feature_plan_summary", lambda _root: {})

    result = asyncio.run(
        main._generate_adaptive_mesh_fallback(
            "abc123",
            reason="The cage representation stalled.",
            feature_task=feature,
        )
    )

    status = main._read_status(root)
    assert result["comparison"]["candidate_is_better"] is True
    assert status["modeling_strategy"] == "adaptive_loft"
    assert status["generic_model"]["version"] == 8
    assert status["working_cage_version"] is None
    assert status["quality_gate"]["active_feature_passed"] is False
    assert progress and progress[0][0:2] == ("body", 8)
    assert (root / "scene/model-v7.blend").read_bytes() == b"preserved cage"


def test_every_visual_stage_uses_one_explicit_model_version(tmp_path, monkeypatch):
    root = job(tmp_path, monkeypatch)
    for version in (1, 2, 3):
        Image.new("RGB", (32, 32), "red").save(root / f"renders/model-v{version}-front.png")
    main._write_status(root, generic_model={"version": 1}, working_cage_version=2)
    _, labels = main._collect_images(root, main.VisionAnalyzeRequest(stage="hard_surface_cage_quality"))
    assert labels == ["renders/model-v1-front.png"]
    _, labels = main._collect_images(root, main.VisionAnalyzeRequest(stage="iterative_cage_quality", render_version=2))
    assert labels == ["renders/model-v2-front.png"]


def test_component_research_keeps_full_prompt_but_bounds_search_query(tmp_path, monkeypatch):
    root = job(tmp_path, monkeypatch)
    original = {"prompt": "Detailed instructions " * 80, "component_job": True,
                "component_name": "handle", "parent_prompt": "A kettle " * 100}
    (root / "request.json").write_text(json.dumps(original))
    queries = []

    async def research(_job_id, request):
        queries.append(request.query)
        return {}

    monkeypatch.setattr(main, "research_job", research)
    with pytest.raises(main.HTTPException) as exc:
        asyncio.run(main._ensure_reference_pack("abc123"))
    assert exc.value.status_code == 424
    assert len(queries) == 2
    assert all(len(q) <= 500 and q.startswith("handle") for q in queries)
    assert json.loads((root / "request.json").read_text()) == original


def test_incomplete_vision_judgment_is_not_a_valid_verdict():
    with pytest.raises(ValidationError):
        FeatureEvaluation.model_validate({"feature_id": "body", "visible": True})


def test_schema_is_visible_to_model_even_if_gateway_ignores_format():
    client = OllamaProxyClient(api_key="test")
    calls = []

    async def post(path, payload):
        calls.append(payload)
        return httpx.Response(200, json={"message": {"content": '{"verdict":true}'}})

    client._post = post
    asyncio.run(client.chat_json(model="test", system="judge", prompt="image",
                                schema={"type": "object", "properties": {"verdict": {"type": "boolean"}}}))
    assert '"verdict"' in calls[0]["messages"][0]["content"]
