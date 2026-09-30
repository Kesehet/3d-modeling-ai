import asyncio
import json

import httpx
import pytest
from PIL import Image
from pydantic import ValidationError

from app import main
from app.artifacts import require_unused_version, reserve_model_version
from app.feature_tasks import FeatureEvaluation, FeatureTask
from app.ollama import OllamaJSONResult, OllamaProxyClient


def job(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOBS_ROOT", tmp_path)
    root = tmp_path / "abc123"
    for name in ("scene", "renders", "exports", "references", "logs"):
        (root / name).mkdir(parents=True)
    (root / "request.json").write_text(json.dumps({"prompt": "A simple object"}))
    return root


def test_initial_base_mesh_blockout_defers_cutters_and_attachments():
    spec = main.HardSurfaceCageSpec(
        title="Primary blockout",
        stations=[
            {"position": p, "profile": [[0, 0], [1, 0], [1, 1], [0, 1]]}
            for p in (-3, -1, 1, 3)
        ],
        cutters=[
            {
                "name": "opening",
                "shape": "cube",
                "location": [0, 0, 0],
                "scale": [1, 1, 1],
                "rotation_deg": [0, 0, 0],
            }
        ],
        attachments=[
            {
                "name": "separate detail",
                "shape": "cube",
                "location": [0, 0, 0],
                "scale": [1, 1, 1],
            }
        ],
    )
    feature = FeatureTask(
        id="primary",
        name="Primary mass",
        strategy="base_mesh_region",
    )

    cleaned, cutters, attachments = main._clean_initial_primary_blockout(
        spec,
        feature,
        has_existing_cage=False,
    )

    assert cutters == 1
    assert attachments == 1
    assert cleaned.cutters == []
    assert cleaned.attachments == []
    assert len(spec.cutters) == 1
    assert len(spec.attachments) == 1


def test_existing_cage_keeps_later_feature_geometry_during_replan():
    spec = main.HardSurfaceCageSpec(
        title="Existing cage",
        stations=[
            {"position": p, "profile": [[0, 0], [1, 0], [1, 1], [0, 1]]}
            for p in (-3, -1, 1, 3)
        ],
        cutters=[
            {
                "name": "accepted opening",
                "shape": "cube",
                "location": [0, 0, 0],
                "scale": [1, 1, 1],
                "rotation_deg": [0, 0, 0],
            }
        ],
    )
    feature = FeatureTask(
        id="primary",
        name="Primary mass",
        strategy="base_mesh_region",
    )

    kept, cutters, attachments = main._clean_initial_primary_blockout(
        spec,
        feature,
        has_existing_cage=True,
    )

    assert kept.cutters
    assert cutters == 0
    assert attachments == 0


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
        cage_edit_stall_count=2,
    )
    feature = FeatureTask(id="body", name="body", strategy="base_mesh_region")
    calls = []

    async def decide(*a, **kw):
        assert kw["allow_replan"] is True
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


def test_first_cage_refinement_locks_representation_replan(tmp_path, monkeypatch):
    root = job(tmp_path, monkeypatch)
    spec = main.HardSurfaceCageSpec(title="Test", stations=[
        {"position": p, "profile": [[0, 0], [1, 0], [1, 1], [0, 1]]}
        for p in (-3, -1, 1, 3)
    ])
    (root / "cage-spec-v1.json").write_text(json.dumps({"spec": spec.model_dump()}))
    main._write_status(
        root,
        modeling_strategy="hard_surface_cage",
        generic_model={"version": 1},
        working_cage_version=1,
        cage_edit_stall_count=0,
    )
    feature = FeatureTask(id="body", name="body", strategy="base_mesh_region")
    flags = []

    async def decide(*a, **kw):
        flags.append(kw["allow_replan"])
        return main.CageEditAction(
            operation="reshape_station",
            target_index=1,
            width_scale=1.1,
            reason="Widen the dominant section before abandoning the representation.",
            expected_visual_effect="Improve the primary silhouette.",
        )

    async def build(*a, version, activate_status, **kw):
        assert activate_status is False
        return {"candidate_model": {"version": version}}

    async def compare(*a, **kw):
        return {"candidate_is_better": False, "summary": "Not enough improvement yet."}

    async def evaluate(*a, **kw):
        return {
            "passed": False,
            "visible": True,
            "criteria_satisfied": False,
            "confidence": 0.99,
            "reference_match_score": 0.2,
            "subject_recognizable": False,
            "regression_detected": False,
            "summary": "Still needs work.",
        }

    monkeypatch.setattr(main, "_decide_hard_surface_cage_edit", decide)
    monkeypatch.setattr(main, "_execute_hard_surface_cage", build)
    monkeypatch.setattr(main, "_compare_generic_versions", compare)
    monkeypatch.setattr(main, "_evaluate_feature_candidate", evaluate)

    result = asyncio.run(
        main._refine_hard_surface_cage_incrementally("abc123", feature_task=feature)
    )

    assert flags == [False]
    assert result["action"]["operation"] == "reshape_station"
    assert result["kept"] is False
    assert main._read_status(root)["cage_edit_stall_count"] == 1


def test_cage_edit_schema_hides_replan_until_it_is_allowed():
    locked = main._cage_edit_action_schema(allow_replan=False)
    unlocked = main._cage_edit_action_schema(allow_replan=True)

    assert "replan_representation" not in locked["properties"]["operation"]["enum"]
    assert "replan_representation" in unlocked["properties"]["operation"]["enum"]


def test_cage_edit_normalizer_clamps_safe_numeric_overshoot():
    payload = main._normalize_cage_edit_action_payload({
        "operation": "reshape_station_region",
        "reason": "Compress the overly tall section.",
        "influence_radius": 9,
        "length_scale": 0.5,
        "height_scale": 0.5,
        "width_scale": 1.8,
        "height_offset_fraction": -0.5,
        "position_offset_fraction": 0.4,
        "width_offset_fraction": 0.6,
        "point_height_offset_fraction": -0.8,
        "insert_fraction": 0.95,
    })

    action = main.CageEditAction.model_validate(payload)

    assert action.influence_radius == 3
    assert action.length_scale == 0.75
    assert action.height_scale == 0.65
    assert action.width_scale == 1.45
    assert action.height_offset_fraction == -0.30
    assert action.position_offset_fraction == 0.20
    assert action.width_offset_fraction == 0.30
    assert action.point_height_offset_fraction == -0.30
    assert action.insert_fraction == 0.85


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


def test_final_quality_rechecks_whole_object_and_current_version(tmp_path, monkeypatch):
    root = job(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "feature_plan_summary", lambda _: {"required_complete": True})
    calls = []

    async def judge(_job_id, *, stage, render_version):
        calls.append(render_version)
        return {"scope": "whole_object", "evaluated_version": render_version, "recognizable": True}

    monkeypatch.setattr(main, "_generic_recognizability_check", judge)
    status = main._write_status(root, generic_model={"version": 4},
                               quality_gate={"scope": "feature", "recognizable": True})
    assert main._auto_improve_goal_reached(root, status) is False
    status = asyncio.run(main._ensure_final_model_quality("abc123", status))
    assert main._auto_improve_goal_reached(root, status) is True
    asyncio.run(main._ensure_final_model_quality("abc123", status))
    status["generic_model"]["version"] = 5
    assert main._auto_improve_goal_reached(root, status) is False
    asyncio.run(main._ensure_final_model_quality("abc123", status))
    assert calls == [4, 5]


def test_final_quality_failure_is_persisted_and_retryable(tmp_path, monkeypatch):
    root = job(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "feature_plan_summary", lambda _: {"required_complete": True})

    async def unavailable(*args, **kwargs):
        raise main.HTTPException(status_code=502, detail="vision timeout")

    monkeypatch.setattr(main, "_generic_recognizability_check", unavailable)
    status = main._write_status(root, stage="generic_recognizable", generic_model={"version": 4},
                               quality_gate={"scope": "feature", "recognizable": True})
    status = asyncio.run(main._ensure_final_model_quality("abc123", status))
    assert status["quality_gate"]["recognizable"] is False
    assert main._auto_improve_goal_reached(root, status) is False
    assert "evaluated_version" not in status["quality_gate"]
    assert "vision timeout" in main._read_status(root)["quality_gate"]["summary"]


def test_research_does_not_rejudge_same_rejected_image_across_queries(tmp_path, monkeypatch):
    job(tmp_path, monkeypatch)
    checked = []

    async def plan(*args):
        return main.ReferenceSearchPlan(primary_query="subject", alternate_queries=["subject side", "subject top"])

    async def search(*args, **kwargs):
        return {"references": [{"sha256": "same-image", "stored_name": "candidate.jpg"}], "pages": []}

    async def verify(_root, *, records, **kwargs):
        checked.extend(r["sha256"] for r in records)
        return [], records

    monkeypatch.setattr(main, "_plan_reference_search", plan)
    monkeypatch.setattr(main, "research_web_references", search)
    monkeypatch.setattr(main, "_verify_reference_candidates", verify)
    result = asyncio.run(main.research_job("abc123", main.ResearchRequest(max_images=3)))
    assert checked == ["same-image"]
    assert result["status"]["stage"] == "references_unavailable"


@pytest.mark.parametrize("malformed", [
    {"decisions": []},
    {"decisions": [{"stored_name": "candidate.jpg", "accept": False}]},
])
def test_reference_verifier_retries_incomplete_judgments(tmp_path, monkeypatch, malformed):
    root = job(tmp_path, monkeypatch)
    Image.new("RGB", (32, 32), "brown").save(root / "references" / "candidate.jpg")
    calls = []

    async def chat(self, **kwargs):
        calls.append(kwargs)
        data = malformed if len(calls) == 1 else {"decisions": [{
            "stored_name": "candidate.jpg", "accept": True, "match_score": 0.95,
            "exact_identity_match": True, "useful_for_geometry": True,
            "reason": "The complete subject is visible with a readable silhouette.",
        }]}
        return OllamaJSONResult(data=data, endpoint="test", usage={})

    monkeypatch.setattr(main, "VISION_MODELS", ["primary", "fallback"])
    monkeypatch.setattr(OllamaProxyClient, "chat_json", chat)
    accepted, rejected = asyncio.run(main._verify_reference_batch(
        root, plan=main.ReferenceSearchPlan(primary_query="subject"), query="subject",
        records=[{"stored_name": "candidate.jpg", "title": "Subject"}],
    ))
    assert [call["model"] for call in calls] == ["primary", "fallback"]
    assert len(accepted) == 1 and rejected == []
    assert accepted[0]["verification_model"] == "fallback"
    required = calls[0]["schema"]["$defs"]["ReferenceCandidateDecision"]["required"]
    assert {"accept", "match_score", "exact_identity_match", "useful_for_geometry", "reason"} <= set(required)
    assert (root / "references" / "candidate.jpg").exists()
    assert len(list((root / "logs").glob("reference-verification-raw-*.json"))) == 2


@pytest.mark.parametrize("malformed", [
    {"candidate_is_better": True},
    {"summary": "An ambiguous response without a decision."},
])
def test_visual_comparison_retries_missing_judgment_fields(tmp_path, monkeypatch, malformed):
    root = job(tmp_path, monkeypatch)
    for version in (1, 2):
        for view in ("front", "front-left", "left", "back", "right", "front-right"):
            Image.new("RGB", (32, 32), "gray").save(root / "renders" / f"model-v{version}-{view}.png")
    calls = []

    async def chat(self, **kwargs):
        calls.append(kwargs)
        data = malformed if len(calls) == 1 else {
            "candidate_is_better": True, "summary": "Candidate restores the reference proportions.",
            "improvements": ["Corrected silhouette"], "regressions": [],
        }
        return OllamaJSONResult(data=data, endpoint="test", usage={})

    monkeypatch.setattr(main, "VISION_MODELS", ["primary", "fallback"])
    monkeypatch.setattr(OllamaProxyClient, "chat_json", chat)
    comparison = asyncio.run(main._compare_generic_versions(root, baseline_version=1, candidate_version=2))
    assert [call["model"] for call in calls] == ["primary", "fallback"]
    assert comparison["candidate_is_better"] is True
    assert comparison["model"] == "fallback"
    assert {"candidate_is_better", "summary"} <= set(calls[0]["schema"]["required"])


@pytest.mark.parametrize("axis,expected", [("y", [2, 4, 1]), ("x", [4, 2, 1])])
def test_planner_dimensions_catch_swapped_axes_and_flattened_cages(axis, expected):
    spec = main.HardSurfaceCageSpec(title="Envelope", axis=axis, stations=[
        {"position": p, "profile": [[0, 0], [1, 0], [1, 1], [0, 1]]}
        for p in (-2, -1, 1, 2)
    ])
    main._validate_cage_dimensions(spec, expected)
    flat = spec.model_copy(deep=True)
    for station in flat.stations:
        station.profile = [(w, z * 0.1) for w, z in station.profile]
    with pytest.raises(ValueError, match="contradict the planner's intended dimensions"):
        main._validate_cage_dimensions(flat, expected)


def test_director_can_choose_vertical_loft_instead_of_horizontal_cage(monkeypatch):
    calls = []

    async def loft(job_id, **kwargs):
        calls.append((job_id, kwargs))
        return {"strategy": "adaptive_loft"}

    async def cage(*args, **kwargs):
        raise AssertionError("An explicit loft decision must not become a horizontal cage.")

    monkeypatch.setattr(main, "_generate_adaptive_mesh_fallback", loft)
    monkeypatch.setattr(main, "_generate_hard_surface_cage", cage)
    result = asyncio.run(main._generate_directed_mesh("abc123", {
        "action": "build_mesh", "mesh_representation": "adaptive_loft",
        "summary": "The primary mass is upright and radial.", "instructions": ["Use world Z."],
    }))
    assert result["strategy"] == "adaptive_loft"
    assert len(calls) == 1 and "world Z" in calls[0][1]["reason"]


def test_primary_geometry_uses_visual_brief_then_reasoning_coordinates(tmp_path, monkeypatch):
    job(tmp_path, monkeypatch)
    calls = []

    async def no_plan(*args):
        return None

    async def chat(self, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            assert kwargs["model"] == main.VISION_MODELS[0]
            assert kwargs["images"] == ["reference_image"]
            # Perspective-based estimates must not veto a coherent geometry plan.
            data = {"dimensions_xyz": [6.5, 8, 3.2], "silhouette_notes": ["Wide base", "Level upper surface"],
                    "construction_notes": ["Use Y as the horizontal sweep axis."]}
        else:
            assert kwargs["model"] == main.REASONING_MODEL and kwargs["images"] is None
            assert "dimensions_xyz" in kwargs["prompt"]
            assert "intended_dimensions_xyz" in kwargs["schema"]["required"]
            data = {"title": "Primary envelope", "axis": "y", "intended_dimensions_xyz": [2, 4, 1],
                    "subdivision_levels": 0, "stations": [
                {"position": p, "profile": [[0, 0], [1, 0], [1, 0.5], [1, 1], [0, 1]]}
                for p in (-2, -1.2, -0.4, 0.4, 1.2, 2)
            ]}
        return OllamaJSONResult(data=data, endpoint="test", usage={})

    monkeypatch.setattr(main, "_ensure_feature_plan", no_plan)
    monkeypatch.setattr(main, "_collect_images", lambda *args: (
        ["reference_image", "failed_render"], ["references/source.jpg", "renders/model-v1-front.png"]))
    monkeypatch.setattr(OllamaProxyClient, "chat_json", chat)
    spec = asyncio.run(main._build_hard_surface_cage_spec("abc123", reason="Construct primary form"))
    assert spec.axis == "y" and len(calls) == 2
    main._validate_cage_dimensions(spec, [2, 4, 1])


def test_failed_regeneration_restores_previous_model_and_quality(tmp_path, monkeypatch):
    root = job(tmp_path, monkeypatch)
    previous = {"version": 2, "blend": "model-v2.blend"}
    quality = {"evaluated_version": 2, "recognizable": False}
    main._write_status(root, state="ready", stage="needs_refinement", generic_model=previous,
                       modeling_strategy="hard_surface_cage", quality_gate=quality)

    async def failed_candidate(*args):
        main._write_status(root, generic_model={"version": 3}, modeling_strategy="procedural",
                           quality_gate={"evaluated_version": 3})
        raise main.HTTPException(status_code=502, detail="Geometry planner failed")

    monkeypatch.setattr(main, "_generate_generic_scene_candidate", failed_candidate)
    with pytest.raises(main.HTTPException):
        asyncio.run(main.generate_generic_scene("abc123", main.GenericGenerateRequest(auto_research=False)))
    result = main._read_status(root)
    assert result["generic_model"] == previous
    assert result["quality_gate"] == quality
    assert result["modeling_strategy"] == "hard_surface_cage"
    assert result["state"] == "ready"


def test_mesh_part_preserves_coordinates_and_rejects_invalid_faces():
    part = {"name": "Shaped panel", "shape": "mesh", "location": [0, 0, 0], "scale": [1, 1, 1],
            "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "faces": [[0, 1, 2]]}
    normalized = main._normalize_scene_spec_payload({"objects": [part]}, "Design")
    validated = main.GenericSceneSpec.model_validate(normalized)
    assert validated.objects[0].shape == "mesh"
    assert validated.objects[0].faces == [[0, 1, 2]]
    assert len(validated.objects[0].vertices) == 3
    for face in [[0, 1, 99], [0, 0, 2], [-1, 1, 2]]:
        with pytest.raises(ValidationError):
            main.SceneObjectSpec.model_validate({**part, "faces": [face]})
    with pytest.raises(ValidationError):
        main.SceneObjectSpec.model_validate({**part, "vertices": [[float("nan"), 0, 0]] * 3})


def test_primary_feature_review_covers_front_side_and_rear():
    feature = FeatureTask(id="body", name="Body", strategy="base_mesh_region",
                          target_regions=["rear_quarter", "front_fascia", "side_panels"])
    assert main._feature_diagnostic_views(feature) == ("front-left", "left", "back-right")


def test_submitted_design_cannot_overwrite_a_better_active_model(tmp_path, monkeypatch):
    root = job(tmp_path, monkeypatch)
    baseline = {"version": 1, "blend": "model-v1.blend"}
    (root / "scene/model-v1.blend").write_bytes(b"baseline")
    main._write_status(root, state="ready", generic_model=baseline, quality_gate={"evaluated_version": 1})
    spec = main.HardSurfaceCageSpec(title="Candidate", stations=[
        {"position": p, "profile": [[0,0],[1,0],[1,1],[0,1]]} for p in [-2,-1,1,2]])

    async def build(*args, version, activate_status):
        assert activate_status is False
        return {"candidate_model": {"version": version}}

    async def compare(*args, **kwargs):
        return {"candidate_is_better": False, "summary": "Visible regression"}

    monkeypatch.setattr(main, "_usable_reference_index", lambda r: [{"verified": True}])
    monkeypatch.setattr(main, "_execute_hard_surface_cage", build)
    monkeypatch.setattr(main, "_compare_generic_versions", compare)
    result = asyncio.run(main.submit_model_design("abc123", spec))
    assert result["kept"] is False
    assert main._read_status(root)["generic_model"] == baseline
    assert main._read_status(root)["quality_gate"]["evaluated_version"] == 1


@pytest.mark.parametrize("incremental", [True, False])
def test_improving_working_mesh_survives_without_replacing_best_model(tmp_path, monkeypatch, incremental):
    root = job(tmp_path, monkeypatch)
    spec = main.HardSurfaceCageSpec(title="Envelope", stations=[
        {"position": p, "profile": [[0,0],[1,0],[1,1],[0,1]]} for p in [-2,-1,1,2]])
    (root / "cage-spec-v1.json").write_text(json.dumps({"spec": spec.model_dump()}))
    (root / "scene/model-v3.blend").write_bytes(b"displayed baseline")
    main._write_status(root, state="ready", generic_model={"version": 3}, working_cage_version=1,
                       quality_gate={"evaluated_version": 3, "recognizable": False})
    feature = FeatureTask(id="body", name="Body", strategy="base_mesh_region")

    async def build(*args, version, **kwargs):
        return {"candidate_model": {"version": version}}

    async def plan(*args, **kwargs):
        return spec

    async def decide(*args, **kwargs):
        return main.CageEditAction(operation="reshape_station", target_index=1, width_scale=1.1,
                                   reason="Improve a section")

    async def compare(*args, baseline_version, **kwargs):
        return {"candidate_is_better": baseline_version == 1, "summary": "Working mesh progressed"}

    async def evaluate(*args, **kwargs):
        return {"passed": False, "reference_match_score": .5, "summary": "More work remains"}

    monkeypatch.setattr(main, "_execute_hard_surface_cage", build)
    monkeypatch.setattr(main, "_build_hard_surface_cage_spec", plan)
    monkeypatch.setattr(main, "_decide_hard_surface_cage_edit", decide)
    monkeypatch.setattr(main, "_compare_generic_versions", compare)
    monkeypatch.setattr(main, "_evaluate_feature_candidate", evaluate)
    fn = (main._refine_hard_surface_cage_incrementally("abc123", feature_task=feature) if incremental
          else main._generate_hard_surface_cage("abc123", reason="Different construction", feature_task=feature))
    result = asyncio.run(fn)
    status = main._read_status(root)
    assert result["kept"] is True and result["promoted"] is False
    assert status["working_cage_version"] == 4
    assert status["generic_model"]["version"] == 3
    assert status["quality_gate"]["evaluated_version"] == 3
    assert status["cage_edit_stall_count"] == 0


def test_visual_edit_brief_drives_reasoning_geometry_with_full_profiles(tmp_path, monkeypatch):
    root = job(tmp_path, monkeypatch)
    (root / "references/source.jpg").write_bytes(b"reference")
    for view in ["front-left", "left", "back-right"]:
        (root / "renders" / f"model-v1-{view}.png").write_bytes(b"render")
    monkeypatch.setattr(main, "_usable_reference_index", lambda r: [{"stored_name": "source.jpg"}])
    monkeypatch.setattr(main, "_encode_vision_images", lambda paths: ["pixels"] * len(paths))
    spec = main.HardSurfaceCageSpec(
        title="Envelope",
        intended_dimensions_xyz=[2, 4, 1],
        stations=[
            {"position": p, "profile": [[0,0],[1,0],[1,1],[0,1]]}
            for p in [-2,-1,1,2]
        ],
    )
    calls = []

    async def chat(self, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            assert kwargs["model"] == main.VISION_MODELS[0] and kwargs["images"]
            data = {"diagnosis": "Top transition is too abrupt", "correction": "Raise the inner profile point", "preserve": ["Lower edge"]}
        else:
            assert kwargs["model"] == main.REASONING_MODEL and kwargs["images"] is None
            assert "Full station profile geometry:" in kwargs["prompt"]
            assert "Top transition is too abrupt" in kwargs["prompt"]
            assert "Declared reference-driven target dimensions XYZ: [2.0, 4.0, 1.0]" in kwargs["prompt"]
            assert "Current cage dimensions XYZ computed from stations: [2.0, 4.0, 1.0]" in kwargs["prompt"]
            assert "envelope_matches_declared_within_12pct=True" in kwargs["prompt"]
            data = {"operation": "reshape_profile_point", "target_index": 1, "point_index": 2,
                    "point_height_offset_fraction": .1, "reason": "Smooth the upper transition"}
        return OllamaJSONResult(data=data, endpoint="test", usage={})

    monkeypatch.setattr(OllamaProxyClient, "chat_json", chat)
    action = asyncio.run(main._decide_hard_surface_cage_edit("abc123", baseline_version=1, spec=spec,
        feature_task=FeatureTask(id="body", name="Body", strategy="base_mesh_region"), allow_replan=False))
    assert action.operation == "reshape_profile_point" and len(calls) == 2
