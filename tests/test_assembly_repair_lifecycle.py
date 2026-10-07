"""Exercise real feature scheduling and repair state across external worker boundaries."""

import asyncio
import json

from PIL import Image

from app import main
from app.feature_tasks import FeaturePlan, FeatureTask, load_feature_plan, save_feature_plan


def test_post_assembly_repair_reinstalls_then_reviews_original_feature(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOBS_ROOT", tmp_path)
    root = tmp_path / "abc123"
    for category in main.ARTIFACT_CATEGORIES:
        (root / category).mkdir(parents=True)
    (root / "request.json").write_text(json.dumps({"prompt": "A generic assembled object"}))

    def artifact(version):
        (root / "scene" / f"model-v{version}.blend").write_bytes(b"preserved blend")
        for view in ("front", "front-left", "left", "back-left", "back",
                     "back-right", "right", "front-right", "top"):
            Image.new("RGB", (32, 32), "white").save(root / "renders" / f"model-v{version}-{view}.png")

    artifact(4)
    artifact(6)
    (root / "scene-spec-v4.json").write_text(json.dumps({"version": 4, "spec": {
        "title": "Preserved parent", "presentation_base": False,
        "objects": [{"name": "body", "shape": "cube", "location": [0, 0, 0], "dimensions": [2, 2, 1]}],
    }}))
    assembly = main.ComponentAssemblySpec(instances=[main.ComponentInstanceSpec(location=[1, 0, 0])])
    (root / "component-assembly-v6.json").write_text(json.dumps({
        "baseline_version": 4, "candidate_version": 6, "feature_id": "part",
        "child_job_id": "def456", "assembly": assembly.model_dump(),
    }))
    save_feature_plan(root, FeaturePlan(subject="Generic assembled object", features=[
        FeatureTask(id="part", name="Installed part", build_mode="component_job",
                    status="accepted", accepted_version=6, acceptance_verified=True,
                    acceptance_score=0.95, component_job_id="def456"),
        FeatureTask(id="finish", name="Joint continuity", strategy="surface_detail",
                    depends_on=["part"], acceptance_criteria=["No visible joint gaps"]),
    ]))
    main._write_status(root, state="ready", stage="component_assembly_accepted",
                       modeling_strategy="procedural", generic_model={
                           "version": 6, "title": "Assembled", "blend": "model-v6.blend",
                           "assembled_components": [{"feature_id": "part"}],
                       })
    monkeypatch.setattr(main, "_usable_reference_index", lambda _root: [{"stored_name": "reference.png"}])
    reviewed = []

    async def evaluate(_job_id, feature, *, baseline_version, candidate_version):
        reviewed.append((feature.id, candidate_version))
        passed = candidate_version == 7
        return {"feature_id": feature.id, "passed": passed, "visible": True,
                "criteria_satisfied": passed, "subject_recognizable": True, "confidence": 1.0,
                "reference_match_score": 0.95 if passed else 0.5, "reference_match_required": False,
                "regression_detected": False, "summary": "Sound joint" if passed else "Joint gap",
                "problems": [] if passed else ["Joint gap"]}

    async def integrity(*args, **kwargs):
        assert kwargs["render_version"] == 6
        return {"pass_integrity": False, "blocking_defects": ["Joint gap"],
                "repair_instructions": ["Correct component placement"], "summary": "Joint gap"}

    async def plan_repair(*args, **kwargs):
        return main.AssemblyRepairDecision(rationale="Reinstall preserved component", parent_feature_ids=[]), []

    async def child_ready(_job_id, feature):
        assert feature.component_job_id == "def456"
        return {"ready": True, "child_job_id": "def456", "version": 2}

    async def plan_assembly(*args):
        return assembly

    async def execute(_job_id, feature, child_id, planned):
        assert main._read_status(root)["generic_model"]["version"] == 4
        assert child_id == "def456"
        artifact(7)
        return {"baseline_version": 4, "candidate_version": 7, "blend": "model-v7.blend",
                "renders": [], "qa": "qa-v7.json"}

    async def compare(*args, **kwargs):
        return {"candidate_is_better": True, "summary": "Corrected contact"}

    async def quality(_job_id, *, stage, render_version=None):
        version = render_version or main._read_status(root)["generic_model"]["version"]
        return {"scope": "whole_object", "evaluated_version": version,
                "recognizable": True, "assembly_integrity_pass": True, "subject_match_score": 0.95}

    monkeypatch.setattr(main, "_evaluate_feature_candidate", evaluate)
    monkeypatch.setattr(main, "_final_assembly_integrity_check", integrity)
    monkeypatch.setattr(main, "_plan_assembled_parent_repair", plan_repair)
    monkeypatch.setattr(main, "_ensure_component_child_ready", child_ready)
    monkeypatch.setattr(main, "_plan_component_assembly", plan_assembly)
    monkeypatch.setattr(main, "_execute_component_assembly_candidate", execute)
    monkeypatch.setattr(main, "_compare_generic_versions", compare)
    monkeypatch.setattr(main, "_generic_recognizability_check", quality)

    async def lifecycle():
        repair = await main.refine_generic_scene("abc123", main.GenericRefineRequest(iterations=1))
        assert repair["assembly_repair"]["prepared"] is True
        assert repair["status"]["generic_model"]["version"] == 4
        plan = load_feature_plan(root)
        assert [feature.status for feature in plan.features] == ["retry", "pending"]
        assert plan.active_feature_id is None
        assert not main._auto_improve_goal_reached(root, repair["status"])

        # The production autonomous loop must drive reinstallation and the original
        # finish retry itself, without a manual retry between these stages.
        await main._run_auto_improve("abc123", 6)
        finished = main._read_status(root)
        assert finished["generic_model"]["version"] == 7
        assert finished["auto_improve"]["state"] == "completed"
        plan = load_feature_plan(root)
        assert all(feature.status == "accepted" and feature.accepted_version == 7 for feature in plan.features)
        assert main._auto_improve_goal_reached(root, finished)

    asyncio.run(lifecycle())
    assert reviewed == [("finish", 6), ("part", 7), ("finish", 7)]
    assert (root / "scene" / "model-v6.blend").read_bytes() == b"preserved blend"
    assert not (root / "scene-spec-v7.json").exists()
