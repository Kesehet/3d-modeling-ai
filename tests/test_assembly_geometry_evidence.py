"""The assembly planner only trusts geometry tied to the actual rejected build."""

import json

from app import main
from app.feature_tasks import FeaturePlan, FeatureTask, load_feature_plan, save_feature_plan


def test_rejected_assembly_includes_only_matching_blender_spatial_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOBS_ROOT", tmp_path / "jobs")
    root = main.JOBS_ROOT / "parent"
    (root / "exports").mkdir(parents=True)
    record = {
        "feature_id": "support",
        "candidate_version": 7,
        "baseline_version": 6,
        "child_job_id": "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
        "child_version": 2,
        "assembly": {"instances": [{"location": [1, 2, 3], "scale": [1, 1, 1]}]},
    }
    path = root / "component-assembly-v7.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    qa_path = root / "exports" / "model-v7-qa.json"
    qa = {
        "parent_source_blend": str(root / "scene" / "model-v6.blend"),
        "component_source_blend": str(main.JOBS_ROOT / "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee" / "scene" / "model-v2.blend"),
        "parent_geometry_preserved": True,
        "installed_instances": 1,
        "component_geometry_evidence": {
            "source_bounds_min_xyz": [-0.5, -0.5, -0.1],
            "source_bounds_max_xyz": [0.5, 0.5, 0.1],
            "projected_triangle_area_over_bbox_area": 1.9,
        },
        "component_instance_spatial_evidence": [{
            "instance_index": 1,
            "world_geometry_center_xyz": [1, 2, 3],
            "nearest_parent_aabb_contacts": [{"aabb_separation": 0.0}],
        }],
    }
    qa_path.write_text(json.dumps(qa), encoding="utf-8")
    result = main._last_rejected_component_assembly(root, "support", 6)
    assert result["candidate_version"] == 7
    assert result["geometry_evidence"]["frozen_child"]["projected_triangle_area_over_bbox_area"] == 1.9
    assert result["geometry_evidence"]["installed_instances"][0]["world_geometry_center_xyz"] == [1, 2, 3]

    # An unrelated frozen child, altered parent, or unpreserved assembly must
    # never be presented to the model as measured proof for this candidate.
    for field, wrong in (
        ("component_source_blend", str(main.JOBS_ROOT / "other" / "scene" / "model-v2.blend")),
        ("parent_source_blend", str(root / "scene" / "model-v4.blend")),
        ("parent_geometry_preserved", False),
        ("installed_instances", 2),
    ):
        tampered = {**qa, field: wrong}
        qa_path.write_text(json.dumps(tampered), encoding="utf-8")
        assert "geometry_evidence" not in main._last_rejected_component_assembly(root, "support", 6)
    assert main._last_rejected_component_assembly(root, "other-feature", 6) == {}


def test_repeated_rejected_frozen_child_gets_one_shape_rebuild(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOBS_ROOT", tmp_path / "jobs")
    parent_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    child_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    root = main.JOBS_ROOT / parent_id
    root.mkdir(parents=True)
    (root / "request.json").write_text(
        json.dumps({"prompt": "A generic multi-part object", "intended_use": "rendering"})
    )
    save_feature_plan(root, FeaturePlan(
        subject="Generic object",
        features=[
            FeatureTask(
                id="body", name="Body", build_mode="in_place",
                status="accepted", acceptance_verified=True, accepted_version=3,
            ),
            FeatureTask(
                id="support", name="Attached support", build_mode="component_job",
                component_job_id=child_id, status="failed", attempts=3,
            ),
        ],
    ))
    for version in [7, 8]:
        (root / f"component-assembly-v{version}.json").write_text(
            json.dumps({
                "baseline_version": 3, "candidate_version": version,
                "feature_id": "support", "child_job_id": child_id,
                "assembly": {"instances": [{"location": [0, 0, 0]}]},
            })
        )
        main.append_history(
            root, "component_assembly_rejected",
            feature_id="support", child_job_id=child_id, candidate_version=version,
            reason="The installed part remains a flat shape; missing structural geometry.",
        )
        if version == 7:
            assert main._schedule_bounded_component_shape_rebuild(root, "support", child_id) is False

    assert main._schedule_bounded_component_shape_rebuild(root, "support", child_id) is True
    plan = load_feature_plan(root)
    support = next(feature for feature in plan.features if feature.id == "support")
    body = next(feature for feature in plan.features if feature.id == "body")
    assert support.component_job_id is None
    assert support.status in {"retry", "ready"}
    assert support.attempts == 0
    assert body.status == "accepted" and body.accepted_version == 3
    assert main._schedule_bounded_component_shape_rebuild(root, "support", child_id) is False

    new_child_id = main._create_component_child_job(parent_id, support)
    assert new_child_id != child_id
    new_child = json.loads((main.JOBS_ROOT / new_child_id / "request.json").read_text())
    assert new_child["component_rebuild_of"] == child_id
    assert "earlier independently accepted frozen" in new_child["prompt"]
    assert "missing structural geometry" in new_child["prompt"]
    assert load_feature_plan(root).features[0].accepted_version == 3
