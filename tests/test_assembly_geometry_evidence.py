"""The assembly planner only trusts geometry tied to the actual rejected build."""

import json

from app import main


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
