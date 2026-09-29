import asyncio
import json

import pytest
from pydantic import ValidationError

from app import main
from app.cage_edits import CageEditAction, apply_cage_edit_action
from app.feature_tasks import FeaturePlan, FeatureTask, load_feature_plan, save_feature_plan


def test_full_dimensions_and_parametric_geometry_survive_planning_normalization():
    objects = [
        {"name": "block", "shape": "cube", "location": [0, 0, 0], "dimensions": [2, 3, 4]},
        {"name": "hollow", "shape": "lathe", "location": [0, 0, 0],
         "profile": [[1, 0], [1, 2], [.8, 2], [.8, 0]], "segments": 48},
        {"name": "curved", "shape": "sweep", "location": [0, 0, 0],
         "path": [[0, 0, 0], [1, 0, 1], [0, 0, 2]], "radius": .1},
        {"name": "legacy", "shape": "cube", "location": [0, 0, 0], "scale": [.2, .3, .4]},
    ]
    scene = main.GenericSceneSpec.model_validate(main._normalize_scene_spec_payload({"objects": objects}, "test"))
    assert scene.objects[0].dimensions == [2, 3, 4]
    assert scene.objects[1].profile == [(1, 0), (1, 2), (.8, 2), (.8, 0)]
    assert scene.objects[1].segments == 48
    assert scene.objects[2].path == [(0, 0, 0), (1, 0, 1), (0, 0, 2)]
    assert scene.objects[2].radius == .1
    assert scene.objects[3].scale == [.2, .3, .4]
    assert scene.objects[3].dimensions is None


@pytest.mark.parametrize("part", [
    {"shape": "lathe", "profile": [[0, 0], [1, 1], [2, 2]]},
    {"shape": "lathe", "profile": [[-1, 0], [1, 1], [1, 0]]},
    {"shape": "sweep", "path": [[0, 0, 0], [1, 1, 1]]},
    {"shape": "sweep", "path": [[0, 0, 0], [0, 0, 0]], "radius": .1},
    {"shape": "cube", "dimensions": [1, 0, 2]},
])
def test_invalid_parametric_geometry_rejected_before_render(part):
    with pytest.raises(ValidationError):
        main.SceneObjectSpec(name="invalid", location=[0, 0, 0], **part)


def test_cage_edits_resize_dimensions_and_convert_cutter_units():
    spec = {"axis": "y", "stations": [
        {"position": p, "profile": [[0, 0], [1, 0], [1, 1], [0, 1]]} for p in (-2, -1, 1, 2)
    ], "attachments": [{"name": "part", "shape": "cube", "location": [0,0,0], "dimensions": [1,2,3]}]}
    resized = apply_cage_edit_action(spec, CageEditAction(
        operation="adjust_attachment", reason="resize", target_index=0, scale_factor=[1.2,1,1],
    ))
    assert resized["attachments"][0]["dimensions"] == [1.2,2,3]
    cut = apply_cage_edit_action(spec, CageEditAction(
        operation="add_cutter", reason="opening", part={
            "name": "cut", "shape": "cylinder", "location": [0,0,0], "dimensions": [1,2,3],
        },
    ))
    assert cut["cutters"][0]["scale"] == [.5,1,1.5]


def setup_job(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOBS_ROOT", tmp_path)
    root = tmp_path / "abc123"
    root.mkdir()
    (root / "request.json").write_text(json.dumps({"prompt": "An assembly"}))
    feature = FeatureTask(id="primary", name="Main mass", strategy="base_mesh_region", status="running")
    save_feature_plan(root, FeaturePlan(features=[feature, FeatureTask(
        id="secondary", name="Secondary part", depends_on=["primary"],
    )], active_feature_id="primary"))
    return root, feature


def evaluation(passed):
    return {"passed": passed, "visible": True, "criteria_satisfied": passed,
            "confidence": .99, "reference_match_score": .9 if passed else .6,
            "regression_detected": False, "subject_recognizable": False,
            "summary": "Local feature review", "model": "vision-test"}


@pytest.mark.parametrize("passed,better,kept,accepted", [
    (False, True, True, False), (True, True, True, True), (True, False, False, False),
])
def test_procedural_progress_completion_and_regression_are_separate(
    tmp_path, monkeypatch, passed, better, kept, accepted,
):
    root, feature = setup_job(tmp_path, monkeypatch)
    previous = main._write_status(root, generic_model={"version": 7}, modeling_strategy="procedural",
                                  quality_gate={"summary": "Preserved quality"})

    async def compare(*a, **kw):
        assert kw == {"baseline_version": 7, "candidate_version": 8}
        return {"candidate_is_better": better}

    async def whole(*a, **kw):
        raise AssertionError("Later planned parts must be built before final whole-object review")

    monkeypatch.setattr(main, "_compare_generic_versions", compare)
    monkeypatch.setattr(main, "_generic_recognizability_check", whole)
    result = asyncio.run(main._review_procedural_candidate(
        "abc123", {"candidate_model": {"version": 8}}, previous, feature, evaluation=evaluation(passed),
    ))
    assert result["kept"] is kept
    assert result["accepted"] is accepted
    status = main._read_status(root)
    assert status["generic_model"]["version"] == (8 if kept else 7)
    plan = load_feature_plan(root)
    assert (plan.features[0].status == "accepted") is accepted
    if not kept:
        assert status["quality_gate"] == previous["quality_gate"]


def test_finished_existing_feature_advances_without_rebuilding(tmp_path, monkeypatch):
    root, feature = setup_job(tmp_path, monkeypatch)
    previous = main._write_status(root, generic_model={"version": 4}, modeling_strategy="procedural")

    async def compare(*a, **kw):
        raise AssertionError("The same model must not be compared against itself")

    monkeypatch.setattr(main, "_compare_generic_versions", compare)
    result = asyncio.run(main._review_procedural_candidate(
        "abc123", {"candidate_model": {"version": 4}}, previous, feature, evaluation=evaluation(True),
    ))
    assert result["accepted"] is True
    plan = load_feature_plan(root)
    assert plan.features[0].accepted_version == 4
    assert plan.features[1].status == "ready"
