import pytest
from PIL import Image

from app.main import (
    GenericRefineRequest,
    ModelingDirectorDecision,
    VisionAnalyzeRequest,
    _catastrophic_visual_failure,
    _collect_images,
    _normalize_refinement_comparison_payload,
)


def test_generic_refine_request_bounds_iterations():
    assert GenericRefineRequest(iterations=1).iterations == 1


def test_ai_director_controls_refinement_action():
    decision = ModelingDirectorDecision(
        action="revise_procedural",
        subject_match_score=0.45,
        summary="Keep the strategy but rebuild the visible proportions.",
        instructions=["Increase the main mass", "Correct the silhouette"],
    )

    assert decision.action == "revise_procedural"
    assert len(decision.instructions) == 2


def test_generic_visual_refinement_collects_only_latest_model_version(tmp_path):
    (tmp_path / "references").mkdir()
    (tmp_path / "renders").mkdir()
    for name in (
        "model-v1-front.png",
        "model-v1-back.png",
        "model-v2-front.png",
        "model-v2-back.png",
        "pikachu-front.png",
    ):
        Image.new("RGB", (32, 32), (120, 130, 140)).save(tmp_path / "renders" / name)

    _, labels = _collect_images(
        tmp_path,
        VisionAnalyzeRequest(
            stage="generic_visual_refinement",
            include_references=False,
            include_renders=True,
            max_images=16,
        ),
    )
    assert labels
    assert all("model-v2-" in label for label in labels)


def test_refinement_comparison_requires_explicit_judgment_for_fallback():
    with pytest.raises(ValueError, match="explicit keep/revert judgment"):
        _normalize_refinement_comparison_payload(
            {"summary": "Unclear result", "regressions": "Lost the tail"}
        )


def test_refinement_comparison_preserves_explicit_rejection():
    normalized = _normalize_refinement_comparison_payload(
        {"candidate_is_better": False, "summary": "Worse silhouette", "regressions": "Lost the tail"}
    )
    assert normalized["candidate_is_better"] is False
    assert normalized["regressions"] == ["Lost the tail"]


def test_generic_visual_refinement_prefers_accepted_status_version(tmp_path):
    (tmp_path / "references").mkdir()
    (tmp_path / "renders").mkdir()
    for name in (
        "model-v1-front.png",
        "model-v1-back.png",
        "model-v2-front.png",
        "model-v2-back.png",
    ):
        Image.new("RGB", (32, 32), (120, 130, 140)).save(tmp_path / "renders" / name)
    (tmp_path / "status.json").write_text(
        '{"generic_model":{"version":1}}',
        encoding="utf-8",
    )

    _, labels = _collect_images(
        tmp_path,
        VisionAnalyzeRequest(
            stage="generic_visual_refinement",
            include_references=False,
            include_renders=True,
            max_images=16,
        ),
    )
    assert labels
    assert all("model-v1-" in label for label in labels)


def test_catastrophic_visual_failure_switches_representation_immediately():
    severe, score = _catastrophic_visual_failure(
        {"recognizable": False, "subject_match_score": 0.0}
    )
    assert severe is True
    assert score == 0.0


def test_marginal_visual_failure_allows_one_same_strategy_retry():
    severe, score = _catastrophic_visual_failure(
        {"recognizable": False, "subject_match_score": 0.45}
    )
    assert severe is False
    assert score == 0.45


def test_scoped_parent_repair_preserves_omitted_and_rewritten_locked_parts():
    from app.main import GenericSceneSpec, _scope_parent_scene_revision

    current = GenericSceneSpec.model_validate({
        "title": "Parent", "presentation_base": False,
        "objects": [
            {"name": "body", "shape": "cube", "location": [0,0,0], "dimensions": [2,2,2]},
            {"name": "support", "shape": "cylinder", "location": [0,0,2], "dimensions": [.2,.2,2]},
            {"name": "trim", "shape": "cube", "location": [0,0,-1], "dimensions": [1,1,.1]},
        ],
        "cutters": [{"name": "opening", "target": "body", "shape": "cube",
                     "location": [0,0,0], "dimensions": [.5,.5,3]}],
    })
    proposed = {
        "title": "Repair", "presentation_base": True,
        "objects": [
            {"name": "body", "shape": "sphere", "location": [10,0,0], "dimensions": [10,10,10]},
            {"name": "support", "shape": "cylinder", "location": [0,0,1.5], "dimensions": [.3,.3,1]},
            {"name": "contact", "shape": "cube", "location": [0,0,1], "dimensions": [.4,.4,.2]},
        ], "cutters": [],
    }
    result = _scope_parent_scene_revision(current, proposed, ["support"])
    by_name = {obj.name: obj for obj in result.objects}
    assert by_name["body"] == current.objects[0]  # Malicious/accidental rewrite discarded.
    assert by_name["trim"] == current.objects[2]  # Omission never deletes accepted parts.
    assert by_name["support"].dimensions == [.3,.3,1]
    assert "contact" in by_name
    assert result.cutters == current.cutters
    assert result.presentation_base is False
    removed = _scope_parent_scene_revision(current, {"title":"Removal", "objects":[]}, ["support"])
    assert {obj.name for obj in removed.objects} == {"body", "trim"}
    for invalid in (None, [], ["unknown"], [12]):
        with pytest.raises(ValueError, match="editable_object_names"):
            _scope_parent_scene_revision(current, proposed, invalid)
    assert current.objects[0].dimensions == [2,2,2]
