from app.main import (
    GenericRefineRequest,
    ModelingDirectorDecision,
    VisionAnalyzeRequest,
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
        (tmp_path / "renders" / name).write_bytes(name.encode("utf-8"))

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


def test_refinement_comparison_defaults_to_reject_when_unclear():
    normalized = _normalize_refinement_comparison_payload(
        {"summary": "Unclear result", "regressions": "Lost the tail"}
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
        (tmp_path / "renders" / name).write_bytes(name.encode("utf-8"))
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
