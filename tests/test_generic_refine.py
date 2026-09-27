from app.main import (
    GenericRefineRequest,
    GenericSceneSpec,
    VisionAnalyzeRequest,
    _collect_images,
    _normalize_refinement_comparison_payload,
    _scene_spec_regression_reasons,
)


def test_generic_refine_request_bounds_iterations():
    assert GenericRefineRequest(iterations=1).iterations == 1


def _spec(names: list[str], *, rods: int = 0) -> GenericSceneSpec:
    objects = []
    for index, name in enumerate(names):
        shape = "rod" if index < rods else "sphere"
        item = {
            "name": name,
            "shape": shape,
            "location": [0.0, 0.0, float(index)],
            "scale": [1.0, 1.0, 1.0],
            "rotation_deg": [0.0, 0.0, 0.0],
            "color": "#808080",
        }
        if shape == "rod":
            item["start"] = [0.0, 0.0, float(index)]
            item["end"] = [0.0, 0.0, float(index + 1)]
            item["radius"] = 0.2
        objects.append(item)
    return GenericSceneSpec(title="test", objects=objects)


def test_regression_guard_rejects_object_count_collapse():
    current = _spec([
        "body", "head", "left ear", "right ear", "left eye",
        "right eye", "left cheek", "right cheek", "tail", "feet",
    ])
    revised = _spec(["body", "head", "antenna one", "antenna two"])
    reasons = _scene_spec_regression_reasons(current, revised)
    assert any("object count collapsed" in reason for reason in reasons)


def test_regression_guard_rejects_semantic_part_loss():
    current = _spec([
        "lamp base", "vertical stem", "angled neck", "neck joint",
        "dome shade", "bulb housing",
    ])
    revised = _spec([
        "lamp base", "vertical stem", "generic disc", "generic top",
        "generic ring", "generic support",
    ])
    reasons = _scene_spec_regression_reasons(current, revised)
    assert any("semantic part coverage regressed" in reason for reason in reasons)


def test_regression_guard_allows_surgical_changes():
    current = _spec([
        "lamp base", "vertical stem", "angled neck", "neck joint",
        "dome shade", "bulb housing",
    ], rods=2)
    revised = _spec([
        "lamp base", "vertical stem", "angled neck", "neck joint",
        "dome shade", "bulb housing", "shade rim",
    ], rods=2)
    assert _scene_spec_regression_reasons(current, revised) == []



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
