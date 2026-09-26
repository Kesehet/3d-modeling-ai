from app.main import (
    GenericRefineRequest,
    GenericSceneSpec,
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
