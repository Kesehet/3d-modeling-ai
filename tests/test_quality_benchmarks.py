from app.main import _normalize_benchmark_visual_payload
from app.quality import BENCHMARKS, evaluate_scene_spec_structural


def _objects(*names: str) -> dict:
    return {
        "objects": [
            {
                "name": name,
                "shape": "sphere",
                "location": [0, 0, 0],
                "scale": [1, 1, 1],
                "rotation_deg": [0, 0, 0],
                "color": "#808080",
            }
            for name in names
        ]
    }


def test_all_p0_benchmarks_are_defined():
    assert set(BENCHMARKS) == {
        "pikachu",
        "desk-lamp",
        "sneaker",
        "office-chair",
        "quadruped-robot",
    }


def test_pikachu_structural_gate_passes_complete_part_inventory():
    spec = _objects(
        "head",
        "body",
        "left ear",
        "right ear",
        "left eye",
        "right eye",
        "left cheek",
        "right cheek",
        "left arm",
        "right arm",
        "left foot",
        "right foot",
        "lightning tail",
    )
    result = evaluate_scene_spec_structural(spec, BENCHMARKS["pikachu"])
    assert result["passed"] is True
    assert result["missing_parts"] == []


def test_desk_lamp_gate_rejects_stacked_blob_without_neck_or_joint():
    spec = _objects(
        "round base",
        "vertical stem",
        "generic middle disc",
        "generic top dome",
        "decorative ring",
    )
    result = evaluate_scene_spec_structural(spec, BENCHMARKS["desk-lamp"])
    assert result["passed"] is False
    assert "neck" in result["missing_parts"]
    assert "joint" in result["missing_parts"]


def test_sneaker_gate_rejects_missing_laces():
    spec = _objects(
        "sole",
        "midsole",
        "toe box",
        "upper",
        "heel counter",
        "tongue",
        "foot opening collar",
        "side panel",
    )
    result = evaluate_scene_spec_structural(spec, BENCHMARKS["sneaker"])
    assert result["passed"] is False
    assert "laces" in result["missing_parts"]


def test_visual_normalizer_requires_every_exact_feature():
    requirements = ("round base", "angled neck", "dome shade")
    normalized = _normalize_benchmark_visual_payload(
        {
            "pass_benchmark": True,
            "recognizable": True,
            "required_features_visible": {
                "round base": True,
                "angled neck": True,
            },
        },
        requirements,
    )
    assert normalized["pass_benchmark"] is False
    assert normalized["required_features_visible"]["dome shade"] is False


def test_visual_normalizer_accepts_complete_visible_feature_set():
    requirements = ("seat", "backrest", "wheels")
    normalized = _normalize_benchmark_visual_payload(
        {
            "pass_benchmark": True,
            "recognizable": True,
            "required_features_visible": {
                "seat": True,
                "backrest": True,
                "wheels": True,
            },
            "major_failures": [],
        },
        requirements,
    )
    assert normalized["pass_benchmark"] is True
