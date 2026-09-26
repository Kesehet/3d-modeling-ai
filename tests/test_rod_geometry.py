from app.generic_builder import generic_scene_script
from app.main import SceneObjectSpec, _normalize_scene_spec_payload


def test_scene_spec_supports_safe_rod():
    raw = {
        "title": "Robot leg",
        "objects": [
            {
                "name": "shin",
                "shape": "rod",
                "start": [0, 0, 1],
                "end": [1, 0, 0],
                "radius": 0.15,
                "color": "darkgray",
            }
        ],
    }
    normalized = _normalize_scene_spec_payload(raw, "Robot leg")
    spec = SceneObjectSpec.model_validate(normalized["objects"][0])
    assert spec.shape == "rod"
    assert spec.start == [0.0, 0.0, 1.0]
    assert spec.end == [1.0, 0.0, 0.0]
    assert spec.radius == 0.15


def test_generic_builder_aligns_rods_between_points():
    script = generic_scene_script()
    assert 'if shape == "rod"' in script
    assert 'delta.to_track_quat("Z", "Y")' in script
