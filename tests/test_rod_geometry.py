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


def test_endpointless_rod_does_not_collapse_broad_body_into_default_stick():
    raw = {
        "title": "Vehicle blockout",
        "objects": [
            {
                "name": "main body lower",
                "shape": "rod",
                "location": [0, 0, 0.6],
                "scale": [1.8, 4.2, 0.8],
                "color": "silver",
            },
            {
                "name": "wheel front left",
                "shape": "rod",
                "location": [-0.9, -1.3, 0.4],
                "scale": [0.3, 0.6, 0.6],
                "color": "black",
            },
        ],
    }

    normalized = _normalize_scene_spec_payload(raw, "Vehicle blockout")

    assert normalized["objects"][0]["shape"] == "cube"
    assert normalized["objects"][0]["start"] is None
    assert normalized["objects"][0]["end"] is None
    assert normalized["objects"][1]["shape"] == "torus"


def test_beam_alias_remains_beam_when_endpoints_exist():
    raw = {
        "title": "Beam",
        "objects": [
            {
                "name": "support beam",
                "shape": "beam",
                "start": [0, 0, 0],
                "end": [2, 0, 1],
                "radius": 0.2,
            }
        ],
    }

    normalized = _normalize_scene_spec_payload(raw, "Beam")
    assert normalized["objects"][0]["shape"] == "beam"
