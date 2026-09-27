from app.main import GenericSceneSpec, _normalize_scene_spec_payload


def test_scene_normalizer_preserves_ai_geometry_choices():
    payload = {
        "title": "agent choice",
        "objects": [
            {
                "name": "primary mass",
                "shape": "wedge",
                "location": [0.0, 0.0, 1.0],
                "scale": [2.0, 3.0, 0.8],
                "rotation_deg": [0.0, 0.0, 0.0],
                "color": "#AABBCC",
            },
            {
                "name": "circular detail",
                "shape": "torus",
                "location": [1.0, 0.0, 0.5],
                "scale": [0.4, 0.2, 0.4],
                "rotation_deg": [90.0, 0.0, 0.0],
                "color": "#111111",
            },
        ],
    }

    normalized = _normalize_scene_spec_payload(payload, "fallback")
    spec = GenericSceneSpec.model_validate(normalized)

    assert spec.objects[0].shape == "wedge"
    assert spec.objects[1].shape == "torus"


def test_endpointless_connectors_use_generic_schema_fallback_not_name_heuristics():
    payload = {
        "title": "malformed connectors",
        "objects": [
            {
                "name": "anything at all",
                "shape": "rod",
                "location": [0.0, 0.0, 1.0],
                "scale": [2.0, 0.2, 0.4],
            },
            {
                "name": "another arbitrary part",
                "shape": "beam",
                "location": [0.0, 0.0, 2.0],
                "scale": [1.0, 2.0, 0.3],
            },
        ],
    }

    normalized = _normalize_scene_spec_payload(payload, "fallback")

    assert normalized["objects"][0]["shape"] == "cylinder"
    assert normalized["objects"][1]["shape"] == "cube"
    assert normalized["objects"][0]["scale"] == [2.0, 0.2, 0.4]
