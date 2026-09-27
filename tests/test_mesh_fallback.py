from app.main import AdaptiveLoftSpec, _normalize_adaptive_loft_payload
from app.mesh_builder import adaptive_loft_script


def test_adaptive_loft_normalizer_accepts_safe_cross_sections_and_attachments():
    payload = {
        "title": "holdout reconstruction",
        "axis": "y",
        "color": "#AABBCC",
        "subdivision_levels": 1,
        "sections": [
            {
                "position": position,
                "contour": [
                    [-1.0, 0.0],
                    [-1.2, 0.4],
                    [-0.9, 1.0],
                    [-0.5, 1.4],
                    [0.5, 1.4],
                    [0.9, 1.0],
                    [1.2, 0.4],
                    [1.0, 0.0],
                ],
            }
            for position in (-2.0, -0.7, 0.7, 2.0)
        ],
        "attachments": [
            {
                "name": "wheel 1",
                "shape": "torus",
                "location": [-1.0, -1.0, 0.2],
                "scale": [0.4, 0.2, 0.4],
                "rotation_deg": [90.0, 0.0, 0.0],
                "color": "#111111",
            }
        ],
    }

    normalized = _normalize_adaptive_loft_payload(payload, "fallback")
    spec = AdaptiveLoftSpec.model_validate(normalized)

    assert spec.axis == "y"
    assert len(spec.sections) == 4
    assert all(len(section.contour) == 8 for section in spec.sections)
    assert spec.attachments[0].name == "wheel 1"


def test_adaptive_loft_builder_creates_continuous_mesh_and_standard_views():
    script = adaptive_loft_script()

    assert 'mesh.from_pydata(vertices, [], faces)' in script
    assert '"front-left"' in script
    assert '"back-right"' in script
    assert 'strategy": "adaptive_loft"' in script
    assert 'bpy.ops.wm.save_as_mainfile' in script
