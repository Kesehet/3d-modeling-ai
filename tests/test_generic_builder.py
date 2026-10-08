import pytest

from app.generic_builder import generic_scene_script
from app.main import GenericSceneSpec


def test_generic_scene_script_is_safe_declarative_builder():
    script = generic_scene_script()
    compile(script, "<generic-scene>", "exec")
    assert 'SPEC = args["spec"]' in script
    assert 'primitive_uv_sphere_add' in script
    assert 'primitive_cube_add' in script
    assert 'primitive_cylinder_add' in script
    assert 'primitive_cone_add' in script
    assert 'primitive_torus_add' in script
    assert 'stl_export' in script
    assert 'eval(' not in script
    assert 'exec(' not in script


def test_boolean_cutter_rejects_unsafe_implicit_unit_dimensions():
    base = {
        "title": "Explicitly sized Boolean test",
        "objects": [
            {"name": "rounded_body", "shape": "sphere", "location": [0, 0, 0], "radius": 0.3}
        ],
    }
    with pytest.raises(ValueError, match="Primitive boolean cutters require explicit dimensions"):
        GenericSceneSpec.model_validate({
            **base,
            "cutters": [{"name": "opening", "target": "rounded_body", "shape": "cylinder",
                         "location": [0, 0, 0]}],
        })
    for explicit_size in (
        {"radius": 0.05},
        {"dimensions": [0.2, 0.2, 1.0]},
        {"scale": [0.1, 0.1, 1.0]},
    ):
        spec = GenericSceneSpec.model_validate({
            **base,
            "cutters": [{"name": "opening", "target": "rounded_body", "shape": "cylinder",
                         "location": [0, 0, 0], **explicit_size}],
        })
        assert len(spec.cutters) == 1


def test_boolean_cutter_explicit_sphere_radius_and_mesh_are_valid():
    base = {"title": "Sized cutters", "objects": [
        {"name": "body", "shape": "cube", "location": [0, 0, 0]}
    ]}
    spec = GenericSceneSpec.model_validate({
        **base,
        "cutters": [{"name": "round_cutout", "target": "body", "shape": "sphere",
                     "location": [0, 0, 0], "radius": 0.12}],
    })
    assert spec.cutters[0].radius == 0.12
