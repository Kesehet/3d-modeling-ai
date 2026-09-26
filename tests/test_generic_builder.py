from app.generic_builder import generic_scene_script


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
