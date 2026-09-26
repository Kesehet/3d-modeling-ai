from app.builders import pikachu_script


def test_pikachu_script_uses_fast_workbench_and_real_newlines():
    script = pikachu_script()
    assert 'scene.render.engine = "BLENDER_WORKBENCH"' in script
    assert '\\nscene.display.shading' not in script
    compile(script, "<pikachu-script>", "exec")


def test_pikachu_script_has_parameter_and_export_hooks():
    script = pikachu_script()
    assert 'args.get("params")' in script
    assert 'export_scene.gltf' in script
    assert 'obj_export' in script
    assert 'stl_export' in script
    assert '"non_manifold_edges"' in script
