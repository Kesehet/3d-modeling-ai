from app.repair import print_repair_script


def test_print_repair_script_compiles_and_has_required_steps():
    script = print_repair_script()
    compile(script, "<print-repair>", "exec")
    assert "voxel_remesh" in script
    assert "remove_doubles" in script
    assert "recalc_face_normals" in script
    assert "connected_components" in script
    assert "stl_export" in script
