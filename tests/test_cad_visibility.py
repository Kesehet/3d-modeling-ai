"""Real mesh visibility checks: QA must notice hidden solids, not just count parts."""
from __future__ import annotations

import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("numpy")

from app.cad_preview import CAMERAS, render_cad_views


def _scene(parts):
    scene = trimesh.Scene()
    for name, extents, center in parts:
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    return scene


def test_enclosed_part_cannot_claim_screen_space_presence(tmp_path):
    scene = _scene([
        ("outer_shell", (100, 100, 100), (0, 0, 0)),
        ("hidden_core", (20, 20, 20), (0, 0, 0)),
    ])
    visibility = {}
    renders = render_cad_views(scene, tmp_path, visibility_out=visibility)

    assert set(renders) == set(CAMERAS)
    assert all((tmp_path / image).stat().st_size > 100 for image in renders.values())
    assert visibility["measurement"] == "five_view_software_zbuffer"
    assert visibility["parts"]["outer_shell"]["max_visible_pixels"] > 0
    assert visibility["parts"]["hidden_core"]["max_projected_pixels"] > 0
    assert visibility["parts"]["hidden_core"]["max_visible_pixels"] == 0
    assert visibility["parts"]["hidden_core"]["views_visible"] == 0
    assert visibility["parts"]["hidden_core"]["max_visible_fraction"] == 0


def test_separated_parts_remain_visible_without_subject_templates(tmp_path):
    scene = _scene([
        ("first", (20, 24, 20), (-65, 0, 0)),
        ("second", (24, 20, 20), (65, 0, 0)),
    ])
    visibility = {}
    render_cad_views(scene, tmp_path, visibility_out=visibility)
    for key in ("first", "second"):
        assert visibility["parts"][key]["max_visible_pixels"] > 100
        assert visibility["parts"][key]["views_visible"] >= 2
        assert 0 < visibility["parts"][key]["max_visible_fraction"] <= 1


def test_visibility_is_diagnostic_not_an_automatic_rejection(tmp_path):
    scene = _scene([("internal_support", (8, 8, 8), (0, 0, 0))])
    assert set(render_cad_views(scene, tmp_path)) == set(CAMERAS)
    visibility = {}
    render_cad_views(scene, tmp_path, visibility_out=visibility)
    assert "limitations" in visibility
    assert visibility["parts"]["internal_support"]["views_visible"] == len(CAMERAS)
