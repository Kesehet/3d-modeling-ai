"""Unmocked kernel/export regression: real BReps, triangles, poses and defects.

The language model is deliberately mocked only at its response boundary. These
checks execute in the dedicated CAD CI image with real OpenCascade installed.
"""
from __future__ import annotations

import copy
import json
import math

import pytest
from test_cad_spatial_regressions import ASSEMBLY, CONTRACT

from app.cad_engine import CadBlueprint, build_cad
from app.cad_spatial import evaluate_spatial_contract

build123d = pytest.importorskip("build123d")
trimesh = pytest.importorskip("trimesh")


def _geometry(plan, directory):
    result = build_cad(CadBlueprint.model_validate(plan), directory)
    scene = trimesh.load(directory / "assembly.glb", force="scene")
    step = build123d.import_step(str(directory / "assembly.step"))
    manifest = json.loads((directory / "manifest.json").read_text())
    return result, scene, step, manifest


def test_full_mechanism_preserves_real_world_poses_and_part_volume(tmp_path):
    result, scene, step, manifest = _geometry(ASSEMBLY, tmp_path)
    expected_centers = [tuple(p.get("position_mm", (0, 0, 0))) for p in ASSEMBLY["parts"]]
    actual_meshes = list(scene.dump())  # world transforms applied, not just GLB node counts
    assert len(actual_meshes) == len(expected_centers)
    unmatched = expected_centers[:]
    for mesh in actual_meshes:
        assert mesh.is_watertight and mesh.is_winding_consistent
        assert mesh.volume > 0
        center = (mesh.bounds[0] + mesh.bounds[1]) / 2
        closest = min(unmatched, key=lambda value: math.dist(center, value))
        assert math.dist(center, closest) < 1.5, (center, closest)
        unmatched.remove(closest)
    assert not unmatched
    assert step.is_valid and step.volume > 0
    assert scene.bounds[1, 2] > 210  # elevated support survives STEP -> GLB
    assert scene.extents[0] > 400 and scene.extents[1] > 400  # radially substantial
    assert result["part_count"] == len(ASSEMBLY["parts"])
    original_parts = CadBlueprint.model_validate(ASSEMBLY).model_dump(mode="json")["parts"]
    assert manifest["blueprint"]["parts"] == original_parts


def test_hollow_bore_survives_true_cad_step_and_glb_round_trip(tmp_path):
    r_outer, r_inner, height = 30.0, 12.0, 60.0
    plan = {"title": "Bored spacer", "parts": [{"name": "bored_spacer", "kind": "tube",
            "radius_mm": r_outer, "inner_radius_mm": r_inner, "height_mm": height}]}
    result, scene, step, _ = _geometry(plan, tmp_path)
    expected_volume = math.pi * (r_outer ** 2 - r_inner ** 2) * height
    assert step.volume == pytest.approx(expected_volume, rel=0.01)
    mesh = next(iter(scene.dump()))
    assert mesh.is_watertight
    assert abs(mesh.volume - expected_volume) / expected_volume < 0.035
    assert result["parts"][0]["volume_mm3"] == pytest.approx(expected_volume, rel=0.01)


def test_a_valid_3d_export_is_not_equivalent_to_correct_assembly(tmp_path):
    """Kernel passes a bad support even though the benchmark blocks promotion."""
    plan = copy.deepcopy(ASSEMBLY)
    plan["parts"][2]["position_mm"] = [0, 0, -115]  # support physically below motor
    result, scene, step, _ = _geometry(plan, tmp_path)
    assert result["step_reimport_valid"] and step.is_valid
    assert len(scene.geometry) == len(plan["parts"])
    verdict = evaluate_spatial_contract(CadBlueprint.model_validate(plan), **CONTRACT)
    assert not verdict["pass"]
    assert any("upward" in problem for problem in verdict["failures"])


def test_real_glb_evidence_detects_missing_geometry_after_export(tmp_path):
    """Prove the checker observes actual file contents, not a success flag."""
    _, scene, _, _ = _geometry(ASSEMBLY, tmp_path)
    original_mesh_count = len(scene.dump())
    incomplete = trimesh.Scene()
    for index, mesh in enumerate(scene.dump()):
        if index < original_mesh_count - 1:
            incomplete.add_geometry(mesh)
    altered = tmp_path / "incomplete.glb"
    altered.write_bytes(incomplete.export(file_type="glb"))
    imported = trimesh.load(altered, force="scene")
    assert len(imported.dump()) == original_mesh_count - 1
    assert len(imported.dump()) != len(ASSEMBLY["parts"])
