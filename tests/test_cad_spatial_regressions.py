"""Behavioral and mutation regressions for assembly realism; no GPU or CAD libraries.

A valid JSON schema, six solids and a successful STEP export are insufficient:
plausible-looking but physically wrong assemblies must fail independently.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from app.cad_engine import CadBlueprint
from app.cad_spatial import evaluate_spatial_contract

# General radial mechanism with an upright centered shaft and independently
# positioned arms. This is a benchmark fixture, never a generation recipe.
ASSEMBLY = {
    "title": "Upright three-arm assembly",
    "parts": [
        {"name": "motor", "kind": "cylinder", "radius_mm": 30, "height_mm": 40},
        {"name": "hub", "kind": "cylinder", "radius_mm": 15, "height_mm": 20},
        {"name": "support_rod", "kind": "cylinder", "radius_mm": 10, "height_mm": 200,
         "position_mm": [0, 0, 120]},
        {"name": "radial_arm_1", "kind": "box", "dimensions_mm": [300, 30, 5],
         "position_mm": [165, 0, 0]},
        {"name": "radial_arm_2", "kind": "box", "dimensions_mm": [300, 30, 5],
         "position_mm": [-82.5, 142.894, 0], "rotation_deg": [0, 0, 120]},
        {"name": "radial_arm_3", "kind": "box", "dimensions_mm": [300, 30, 5],
         "position_mm": [-82.5, -142.894, 0], "rotation_deg": [0, 0, 240]},
    ],
}
CONTRACT = {
    "coaxial_terms": ("motor", "hub", "rod"),
    "vertical_support_term": "rod",
    "supported_term": "motor",
    "radial_term": "radial_arm",
    "radial_count": 3,
}


def _check(data):
    # Exercise the actual Pydantic schema and independent spatial review.
    return evaluate_spatial_contract(CadBlueprint.model_validate(data), **CONTRACT)


def _change(data, index, field, value):
    mutated = deepcopy(data)
    mutated["parts"][index][field] = value
    return mutated


def test_well_assembled_mechanism_has_measurable_contact_and_symmetry():
    verdict = _check(ASSEMBLY)
    assert verdict["pass"], verdict["failures"]
    assert len(verdict["measurements"]["radial_parts"]) == 3
    assert max(abs(v - 120) for v in verdict["measurements"]["angular_gaps_deg"]) < 0.1
    assert verdict["measurements"]["support"]["bottom_mm"] == 20


@pytest.mark.parametrize(("index", "field", "value", "diagnostic"), [
    (1, "position_mm", [0, 55, 0], "central axis"),
    (2, "position_mm", [0, 0, -115], "upward"),
    (2, "position_mm", [0, 0, 180], "end contact"),
    (2, "rotation_deg", [0, 85, 0], "upright"),
    (3, "position_mm", [215, 0, 0], "detached"),
    (4, "rotation_deg", [0, 0, 30], "radial direction"),
    (5, "position_mm", [-50, -50, 0], "evenly spaced"),
    (3, "dimensions_mm", [170, 30, 5], "inconsistent"),
    (4, "position_mm", [-82.5, 142.894, 90], "vertical overlap"),
])
def test_mutations_are_rejected_with_actionable_diagnostics(index, field, value, diagnostic):
    verdict = _check(_change(ASSEMBLY, index, field, value))
    assert not verdict["pass"], f"Undetected geometry mutation: {index}.{field}={value}"
    assert any(diagnostic in issue for issue in verdict["failures"]), verdict


def test_lost_radial_component_is_not_hidden_by_successful_other_parts():
    data = deepcopy(ASSEMBLY)
    data["parts"].pop()
    result = _check(data)
    assert not result["pass"]
    assert "Expected 3 radial parts, got 2" in result["failures"]


def test_axis_convention_and_translation_do_not_depend_on_world_origin():
    data = deepcopy(ASSEMBLY)
    for part in data["parts"]:
        x, y, z = part.get("position_mm", [0, 0, 0])
        part["position_mm"] = [x + 500, y - 350, z + 80]
    assert _check(data)["pass"], _check(data)["failures"]


def test_rotating_every_arm_together_keeps_valid_relative_arrangement():
    # Rotate the complete repeated assembly instead of memorizing one pose.
    import math

    data = deepcopy(ASSEMBLY)
    angle = math.radians(37)
    for part in data["parts"]:
        x, y, z = part.get("position_mm", [0, 0, 0])
        part["position_mm"] = [x * math.cos(angle) - y * math.sin(angle),
                               x * math.sin(angle) + y * math.cos(angle), z]
        rx, ry, rz = part.get("rotation_deg", [0, 0, 0])
        part["rotation_deg"] = [rx, ry, rz + 37]
    assert _check(data)["pass"], _check(data)["failures"]
