"""Generic part-graph gate: no car, fan, or furniture-specific geometry."""
import pytest

from app.cad_assembly_qa import evaluate_assembly_bounds


def box(x0, x1, y0=-5, y1=5, z0=-5, z1=5):
    return [[x0, y0, z0], [x1, y1, z1]]


def test_connected_three_part_chain_passes_without_name_templates():
    parts = {
        "nucleus": box(-10, 10),
        "branch_A": box(8, 48),
        "branch_B": box(-48, -8),
    }
    verdict = evaluate_assembly_bounds(parts)
    assert verdict["pass"]
    assert len(verdict["groups"]) == 1
    assert not verdict["contact_verified"]  # AABB proximity != mesh contact.


def test_disconnected_artifacts_expose_measured_gaps():
    parts = {
        "middle": box(-10, 10),
        "floating_1": box(120, 155),
        "floating_2": box(-155, -120),
        "remote": box(500, 510),
    }
    verdict = evaluate_assembly_bounds(parts)
    assert not verdict["pass"]
    assert len(verdict["groups"]) == 4
    assert any("Disconnected" in item for item in verdict["failures"])
    assert any("mm" in item for item in verdict["failures"])
    assert verdict["measurement"] == "aabb_only"


def test_explicitly_independent_assemblies_do_not_need_contact():
    verdict = evaluate_assembly_bounds(
        {"a": box(0, 1), "b": box(1000, 1001)},
        assembly_mode="independent",
    )
    assert verdict["pass"] and len(verdict["groups"]) == 2


def test_reject_non_finite_or_inverted_geometry():
    with pytest.raises(ValueError):
        evaluate_assembly_bounds({"bad": [[0, 0, 0], [float("nan"), 1, 1]]})
    with pytest.raises(ValueError):
        evaluate_assembly_bounds({"bad": [[4, 0, 0], [1, 1, 1]]})
