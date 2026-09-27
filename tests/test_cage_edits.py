import pytest

from app.cage_edits import CageEditAction, apply_cage_edit_action


def _spec():
    return {
        "title": "generic object",
        "axis": "y",
        "stations": [
            {"position": -3.0, "profile": [[0, -1.0], [1.0, -0.6], [1.2, 0.4], [0.6, 1.0], [0, 1.2]]},
            {"position": -1.0, "profile": [[0, -1.0], [1.1, -0.6], [1.3, 0.5], [0.7, 1.1], [0, 1.3]]},
            {"position": 1.0, "profile": [[0, -1.0], [1.1, -0.6], [1.3, 0.5], [0.7, 1.1], [0, 1.3]]},
            {"position": 3.0, "profile": [[0, -1.0], [1.0, -0.6], [1.2, 0.4], [0.6, 1.0], [0, 1.2]]},
        ],
        "cutters": [
            {
                "name": "opening",
                "shape": "cube",
                "location": [0.0, -2.0, 0.0],
                "scale": [0.5, 0.3, 0.4],
                "rotation_deg": [0.0, 0.0, 0.0],
            }
        ],
    }


def test_reshape_station_is_relative_and_preserves_mirror_plane():
    action = CageEditAction(
        operation="reshape_station",
        target_index=1,
        width_scale=1.1,
        height_offset_fraction=-0.1,
        reason="The second section is too narrow and tall.",
    )

    edited = apply_cage_edit_action(_spec(), action)

    assert edited["stations"][1]["profile"][1][0] > 1.1
    assert edited["stations"][1]["profile"][0][0] == 0.0
    assert edited["stations"][1]["profile"][-1][0] == 0.0


def test_insert_station_adds_local_control_without_replacing_mesh():
    action = CageEditAction(
        operation="insert_station",
        target_index=1,
        insert_fraction=0.5,
        reason="The silhouette transition needs another control section.",
    )

    edited = apply_cage_edit_action(_spec(), action)

    assert len(edited["stations"]) == 5
    assert edited["stations"][2]["position"] == pytest.approx(0.0)


def test_adjust_cutter_uses_model_relative_deltas():
    action = CageEditAction(
        operation="adjust_cutter",
        target_index=0,
        location_delta_fraction=[0.1, 0.1, -0.1],
        scale_factor=[1.2, 0.8, 1.0],
        rotation_delta_deg=[0, 15, 0],
        reason="Move and rotate the existing opening.",
    )

    edited = apply_cage_edit_action(_spec(), action)
    cutter = edited["cutters"][0]

    assert cutter["location"] != [0.0, -2.0, 0.0]
    assert cutter["scale"][0] == pytest.approx(0.6)
    assert cutter["rotation_deg"][1] == pytest.approx(15.0)


def test_remove_station_refuses_to_destroy_minimum_cage():
    action = CageEditAction(
        operation="remove_station",
        target_index=1,
        reason="Try removing a section.",
    )

    with pytest.raises(ValueError, match="minimum cage"):
        apply_cage_edit_action(_spec(), action)


def test_position_edit_rejects_collapsed_sections():
    action = CageEditAction(
        operation="reshape_station",
        target_index=1,
        position_offset_fraction=0.33,
        reason="Move a station too close to the next one.",
    )

    with pytest.raises(ValueError, match="collapsed adjacent"):
        apply_cage_edit_action(_spec(), action)
