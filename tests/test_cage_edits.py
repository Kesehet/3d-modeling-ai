import pytest

from app.cage_edits import CageEditAction, apply_cage_edit_action
from app.main import _normalize_cage_edit_action_payload


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


def test_reshape_cage_proportions_changes_the_whole_blockout_envelope():
    spec = _spec()
    original_span = spec["stations"][-1]["position"] - spec["stations"][0]["position"]
    original_width = spec["stations"][1]["profile"][1][0]

    action = CageEditAction(
        operation="reshape_cage_proportions",
        length_scale=1.2,
        width_scale=1.1,
        height_scale=0.9,
        reason="The entire blockout is too short and narrow.",
    )

    edited = apply_cage_edit_action(spec, action)

    edited_span = edited["stations"][-1]["position"] - edited["stations"][0]["position"]
    assert edited_span == pytest.approx(original_span * 1.2)
    assert edited["stations"][1]["profile"][1][0] == pytest.approx(original_width * 1.1)
    assert all(station["profile"][0][0] == 0.0 for station in edited["stations"])
    assert all(station["profile"][-1][0] == 0.0 for station in edited["stations"])


def test_reshape_station_region_uses_proportional_falloff():
    spec = _spec()
    original = [
        station["profile"][1][0]
        for station in spec["stations"]
    ]
    action = CageEditAction(
        operation="reshape_station_region",
        target_index=1,
        influence_radius=1,
        width_scale=1.4,
        reason="The broad middle transition is too narrow.",
    )

    edited = apply_cage_edit_action(spec, action)
    widths = [station["profile"][1][0] for station in edited["stations"]]

    assert widths[1] == pytest.approx(original[1] * 1.4)
    assert widths[0] == pytest.approx(original[0] * 1.2)
    assert widths[2] == pytest.approx(original[2] * 1.2)
    assert widths[3] == pytest.approx(original[3])


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
    spec = _spec()
    for station, position in zip(
        spec["stations"],
        [-1.5, -0.3, 0.3, 1.5],
        strict=True,
    ):
        station["position"] = position

    action = CageEditAction(
        operation="reshape_station",
        target_index=1,
        position_offset_fraction=0.2,
        reason="Move a station too close to the next one.",
    )

    with pytest.raises(ValueError, match="collapsed adjacent"):
        apply_cage_edit_action(spec, action)



def test_visual_edit_payload_accepts_action_alias_from_vision_model():
    payload = _normalize_cage_edit_action_payload(
        {
            "action": "replan_representation",
            "summary": "The current representation is too coarse to refine locally.",
            "expected_improvement": "Rebuild a more editable baseline before local corrections.",
        }
    )

    action = CageEditAction.model_validate(payload)

    assert action.operation == "replan_representation"
    assert "too coarse" in action.reason
    assert "editable baseline" in action.expected_visual_effect


def test_attachment_lifecycle_preserves_unrelated_geometry():
    original = _spec()
    added = apply_cage_edit_action(original, CageEditAction(
        operation="add_attachment", reason="Add a missing independent part",
        part={"name": "part", "shape": "cube", "location": [0, 1, 1], "scale": [0.3, 0.2, 0.1]},
    ))
    moved = apply_cage_edit_action(added, CageEditAction(
        operation="adjust_attachment", target_index=0, location_delta_fraction=[0, 0, 0.1],
        reason="Align the part with its mating surface",
    ))
    assert added["attachments"][0]["location"] == [0, 1, 1]
    assert moved["attachments"][0]["location"][2] > 1
    removed = apply_cage_edit_action(moved, CageEditAction(
        operation="remove_attachment", target_index=0, reason="Remove the incorrect part",
    ))
    assert removed["attachments"] == []
    assert removed["stations"] == original["stations"]
    assert removed["cutters"] == original["cutters"]
    assert "attachments" not in original


def test_surface_edit_and_noop_rejection():
    spec = {**_spec(), "subdivision_levels": 2, "bevel_width": 0.03, "smooth": True}
    action = CageEditAction(operation="set_surface", subdivision_levels=0, reason="Preserve sharp edges")
    edited = apply_cage_edit_action(spec, action)
    assert edited["subdivision_levels"] == 0
    assert edited["stations"] == spec["stations"]
    with pytest.raises(ValueError, match="does not change"):
        apply_cage_edit_action(edited, action)


def test_new_part_rejects_nonfinite_and_oversized_geometry():
    with pytest.raises(ValueError):
        CageEditAction(operation="adjust_cutter", reason="bad", scale_factor=[float("nan"), 1, 1])
    action = CageEditAction(operation="add_cutter", reason="bad",
                           part={"name": "hole", "shape": "cube", "location": [90, 0, 0], "scale": [1, 1, 1]})
    with pytest.raises(ValueError, match="safe modeling volume"):
        apply_cage_edit_action(_spec(), action)
