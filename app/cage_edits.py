"""Generic bounded edits for iterative hard-surface cage refinement.

The vision/LLM layer chooses an intent-level edit. This module translates that
intent into deterministic changes to the current cage specification. It contains
no object-family knowledge.
"""

from __future__ import annotations

import copy
import math
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, Field


class CageEditAction(BaseModel):
    operation: Literal[
        "reshape_station",
        "reshape_profile_point",
        "insert_station",
        "remove_station",
        "adjust_cutter",
        "remove_cutter",
        "replan_representation",
    ]
    reason: str = Field(min_length=1, max_length=1600)
    expected_visual_effect: str = Field(default="", max_length=1200)
    focus_views: list[str] = Field(default_factory=list, max_length=6)

    target_index: int | None = Field(default=None, ge=0, le=64)
    point_index: int | None = Field(default=None, ge=0, le=32)

    width_scale: float = Field(default=1.0, ge=0.65, le=1.45)
    height_scale: float = Field(default=1.0, ge=0.65, le=1.45)
    height_offset_fraction: float = Field(default=0.0, ge=-0.30, le=0.30)
    position_offset_fraction: float = Field(default=0.0, ge=-0.20, le=0.20)

    width_offset_fraction: float = Field(default=0.0, ge=-0.30, le=0.30)
    point_height_offset_fraction: float = Field(default=0.0, ge=-0.30, le=0.30)

    insert_fraction: float = Field(default=0.5, ge=0.15, le=0.85)

    location_delta_fraction: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        min_length=3,
        max_length=3,
    )
    scale_factor: list[float] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0],
        min_length=3,
        max_length=3,
    )
    rotation_delta_deg: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        min_length=3,
        max_length=3,
    )


def _extents(spec: dict) -> tuple[float, float, float, float, float]:
    stations = spec.get("stations") or []
    if len(stations) < 2:
        raise ValueError("Cage edit needs at least two stations.")

    positions = [float(station["position"]) for station in stations]
    points = [
        pair
        for station in stations
        for pair in (station.get("profile") or [])
        if isinstance(pair, (list, tuple)) and len(pair) >= 2
    ]
    if not points:
        raise ValueError("Cage edit needs profile points.")

    axis_span = max(positions) - min(positions)
    max_half_width = max(float(pair[0]) for pair in points)
    min_z = min(float(pair[1]) for pair in points)
    max_z = max(float(pair[1]) for pair in points)
    height_span = max_z - min_z
    if axis_span <= 0.0 or max_half_width <= 0.0 or height_span <= 0.0:
        raise ValueError("Cage edit cannot operate on degenerate geometry.")
    return axis_span, max_half_width, min_z, max_z, height_span


def _require_index(items: list, index: int | None, label: str) -> int:
    if index is None or index < 0 or index >= len(items):
        raise ValueError(f"{label} index is missing or out of range.")
    return index


def apply_cage_edit_action(spec: dict, action: CageEditAction) -> dict:
    """Return a new cage spec with exactly one bounded edit applied."""

    updated = copy.deepcopy(spec)
    stations = updated.get("stations") or []
    cutters = updated.get("cutters") or []
    axis_span, max_half_width, min_z, max_z, height_span = _extents(updated)
    center_z = (min_z + max_z) / 2.0

    if action.operation == "reshape_station":
        index = _require_index(stations, action.target_index, "station")
        station = stations[index]
        profile = station.get("profile") or []
        if len(profile) < 4:
            raise ValueError("Target station does not have a usable profile.")

        station["position"] = float(station["position"]) + action.position_offset_fraction * axis_span
        for point_index, point in enumerate(profile):
            width = max(0.0, float(point[0]) * action.width_scale)
            height = (
                (float(point[1]) - center_z) * action.height_scale
                + center_z
                + action.height_offset_fraction * height_span
            )
            if point_index in {0, len(profile) - 1}:
                width = 0.0
            profile[point_index] = [width, height]

    elif action.operation == "reshape_profile_point":
        station_index = _require_index(stations, action.target_index, "station")
        profile = stations[station_index].get("profile") or []
        point_index = _require_index(profile, action.point_index, "profile point")
        width = max(
            0.0,
            float(profile[point_index][0])
            + action.width_offset_fraction * max_half_width,
        )
        height = (
            float(profile[point_index][1])
            + action.point_height_offset_fraction * height_span
        )
        if point_index in {0, len(profile) - 1}:
            width = 0.0
        profile[point_index] = [width, height]

    elif action.operation == "insert_station":
        left_index = _require_index(stations, action.target_index, "left station")
        if left_index >= len(stations) - 1:
            raise ValueError("A station can only be inserted between existing stations.")
        if len(stations) >= 16:
            raise ValueError("Cage already has the maximum number of stations.")

        left = stations[left_index]
        right = stations[left_index + 1]
        left_profile = left.get("profile") or []
        right_profile = right.get("profile") or []
        if len(left_profile) != len(right_profile) or len(left_profile) < 4:
            raise ValueError("Adjacent profiles are incompatible for interpolation.")

        t = action.insert_fraction
        stations.insert(
            left_index + 1,
            {
                "position": float(left["position"]) * (1.0 - t)
                + float(right["position"]) * t,
                "profile": [
                    [
                        float(a[0]) * (1.0 - t) + float(b[0]) * t,
                        float(a[1]) * (1.0 - t) + float(b[1]) * t,
                    ]
                    for a, b in zip(left_profile, right_profile)
                ],
            },
        )

    elif action.operation == "remove_station":
        index = _require_index(stations, action.target_index, "station")
        if len(stations) <= 4:
            raise ValueError("Cannot remove a station from the minimum cage.")
        if index in {0, len(stations) - 1}:
            raise ValueError("Cannot remove an end station with a bounded local edit.")
        stations.pop(index)

    elif action.operation == "adjust_cutter":
        index = _require_index(cutters, action.target_index, "cutter")
        cutter = cutters[index]
        axis = str(updated.get("axis") or "y").lower()
        xyz_spans = (
            [max_half_width * 2.0, axis_span, height_span]
            if axis == "y"
            else [axis_span, max_half_width * 2.0, height_span]
        )
        location = [float(value) for value in cutter.get("location", [0, 0, 0])[:3]]
        scale = [float(value) for value in cutter.get("scale", [1, 1, 1])[:3]]
        rotation = [float(value) for value in cutter.get("rotation_deg", [0, 0, 0])[:3]]
        for dimension in range(3):
            delta = max(-0.35, min(0.35, float(action.location_delta_fraction[dimension])))
            factor = max(0.6, min(1.6, float(action.scale_factor[dimension])))
            rotation_delta = max(-90.0, min(90.0, float(action.rotation_delta_deg[dimension])))
            location[dimension] += delta * xyz_spans[dimension]
            scale[dimension] = max(0.02, scale[dimension] * factor)
            rotation[dimension] = max(-360.0, min(360.0, rotation[dimension] + rotation_delta))
        cutter["location"] = location
        cutter["scale"] = scale
        cutter["rotation_deg"] = rotation

    elif action.operation == "remove_cutter":
        index = _require_index(cutters, action.target_index, "cutter")
        cutters.pop(index)

    elif action.operation == "replan_representation":
        return updated

    else:  # pragma: no cover - Literal validation should make this unreachable.
        raise ValueError(f"Unsupported cage edit operation: {action.operation}")

    # Preserve longitudinal station ordering after a bounded position edit.
    updated["stations"] = sorted(stations, key=lambda station: float(station["position"]))
    updated["cutters"] = cutters

    # Prevent near-duplicate station positions, which create collapsed faces.
    positions = [float(station["position"]) for station in updated["stations"]]
    for left, right in pairwise(positions):
        if not math.isfinite(left) or not math.isfinite(right) or right - left < 0.03:
            raise ValueError("Cage edit collapsed adjacent longitudinal stations.")

    return updated
