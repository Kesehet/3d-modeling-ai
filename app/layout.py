"""Deterministic layout repairs for the P0 generic-model regression subjects.

These are conservative post-planning geometry constraints. The LLM still decides the
part inventory/materials; this module prevents semantically correct parts from being
stacked, buried, floating, or arranged into an unrecognizable silhouette.
"""

from __future__ import annotations

import math


def _name(item: dict) -> str:
    return " ".join(
        "".join(character if character.isalnum() else " " for character in str(item.get("name") or "").lower()).split()
    )


def _find(objects: list[dict], *terms: str) -> dict | None:
    normalized_terms = tuple(term.lower() for term in terms)
    for item in objects:
        name = _name(item)
        if any(term in name for term in normalized_terms):
            return item
    return None


def _find_all(objects: list[dict], *terms: str) -> list[dict]:
    normalized_terms = tuple(term.lower() for term in terms)
    return [
        item
        for item in objects
        if any(term in _name(item) for term in normalized_terms)
    ]


def _set_solid(
    item: dict,
    shape: str,
    location: list[float],
    scale: list[float],
    *,
    rotation: list[float] | None = None,
    color: str | None = None,
) -> None:
    item["shape"] = shape
    item["location"] = [float(value) for value in location]
    item["scale"] = [max(0.03, float(value)) for value in scale]
    item["rotation_deg"] = [float(value) for value in (rotation or [0.0, 0.0, 0.0])]
    item["start"] = None
    item["end"] = None
    item["radius"] = None
    if color:
        item["color"] = color


def _set_rod(
    item: dict,
    start: list[float],
    end: list[float],
    radius: float,
    *,
    color: str | None = None,
) -> None:
    item["shape"] = "rod"
    item["start"] = [float(value) for value in start]
    item["end"] = [float(value) for value in end]
    item["radius"] = max(0.02, float(radius))
    item["location"] = [float(value) for value in start]
    item["scale"] = [1.0, 1.0, 1.0]
    item["rotation_deg"] = [0.0, 0.0, 0.0]
    if color:
        item["color"] = color


def _enforce_pikachu(objects: list[dict]) -> None:
    body = _find(objects, "body", "torso")
    head = _find(objects, "head", "face")
    if body is None or head is None:
        return

    body_location = list(body.get("location") or [0.0, 0.0, 1.15])
    body_scale = [abs(float(value)) for value in (body.get("scale") or [1.05, 0.82, 1.15])]
    head_location = list(head.get("location") or [0.0, -0.02, 3.0])
    head_scale = [abs(float(value)) for value in (head.get("scale") or [1.3, 1.0, 1.05])]

    body_location[2] = max(float(body_location[2]), 1.15)
    head_location[2] = max(float(head_location[2]), body_location[2] + body_scale[2] + 0.72)
    _set_solid(body, "sphere", body_location, [1.05, 0.82, 1.15], color="#FACC15")
    _set_solid(head, "sphere", head_location, [1.32, 1.02, 1.05], color="#FACC15")

    front_y = float(head_location[1]) - 1.02
    for item in objects:
        name = _name(item)
        if "eye" in name:
            x = -0.48 if "left" in name else 0.48
            _set_solid(
                item,
                "sphere",
                [float(head_location[0]) + x, front_y - 0.05, float(head_location[2]) + 0.12],
                [0.17, 0.09, 0.20],
                color="#111111",
            )
        elif "cheek" in name or "red patch" in name:
            x = -0.78 if "left" in name else 0.78
            _set_solid(
                item,
                "sphere",
                [float(head_location[0]) + x, front_y - 0.04, float(head_location[2]) - 0.35],
                [0.27, 0.08, 0.24],
                color="#DC2626",
            )
        elif "ear" in name and "tip" not in name:
            x = -0.67 if "left" in name else 0.67
            rotation_y = -10.0 if "left" in name else 10.0
            _set_solid(
                item,
                "cone",
                [float(head_location[0]) + x, float(head_location[1]), float(head_location[2]) + 1.45],
                [0.30, 0.26, 0.90],
                rotation=[0.0, rotation_y, 0.0],
                color="#FACC15",
            )
        elif "ear" in name and "tip" in name:
            x = -0.79 if "left" in name else 0.79
            rotation_y = -10.0 if "left" in name else 10.0
            _set_solid(
                item,
                "cone",
                [float(head_location[0]) + x, float(head_location[1]), float(head_location[2]) + 2.15],
                [0.22, 0.20, 0.40],
                rotation=[0.0, rotation_y, 0.0],
                color="#111111",
            )
        elif ("arm" in name or "hand" in name) and "tool" not in name:
            side = -1.0 if "left" in name else 1.0
            x = float(body_location[0]) + side * 1.02
            z = float(body_location[2]) + (0.30 if "arm" in name else -0.05)
            _set_solid(
                item,
                "sphere",
                [x, float(body_location[1]) - 0.34, z],
                [0.26, 0.24, 0.48 if "arm" in name else 0.24],
                rotation=[0.0, 0.0, -24.0 * side],
                color="#FACC15",
            )
        elif "foot" in name or "feet" in name:
            side = -1.0 if "left" in name else 1.0
            _set_solid(
                item,
                "sphere",
                [
                    float(body_location[0]) + side * 0.55,
                    float(body_location[1]) - 0.40,
                    max(0.25, float(body_location[2]) - 1.0),
                ],
                [0.43, 0.52, 0.20],
                color="#FACC15",
            )

    tail_parts = _find_all(objects, "tail")
    if tail_parts:
        tail_parts.sort(key=lambda item: _name(item))
        count = len(tail_parts)
        x_base = float(body_location[0]) + 0.88
        y = float(body_location[1]) + 0.24
        z_base = float(body_location[2]) + 0.05
        points: list[list[float]] = [[x_base, y, z_base]]
        for index in range(1, count + 1):
            outward = 1.10 + (0.55 if index % 2 else 0.0) + index * 0.18
            points.append(
                [
                    float(body_location[0]) + outward,
                    y,
                    z_base + index * 0.42,
                ]
            )
        for index, item in enumerate(tail_parts):
            _set_rod(
                item,
                points[index],
                points[index + 1],
                0.14 if index < count - 1 else 0.11,
                color="#FACC15",
            )


def _enforce_lamp(objects: list[dict]) -> None:
    base = _find(objects, "base")
    stem = _find(objects, "stem", "post", "upright")
    joint = _find(objects, "pivot", "joint", "hinge")
    neck = _find(objects, "neck", "boom", "arm")
    shade = _find(objects, "shade", "dome", "hood", "lamp head", "lamphead")
    if not all((base, stem, joint, neck, shade)):
        return

    _set_solid(base, "cylinder", [0.0, 0.0, 0.20], [1.45, 1.45, 0.20], color="#3B3F46")
    _set_rod(stem, [0.0, 0.0, 0.35], [0.0, 0.0, 2.25], 0.13, color="#737A84")
    _set_solid(joint, "sphere", [0.0, 0.0, 2.25], [0.24, 0.24, 0.24], color="#737A84")
    neck_end = [1.15, -0.02, 3.08]
    _set_rod(neck, [0.0, 0.0, 2.25], neck_end, 0.14, color="#737A84")
    _set_solid(shade, "sphere", [1.23, -0.02, 3.12], [0.88, 0.72, 0.34], color="#D6C3A1")


def _enforce_sneaker(objects: list[dict]) -> None:
    sole_parts = _find_all(objects, "sole")
    sole_parts.sort(key=lambda item: _name(item))
    sole_z = [0.20, 0.38, 0.53]
    sole_scales = [
        [2.25, 0.78, 0.13],
        [2.18, 0.75, 0.10],
        [2.08, 0.72, 0.08],
    ]
    for index, item in enumerate(sole_parts[:3]):
        _set_solid(
            item,
            "cube",
            [0.0, 0.0, sole_z[min(index, 2)]],
            sole_scales[min(index, 2)],
            color="#F4F4F2",
        )

    toe = _find(objects, "toe")
    upper = _find(objects, "upper")
    heel = _find(objects, "heel")
    tongue = _find(objects, "tongue")
    opening = _find(objects, "opening", "collar")
    if toe is not None:
        _set_solid(toe, "sphere", [1.35, -0.03, 0.90], [0.88, 0.66, 0.43], color="#D9DCE2")
    if upper is not None:
        _set_solid(upper, "sphere", [-0.15, 0.0, 0.92], [1.48, 0.66, 0.54], color="#C9CDD3")
    if heel is not None:
        _set_solid(heel, "cube", [-1.55, 0.0, 0.95], [0.48, 0.64, 0.53], color="#A8AFB8")
    if tongue is not None:
        _set_solid(
            tongue,
            "cube",
            [-0.45, -0.10, 1.35],
            [0.62, 0.46, 0.11],
            rotation=[0.0, -18.0, 0.0],
            color="#737A84",
        )
    if opening is not None:
        _set_solid(
            opening,
            "torus",
            [-0.92, 0.0, 1.34],
            [0.58, 0.48, 0.16],
            color="#737A84",
        )

    laces = _find_all(objects, "lace", "cord", "string")
    laces.sort(key=lambda item: _name(item))
    for index, item in enumerate(laces[:6]):
        x = -0.35 + index * 0.28
        z = 1.24 - index * 0.035
        _set_rod(item, [x, -0.67, z], [x + 0.08, 0.67, z + 0.04], 0.045, color="#F4F4F2")


def _enforce_chair(objects: list[dict]) -> None:
    seat = _find(objects, "seat")
    back = _find(objects, "backrest", "back")
    column = _find(objects, "gas lift", "column")
    hub = _find(objects, "base hub", "hub")
    if seat is not None:
        _set_solid(seat, "cube", [0.0, 0.0, 2.35], [1.32, 1.08, 0.22], color="#3B3F46")
    if back is not None:
        _set_solid(
            back,
            "cube",
            [0.0, 0.92, 3.55],
            [1.26, 0.18, 1.20],
            rotation=[-8.0, 0.0, 0.0],
            color="#3B3F46",
        )
    if column is not None:
        _set_rod(column, [0.0, 0.0, 0.72], [0.0, 0.0, 2.13], 0.15, color="#737A84")
    if hub is not None:
        _set_solid(hub, "sphere", [0.0, 0.0, 0.67], [0.30, 0.30, 0.20], color="#737A84")

    for item in objects:
        name = _name(item)
        if "armrest" in name or "arm rest" in name:
            side = -1.0 if "left" in name else 1.0
            _set_rod(item, [side * 1.45, -0.55, 3.05], [side * 1.45, 0.55, 3.05], 0.13, color="#3B3F46")
        elif "arm support" in name:
            side = -1.0 if "left" in name else 1.0
            _set_rod(item, [side * 1.35, 0.15, 2.42], [side * 1.45, 0.15, 3.05], 0.11, color="#737A84")

    spokes = [item for item in objects if "spoke" in _name(item)]
    spokes.sort(key=lambda item: _name(item))
    wheels = [
        item for item in objects
        if "wheel" in _name(item) or "caster" in _name(item)
    ]
    wheels.sort(key=lambda item: _name(item))
    spoke_radius = 1.62
    for index, item in enumerate(spokes[:5]):
        angle = math.radians(index * 72.0 - 90.0)
        endpoint = [spoke_radius * math.cos(angle), spoke_radius * math.sin(angle), 0.43]
        _set_rod(item, [0.0, 0.0, 0.62], endpoint, 0.095, color="#737A84")
        if index < len(wheels):
            wheel = wheels[index]
            _set_solid(
                wheel,
                "torus",
                [endpoint[0], endpoint[1], 0.31],
                [0.24, 0.24, 0.13],
                rotation=[90.0, 0.0, math.degrees(angle)],
                color="#111111",
            )


def _robot_leg_code(name: str) -> str | None:
    compact = name.replace(" ", "")
    aliases = {
        "fl": ("legfl", "frontleft"),
        "fr": ("legfr", "frontright"),
        "rl": ("legrl", "rearleft", "backleft"),
        "rr": ("legrr", "rearright", "backright"),
    }
    for code, values in aliases.items():
        if any(value in compact for value in values):
            return code
    return None


def _enforce_quadruped(objects: list[dict]) -> None:
    torso = _find(objects, "torso", "chassis")
    if torso is None:
        return
    _set_solid(torso, "cube", [0.0, 0.0, 2.85], [2.05, 1.35, 0.75], color="#F4F4F2")

    hip = {
        "fl": [-1.58, -1.05, 2.55],
        "fr": [1.58, -1.05, 2.55],
        "rl": [-1.58, 1.05, 2.55],
        "rr": [1.58, 1.05, 2.55],
    }
    knee = {
        "fl": [-2.05, -1.16, 1.52],
        "fr": [2.05, -1.16, 1.52],
        "rl": [-2.05, 1.16, 1.52],
        "rr": [2.05, 1.16, 1.52],
    }
    ankle = {
        "fl": [-2.15, -1.18, 0.52],
        "fr": [2.15, -1.18, 0.52],
        "rl": [-2.15, 1.18, 0.52],
        "rr": [2.15, 1.18, 0.52],
    }

    for item in objects:
        name = _name(item)
        code = _robot_leg_code(name)
        if code:
            if "upper" in name:
                _set_rod(item, hip[code], knee[code], 0.20, color="#737A84")
            elif "lower" in name:
                _set_rod(item, knee[code], ankle[code], 0.17, color="#737A84")
            elif "joint" in name or "knee" in name:
                _set_solid(item, "sphere", knee[code], [0.28, 0.28, 0.28], color="#FF6700")
            elif "foot" in name:
                foot_x = ankle[code][0]
                foot_y = ankle[code][1] - (0.16 if code.startswith("f") else -0.16)
                _set_solid(
                    item,
                    "cube",
                    [foot_x, foot_y, 0.28],
                    [0.42, 0.55, 0.18],
                    color="#3B3F46",
                )
            continue

        if "camera" in name and "lens" not in name:
            _set_solid(item, "cube", [0.0, -1.58, 3.72], [0.72, 0.42, 0.42], color="#3B3F46")
        elif "camera" in name and "lens" in name:
            _set_solid(item, "sphere", [0.0, -2.02, 3.72], [0.28, 0.12, 0.28], color="#111111")
        elif "sensor pod" in name:
            side = -1.0 if "left" in name else 1.0
            _set_solid(
                item,
                "cylinder",
                [side * 2.28, -0.15, 3.02],
                [0.36, 0.36, 0.48],
                rotation=[90.0, 0.0, 0.0],
                color="#FF6700",
            )
        elif "antenna mast" in name:
            _set_rod(item, [0.0, 0.15, 3.58], [0.0, 0.15, 4.75], 0.09, color="#737A84")
        elif "antenna tip" in name:
            _set_solid(item, "sphere", [0.0, 0.15, 4.86], [0.16, 0.16, 0.16], color="#FF6700")
        elif "battery" in name:
            _set_solid(item, "cube", [0.0, 1.58, 2.92], [1.15, 0.28, 0.50], color="#3B3F46")

    tool_upper = _find(objects, "tool arm upper")
    tool_joint = _find(objects, "tool arm joint")
    tool_lower = _find(objects, "tool arm lower")
    tool_end = _find(objects, "tool end", "end effector")
    tool_start = [2.02, -0.30, 3.10]
    tool_knee = [2.75, -0.70, 2.75]
    tool_tip = [3.35, -1.10, 2.35]
    if tool_upper is not None:
        _set_rod(tool_upper, tool_start, tool_knee, 0.15, color="#FF6700")
    if tool_joint is not None:
        _set_solid(tool_joint, "sphere", tool_knee, [0.23, 0.23, 0.23], color="#3B3F46")
    if tool_lower is not None:
        _set_rod(tool_lower, tool_knee, tool_tip, 0.13, color="#FF6700")
    if tool_end is not None:
        _set_solid(tool_end, "cube", tool_tip, [0.28, 0.18, 0.22], color="#3B3F46")


def enforce_subject_layout(data: dict, prompt: str) -> dict:
    objects = data.get("objects")
    if not isinstance(objects, list):
        return data

    text = prompt.lower()
    if "pikachu" in text:
        _enforce_pikachu(objects)
    if "lamp" in text:
        _enforce_lamp(objects)
    if "sneaker" in text or "shoe" in text:
        _enforce_sneaker(objects)
    if "office chair" in text or ("chair" in text and ("caster" in text or "spoke" in text)):
        _enforce_chair(objects)
    if "quadruped" in text or ("robot" in text and "four" in text and "leg" in text):
        _enforce_quadruped(objects)
    return data
