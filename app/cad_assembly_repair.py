"""Bounded, subject-agnostic correction of disconnected CAD part groups.

A structural correction never invents new geometry or changes rotation/scale.
It shifts entire existing disconnected islands using evaluated CAD bounding
evidence. A subsequent real BRep check must verify improvement.
"""
from __future__ import annotations

import math


def suggest_group_attachment(blueprint, report: dict, *, max_fraction: float = 0.20) -> dict:
    qa = report.get("assembly_qa") or {}
    groups = qa.get("groups") or []
    bounds = report.get("part_bounds_mm") or {}
    if qa.get("pass") is True or blueprint.assembly_mode != "single" or len(groups) < 2:
        return {"changed": False, "moves": [], "reason": "no disconnected single assembly"}
    if not all(part.name in bounds for part in blueprint.parts):
        return {"changed": False, "moves": [], "reason": "missing evaluated part bounds"}

    evidence = {row["name"]: float(row["volume_mm3"])
                for row in report.get("parts", [])}
    if not evidence:
        return {"changed": False, "moves": [], "reason": "no part volume evidence"}

    # Retain the largest evaluated solid at its original coordinates.
    anchor_name = max(evidence, key=evidence.get)
    anchor_group = next((group for group in groups if anchor_name in group), None)
    if not anchor_group:
        return {"changed": False, "moves": [], "reason": "anchor not in topology"}
    low = [min(bounds[name][0][axis] for name in bounds) for axis in range(3)]
    high = [max(bounds[name][1][axis] for name in bounds) for axis in range(3)]
    diagonal = math.dist(low, high)
    limit_mm = diagonal * max_fraction
    candidates = qa.get("attachment_candidates") or []
    if qa.get("measurement") != "brep_closest_solid_distance" or not candidates:
        return {"changed": False, "moves": [], "reason": "missing exact attachment evidence"}

    shifts: dict[str, tuple[float, float, float]] = {}
    moves: list[dict] = []
    connected = set(anchor_group)
    pending = [group for group in groups if group is not anchor_group]

    while pending:
        proposals = []
        for index, group in enumerate(pending):
            for pair in candidates:
                first, second = pair["first"], pair["second"]
                if first in connected and second in group:
                    target, moving = first, second
                    target_point, moving_point = pair["first_point_mm"], pair["second_point_mm"]
                elif second in connected and first in group:
                    target, moving = second, first
                    target_point, moving_point = pair["second_point_mm"], pair["first_point_mm"]
                else:
                    continue
                target_shift = shifts.get(target, (0.0, 0.0, 0.0))
                # Points were measured before any moves. Account for the target
                # island's translation, and bound the TOTAL move from the original.
                delta = tuple(target_point[axis] + target_shift[axis] - moving_point[axis]
                              for axis in range(3))
                magnitude = math.sqrt(sum(value**2 for value in delta))
                if 0 < magnitude <= limit_mm:
                    proposals.append((magnitude, index, target, moving, delta))
        if not proposals:
            moves.extend({"parts": group, "status": "not_moved",
                          "reason": "no measured attachment within bounded total displacement"}
                         for group in pending)
            break
        magnitude, index, target, moving, delta = min(proposals)
        group = pending.pop(index)
        for name in group:
            shifts[name] = delta
        connected.update(group)
        moves.append({"parts": group, "status": "proposed", "offset_mm": list(delta),
                      "distance_mm": round(magnitude, 3), "target_part": target,
                      "source_part": moving, "measurement": "brep_closest_points"})

    if not shifts:
        return {"changed": False, "moves": moves, "reason": "no safe corrections"}
    parts = []
    for part in blueprint.parts:
        if part.name not in shifts:
            parts.append(part)
        else:
            delta = shifts[part.name]
            parts.append(part.model_copy(update={
                "position_mm": tuple(part.position_mm[axis] + delta[axis] for axis in range(3))
            }))
    return {
        "changed": True,
        "blueprint": blueprint.model_copy(update={"parts": parts}),
        "moves": moves,
        "reason": "bounded evaluated-part island translations; must reverify geometry",
    }
