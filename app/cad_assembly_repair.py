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

    names = {part.name: part for part in blueprint.parts}
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
    overlap_mm = max(0.5, min(3.0, diagonal * 0.0003))

    def group_box(group: list[str]) -> list[list[float]]:
        return [
            [min(bounds[name][0][axis] for name in group) for axis in range(3)],
            [max(bounds[name][1][axis] for name in group) for axis in range(3)],
        ]

    anchor_box = group_box(anchor_group)
    shifts: dict[str, tuple[float, float, float]] = {}
    moves: list[dict] = []
    pending = [group for group in groups if group is not anchor_group]

    while pending:
        # Attach nearest island first; then expand the connected anchor bounds.
        def gap_score(group):
            box = group_box(group)
            return math.sqrt(sum(
                max(0.0, box[0][axis] - anchor_box[1][axis],
                    anchor_box[0][axis] - box[1][axis]) ** 2
                for axis in range(3)
            ))
        group = min(pending, key=gap_score)
        pending.remove(group)
        group_bounds = group_box(group)
        motion = []
        for axis in range(3):
            if group_bounds[0][axis] > anchor_box[1][axis]:
                motion.append(anchor_box[1][axis] - group_bounds[0][axis] - overlap_mm)
            elif group_bounds[1][axis] < anchor_box[0][axis]:
                motion.append(anchor_box[0][axis] - group_bounds[1][axis] + overlap_mm)
            else:
                motion.append(0.0)
        magnitude = math.sqrt(sum(delta**2 for delta in motion))
        if not 0 < magnitude <= limit_mm:
            moves.append({
                "parts": group, "status": "not_moved",
                "reason": "overlapping AABBs or correction exceeds bounded limit",
                "distance_mm": round(magnitude, 3),
            })
            continue
        delta = tuple(motion)
        for name in group:
            shifts[name] = delta
        moves.append({"parts": group, "status": "proposed", "offset_mm": list(delta),
                      "distance_mm": round(magnitude, 3)})
        for axis in range(3):
            anchor_box[0][axis] = min(anchor_box[0][axis], group_bounds[0][axis] + delta[axis])
            anchor_box[1][axis] = max(anchor_box[1][axis], group_bounds[1][axis] + delta[axis])

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
