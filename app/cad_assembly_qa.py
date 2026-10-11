"""Subject-agnostic assembly screening on evaluated CAD part bounds.

This is a conservative early rejection gate, NOT a proof of mesh contact:
overlapping AABBs can still contain disconnected solids. Independent
subassemblies must be explicitly requested by the planner.
"""
from __future__ import annotations

import math


def _gap(a: list[list[float]], b: list[list[float]]) -> float:
    """Euclidean gap between axis-aligned bounding boxes, in millimetres."""
    return math.sqrt(sum(
        max(0.0, a[0][axis] - b[1][axis], b[0][axis] - a[1][axis]) ** 2
        for axis in range(3)
    ))


def evaluate_assembly_bounds(
    part_bounds: dict[str, list[list[float]]],
    *,
    assembly_mode: str = "single",
) -> dict:
    """Report spatially disconnected groups without any object-name rules.

    AABB overlap is necessary but not sufficient for physical contact. Units
    are mm, and the bounded scale-relative tolerance absorbs tessellation gaps
    rather than hiding implausible assembly errors.
    """
    if not part_bounds:
        return {"pass": False, "failures": ["No evaluated CAD parts"], "groups": [],
                "measurement": "aabb_only"}
    names = list(part_bounds)
    for name, bounds in part_bounds.items():
        if (len(bounds) != 2 or any(len(v) != 3 for v in bounds)
                or any(not math.isfinite(x) for side in bounds for x in side)
                or any(bounds[0][i] > bounds[1][i] for i in range(3))):
            raise ValueError(f"Invalid evaluated bounds for {name}")

    low = [min(part_bounds[name][0][i] for name in names) for i in range(3)]
    high = [max(part_bounds[name][1][i] for name in names) for i in range(3)]
    diagonal = math.dist(low, high)
    tolerance = min(30.0, max(1.0, diagonal * 0.005))
    adj: dict[str, set[str]] = {name: set() for name in names}
    gaps: list[dict] = []
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            gap_mm = _gap(part_bounds[first], part_bounds[second])
            gaps.append({"first": first, "second": second, "gap_mm": round(gap_mm, 3)})
            if gap_mm <= tolerance:
                adj[first].add(second)
                adj[second].add(first)

    groups: list[list[str]] = []
    remaining = set(names)
    while remaining:
        seed = next(name for name in names if name in remaining)
        group, todo = [], [seed]
        remaining.remove(seed)
        while todo:
            current = todo.pop()
            group.append(current)
            for candidate in sorted(adj[current] & remaining):
                remaining.remove(candidate)
                todo.append(candidate)
        groups.append(sorted(group))
    failures: list[str] = []
    if assembly_mode == "single" and len(groups) > 1:
        for group in groups[1:8]:
            nearest = min(
                (entry for entry in gaps if (entry["first"] in group) !=
                 (entry["second"] in group)),
                key=lambda entry: entry["gap_mm"],
            )
            failures.append(
                f"Disconnected part group {', '.join(group)}; nearest other group "
                f"{nearest['gap_mm']:.1f} mm away ({nearest['first']} / {nearest['second']})"
            )
    return {
        "pass": not failures,
        "assembly_mode": assembly_mode,
        "measurement": "aabb_only",
        "contact_verified": False,
        "gap_tolerance_mm": round(tolerance, 3),
        "groups": groups,
        "failures": failures,
        "nearest_noncontact_pairs": sorted(
            (entry for entry in gaps if entry["gap_mm"] > tolerance),
            key=lambda entry: entry["gap_mm"],
        )[:12],
    }
