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


def evaluate_assembly_solids(
    solids: dict[str, object],
    part_bounds: dict[str, list[list[float]]],
    *,
    assembly_mode: str = "single",
) -> dict:
    """Validate real OpenCascade shape proximity, not just overlapping bounds.

    Broad phase uses AABB lower bound; narrow phase uses BRep closest-surface
    distance. This checks proximity, not collision-free printable assembly.
    """
    bounding = evaluate_assembly_bounds(part_bounds, assembly_mode=assembly_mode)
    names = list(part_bounds)
    if set(names) != set(solids):
        raise ValueError("CAD evaluated solids and bounds have different part identities")
    if not names:
        return {**bounding, "measurement": "brep_closest_solid_distance",
                "attachment_candidates": []}
    low = [min(part_bounds[name][0][i] for name in names) for i in range(3)]
    high = [max(part_bounds[name][1][i] for name in names) for i in range(3)]
    diagonal = math.dist(low, high)
    exact_tolerance = min(2.0, max(0.25, diagonal * 0.0002))
    graph: dict[str, set[str]] = {name: set() for name in names}
    measurements: list[dict] = []
    for index, name in enumerate(names):
        for other in names[index + 1:]:
            aabb_gap = _gap(part_bounds[name], part_bounds[other])
            if aabb_gap > exact_tolerance:
                actual_gap = None  # Proven lower bound already exceeds tolerance.
                connected = False
            else:
                try:
                    actual_gap = float(solids[name].distance_to(solids[other]))
                except (AttributeError, TypeError, RuntimeError, ValueError) as exc:
                    raise ValueError(f"BRep closest-distance measurement failed: {name} / {other}") from exc
                if not math.isfinite(actual_gap) or actual_gap < -1e-7:
                    raise ValueError("OpenCascade returned an invalid solid distance")
                connected = actual_gap <= exact_tolerance
            if connected:
                graph[name].add(other)
                graph[other].add(name)
            else:
                measurements.append({
                    "first": name,
                    "second": other,
                    "gap_mm": round(actual_gap if actual_gap is not None else aabb_gap, 3),
                    "gap_is_lower_bound": actual_gap is None,
                })

    unseen = set(names)
    groups = []
    while unseen:
        start = next(name for name in names if name in unseen)
        pending = [start]
        unseen.remove(start)
        group = []
        while pending:
            current = pending.pop()
            group.append(current)
            for neighbor in sorted(graph[current] & unseen):
                unseen.remove(neighbor)
                pending.append(neighbor)
        groups.append(sorted(group))
    failures = []
    if assembly_mode == "single" and len(groups) > 1:
        for group in groups[1:8]:
            nearest = min(
                (pair for pair in measurements
                 if (pair["first"] in group) != (pair["second"] in group)),
                key=lambda pair: pair["gap_mm"],
            )
            caveat = "at least " if nearest["gap_is_lower_bound"] else ""
            failures.append(
                f"Disconnected solid group {', '.join(group)}; nearest separate solid "
                f"{caveat}{nearest['gap_mm']:.1f} mm away "
                f"({nearest['first']} / {nearest['second']})"
            )

    # Keep one exact nearest-surface bridge per pair of disconnected islands.
    # Bounding-box gaps only order/prune the search; they never choose a move.
    # Unlike a union AABB this cannot attract parts into empty space or cavities.
    attachments = []
    if assembly_mode == "single" and len(groups) > 1:
        for index, first_group in enumerate(groups):
            for second_group in groups[index + 1:]:
                best = None
                candidates = sorted(
                    (_gap(part_bounds[a], part_bounds[b]), a, b)
                    for a in first_group for b in second_group
                )
                for lower_bound, first, second in candidates:
                    if best is not None and lower_bound > best["gap_mm"]:
                        break
                    try:
                        gap, point_a, point_b = solids[first].distance_to_with_closest_points(solids[second])
                        points = [list(point_a), list(point_b)]
                        gap = float(gap)
                        if (not math.isfinite(gap) or gap < 0
                                or any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in points)
                                or not math.isclose(math.dist(*points), gap, rel_tol=1e-5, abs_tol=1e-5)):
                            raise ValueError("Invalid nearest-surface evidence")
                    except (AttributeError, TypeError, RuntimeError, ValueError) as exc:
                        raise ValueError(f"BRep attachment measurement failed: {first} / {second}") from exc
                    if best is None or gap < best["gap_mm"]:
                        best = {"first": first, "second": second, "gap_mm": gap,
                                "first_point_mm": points[0], "second_point_mm": points[1]}
                if best is not None:
                    attachments.append(best)

    return {
        **bounding,
        "pass": not failures,
        "measurement": "brep_closest_solid_distance",
        "contact_verified": False,
        "solid_proximity_verified": not failures if assembly_mode == "single" else None,
        "physical_fit_verified": False,
        "gap_tolerance_mm": round(exact_tolerance, 3),
        "groups": groups,
        "failures": failures,
        "nearest_noncontact_pairs": sorted(measurements, key=lambda pair: pair["gap_mm"])[:12],
        "attachment_candidates": attachments,
    }
