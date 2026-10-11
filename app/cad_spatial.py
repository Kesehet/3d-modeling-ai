"""Declarative spatial-contract quality checks for CAD benchmark outputs."""
from __future__ import annotations

import math


def evaluate_spatial_contract(
    blueprint,
    *,
    coaxial_terms: tuple[str, ...],
    vertical_support_term: str,
    supported_term: str,
    radial_term: str,
    radial_count: int,
) -> dict:
    """Benchmark-only spatial QA. Never changes CAD geometry."""
    parts = blueprint.parts
    findings = []
    centers = {}
    for term in coaxial_terms:
        matches = [p for p in parts if term in p.name.lower()]
        if len(matches) != 1:
            findings.append(f"Expected one part containing {term!r}, got {len(matches)}")
        else:
            centers[term] = matches[0]
    if len(centers) == len(coaxial_terms):
        anchor = centers[coaxial_terms[0]].position_mm
        for term, part in centers.items():
            xy_offset = math.hypot(part.position_mm[0] - anchor[0],
                                   part.position_mm[1] - anchor[1])
            if xy_offset > 5:
                findings.append(f"{term!r} is {xy_offset:.1f} mm off central axis")
    supports = [p for p in parts if vertical_support_term in p.name.lower()]
    parents = [p for p in parts if supported_term in p.name.lower()]
    if len(supports) == 1 and len(parents) == 1:
        support, parent = supports[0], parents[0]
        if support.height_mm is None or support.position_mm[2] - support.height_mm / 2 < parent.position_mm[2]:
            findings.append("Vertical support does not extend upward from parent")
    arms = [p for p in parts if radial_term in p.name.lower()]
    if len(arms) != radial_count:
        findings.append(f"Expected {radial_count} radial parts, got {len(arms)}")
    elif coaxial_terms[0] in centers:
        origin = centers[coaxial_terms[0]].position_mm
        angles = []
        for part in arms:
            dx = part.position_mm[0] - origin[0]
            dy = part.position_mm[1] - origin[1]
            if math.hypot(dx, dy) < 5:
                findings.append(f"Radial part {part.name} lies on central axis")
            else:
                angles.append(math.atan2(dy, dx) % (2 * math.pi))
        if len(angles) == radial_count:
            angles.sort()
            gaps = [(angles[(i + 1) % radial_count] - angles[i]) % (2 * math.pi)
                    for i in range(radial_count)]
            if any(abs(gap - 2 * math.pi / radial_count) > 0.14 for gap in gaps):
                findings.append("Radial parts are not evenly spaced")
    return {"pass": not findings, "failures": findings}

