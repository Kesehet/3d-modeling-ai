"""Independent spatial benchmark checks for declarative CPU CAD assemblies.

These checks evaluate proposed geometry. They never alter it or certify an
AI-created model from a successful STEP export alone. Role names belong to the
benchmark prompt, not to subject-specific geometry generation code.
"""
from __future__ import annotations

import math
from statistics import median


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return tuple(v / norm for v in vector) if norm > 1e-9 else (0.0, 0.0, 0.0)


def _rotate(part, local: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate a local axis with Blender-style XYZ Euler angles (Rz * Ry * Rx)."""
    rx, ry, rz = (math.radians(value) for value in part.rotation_deg)
    x, y, z = local
    y, z = y * math.cos(rx) - z * math.sin(rx), y * math.sin(rx) + z * math.cos(rx)
    x, z = x * math.cos(ry) + z * math.sin(ry), -x * math.sin(ry) + z * math.cos(ry)
    x, y = x * math.cos(rz) - y * math.sin(rz), x * math.sin(rz) + y * math.cos(rz)
    return _unit((x, y, z))


def _axis_length(part) -> tuple[tuple[float, float, float], float] | None:
    """Measure a distinguishable longest axis; don't guess on spheres/cubes."""
    if part.kind == "box":
        lengths = list(part.dimensions_mm)
        major = max(range(3), key=lengths.__getitem__)
        if lengths[major] < 1.25 * sorted(lengths)[-2]:
            return None
        direction = tuple(1.0 if i == major else 0.0 for i in range(3))
        return _rotate(part, direction), lengths[major]
    if part.kind in ("cylinder", "tube", "cone"):
        radius = part.radius_mm
        if part.height_mm < 1.25 * 2 * radius:
            return None
        return _rotate(part, (0.0, 0.0, 1.0)), part.height_mm
    return None


def _cross_section_radius(part) -> float | None:
    if part.kind in ("cylinder", "tube", "cone", "sphere"):
        return part.radius_mm
    if part.kind == "box":
        return max(part.dimensions_mm[:2]) / 2
    return None


def evaluate_spatial_contract(
    blueprint,
    *,
    coaxial_terms: tuple[str, ...],
    vertical_support_term: str,
    supported_term: str,
    radial_term: str,
    radial_count: int,
    contact_tolerance_mm: float = 5.0,
    max_axis_offset_mm: float = 5.0,
    max_radial_angle_error_deg: float = 8.0,
    max_radial_size_deviation: float = 0.25,
) -> dict:
    """Fail closed on missing parts, impossible contact, direction, or symmetry.

    Intended for holdout fixtures with explicit coaxial/support/radial intent.
    It is not a universal semantic classifier; visual reference comparison is
    a separate gate. Measurements accompany failures for actionable repairs.
    """
    parts = blueprint.parts
    failures: list[str] = []
    measurements: dict = {"coaxial_offsets_mm": {}, "radial_parts": {}}
    if not coaxial_terms or radial_count < 2:
        raise ValueError("Spatial benchmark requires an axis anchor and at least two radial parts")
    if contact_tolerance_mm < 0 or max_axis_offset_mm < 0:
        raise ValueError("Distance tolerances cannot be negative")

    named = {}
    for term in coaxial_terms:
        matches = [p for p in parts if term.lower() in p.name.lower()]
        if len(matches) != 1:
            failures.append(f"Expected exactly one part containing {term!r}, got {len(matches)}")
        else:
            named[term] = matches[0]
    anchor = named.get(coaxial_terms[0])
    if anchor is not None:
        for term, part in named.items():
            if part.kind in ("cylinder", "tube", "cone"):
                axis = _rotate(part, (0, 0, 1))
                if abs(axis[2]) < 0.95:
                    failures.append(f"{term!r} has an axis tilted away from central axis")
            offset = math.dist(part.position_mm[:2], anchor.position_mm[:2])
            measurements["coaxial_offsets_mm"][term] = round(offset, 3)
            if offset > max_axis_offset_mm:
                failures.append(f"{term!r} is {offset:.1f} mm off central axis")

    supports = [p for p in parts if vertical_support_term.lower() in p.name.lower()]
    parents = [p for p in parts if supported_term.lower() in p.name.lower()]
    if len(supports) != 1 or len(parents) != 1:
        failures.append("Expected exactly one vertical support and one supported body")
    else:
        support, parent = supports[0], parents[0]
        direction = _rotate(support, (0, 0, 1))
        if support.kind not in ("cylinder", "tube", "cone") or direction[2] ** 2 < 0.9:
            failures.append("Vertical support is missing its upright cylindrical axis")
        else:
            bottom = support.position_mm[2] - support.height_mm * abs(direction[2]) / 2
            top = support.position_mm[2] + support.height_mm * abs(direction[2]) / 2
            parent_height = parent.height_mm if parent.height_mm is not None else (
                parent.dimensions_mm[2] if parent.dimensions_mm else 0
            )
            parent_top = parent.position_mm[2] + parent_height / 2
            measurements["support"] = {"bottom_mm": round(bottom, 3),
                                       "top_mm": round(top, 3),
                                       "parent_top_mm": round(parent_top, 3)}
            if top <= parent_top or abs(bottom - parent_top) > contact_tolerance_mm:
                failures.append("Vertical support does not extend upward from parent with end contact")
            offset = math.dist(support.position_mm[:2], parent.position_mm[:2])
            if offset > max_axis_offset_mm:
                failures.append(f"Vertical support is {offset:.1f} mm off supported-body axis")

    arms = [p for p in parts if radial_term.lower() in p.name.lower()]
    if len(arms) != radial_count:
        failures.append(f"Expected {radial_count} radial parts, got {len(arms)}")
    if anchor is not None and arms:
        angles, radii, lengths = [], [], []
        radius = _cross_section_radius(anchor)
        for part in arms:
            dx = part.position_mm[0] - anchor.position_mm[0]
            dy = part.position_mm[1] - anchor.position_mm[1]
            dist = math.hypot(dx, dy)
            axis = _axis_length(part)
            if dist <= contact_tolerance_mm:
                failures.append(f"Radial part {part.name} lies on central axis")
                continue
            angles.append(math.atan2(dy, dx) % (2 * math.pi))
            radii.append(dist)
            if axis is None:
                failures.append(f"Radial part {part.name} has no measurable long axis")
                continue
            direction, length = axis
            lengths.append(length)
            alignment = abs((dx * direction[0] + dy * direction[1]) / dist)
            projected_inner_end = dist - length * alignment / 2
            lateral_offset = abs(dx * direction[1] - dy * direction[0])
            measurements["radial_parts"][part.name] = {
                "radius_mm": round(dist, 3), "length_mm": round(length, 3),
                "radial_alignment": round(alignment, 4),
                "inner_end_radius_mm": round(projected_inner_end, 3),
                "axis_vertical_component": round(abs(direction[2]), 4),
                "lateral_offset_mm": round(lateral_offset, 3),
            }
            if alignment < math.cos(math.radians(20)) or abs(direction[2]) > 0.25:
                failures.append(f"Radial part {part.name} points away from its radial direction")
            if radius is not None and lateral_offset > radius + contact_tolerance_mm:
                failures.append(f"Radial part {part.name} misses its central attachment axis")
            if radius is not None and projected_inner_end > radius + contact_tolerance_mm:
                failures.append(f"Radial part {part.name} is detached from central body")
            if dist < length / 2 - (radius or 0) - contact_tolerance_mm:
                failures.append(f"Radial part {part.name} crosses too far through central body")
            thickness = (min(part.dimensions_mm) if part.kind == "box" else 2 * part.radius_mm)
            if abs(part.position_mm[2] - anchor.position_mm[2]) > (
                (anchor.height_mm or (anchor.dimensions_mm[2] if anchor.dimensions_mm else 0)) / 2
                + thickness / 2 + contact_tolerance_mm
            ):
                failures.append(f"Radial part {part.name} has no vertical overlap with central body")

        if len(angles) == radial_count:
            angles.sort()
            gaps = [(angles[(i + 1) % radial_count] - angles[i]) % (2 * math.pi)
                    for i in range(radial_count)]
            measurements["angular_gaps_deg"] = [round(math.degrees(v), 2) for v in gaps]
            if any(abs(math.degrees(gap) - 360 / radial_count) > max_radial_angle_error_deg
                   for gap in gaps):
                failures.append("Radial parts are not evenly spaced")
        for metric, values in (("length", lengths), ("distance", radii)):
            if len(values) == radial_count and median(values) > 0 and any(
                abs(v / median(values) - 1) > max_radial_size_deviation for v in values
            ):
                failures.append(f"Repeated radial part {metric}s are inconsistent")
    return {"pass": not failures, "failures": failures, "measurements": measurements}
