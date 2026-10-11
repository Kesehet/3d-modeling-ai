"""Opt-in CPU-only parametric CAD backend.

The LLM supplies a declarative, bounded part graph; no LLM-generated Python runs.
Object geometry is never selected by subject name.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CadPart(BaseModel):
    name: str = Field(min_length=1, max_length=60, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    kind: Literal["box", "cylinder", "cone", "sphere", "tube", "profile_extrusion", "section_loft"]
    position_mm: tuple[float, float, float] = (0, 0, 0)
    rotation_deg: tuple[float, float, float] = (0, 0, 0)
    dimensions_mm: tuple[float, float, float] | None = None
    radius_mm: float | None = None
    top_radius_mm: float | None = None
    inner_radius_mm: float | None = None
    height_mm: float | None = None
    profile_points_mm: list[tuple[float, float]] | None = Field(default=None, max_length=24)
    profile_plane: Literal["xy", "xz", "yz"] = "xy"
    loft_sections_mm: list[tuple[float, float, float, float, float]] | None = Field(
        default=None, max_length=12
    )

    @model_validator(mode="after")
    def check_geometry(self) -> CadPart:
        vals = [*self.position_mm, *self.rotation_deg]
        if any(not math.isfinite(v) for v in vals):
            raise ValueError("Coordinates and angles must be finite")
        if any(abs(v) > 10_000 for v in self.position_mm):
            raise ValueError("Position exceeds 10000 mm")
        if any(abs(v) > 360 for v in self.rotation_deg):
            raise ValueError("Rotation exceeds 360 degrees")
        if self.kind == "profile_extrusion":
            pts = self.profile_points_mm or []
            if not 3 <= len(pts) <= 24:
                raise ValueError("profile_extrusion requires 3 to 24 2D profile points")
            if self.height_mm is None or not math.isfinite(self.height_mm) or not 0 < self.height_mm <= 20000:
                raise ValueError("profile_extrusion requires positive extrusion height_mm")
            if any(not math.isfinite(v) or abs(v) > 20000 for point in pts for v in point):
                raise ValueError("Profile contains nonfinite/out-of-range coordinates")
            signed_area = sum(
                pts[i][0] * pts[(i + 1) % len(pts)][1]
                - pts[(i + 1) % len(pts)][0] * pts[i][1]
                for i in range(len(pts))
            ) / 2
            if abs(signed_area) < 1e-3:
                raise ValueError("Profile has zero/tiny planar area")
            return self
        if self.kind == "section_loft":
            stations = self.loft_sections_mm or []
            if not 2 <= len(stations) <= 12:
                raise ValueError("section_loft requires 2 to 12 stations")
            if any(not math.isfinite(v) or abs(v) > 20000 for entry in stations for v in entry):
                raise ValueError("Loft stations contain nonfinite/out-of-range coordinates")
            if any(entry[1] <= 0 or entry[2] <= 0 for entry in stations):
                raise ValueError("Loft station widths/depths must be positive")
            if any(stations[i + 1][0] - stations[i][0] <= 0.01
                   for i in range(len(stations) - 1)):
                raise ValueError("Loft station Z coordinates must be strictly increasing")
            return self
        if self.kind == "box":
            if self.dimensions_mm is None:
                raise ValueError("Box requires dimensions_mm")
            sizes = self.dimensions_mm
        elif self.kind == "sphere":
            sizes = (self.radius_mm,)
        elif self.kind == "cone":
            sizes = (self.radius_mm, self.top_radius_mm, self.height_mm)
        elif self.kind == "tube":
            sizes = (self.radius_mm, self.inner_radius_mm, self.height_mm)
        else:
            sizes = (self.radius_mm, self.height_mm)
        if any(v is None or not math.isfinite(v) or v <= 0 or v > 20_000 for v in sizes):
            # A cone's upper radius may approach zero but should not be exactly zero here.
            raise ValueError("Required dimensions must be positive finite mm values <= 20000")
        if self.kind == "tube" and not (self.inner_radius_mm < self.radius_mm):
            raise ValueError("Tube inner radius must be smaller than outer radius")
        return self


class CadBlueprint(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    units: Literal["mm"] = "mm"
    parts: list[CadPart] = Field(min_length=1, max_length=32)
    assembly_mode: Literal["single", "independent"] = "single"

    @model_validator(mode="after")
    def unique_part_names(self) -> CadBlueprint:
        names = [part.name.lower() for part in self.parts]
        if len(names) != len(set(names)):
            raise ValueError("Each part must have a distinct name")
        return self


def cad_generation_schema() -> dict:
    """Minimal per-shape LLM schema. Avoid forcing irrelevant nullable fields.

    The true source of validation remains CadBlueprint/Pydantic after inference.
    """
    number = {"type": "number"}
    triple = {"type": "array", "items": number, "minItems": 3, "maxItems": 3}
    pair = {"type": "array", "items": number, "minItems": 2, "maxItems": 2}
    loft_section = {"type": "array", "items": number, "minItems": 5, "maxItems": 5}
    common = {
        "name": {"type": "string", "minLength": 1},
        "position_mm": triple,
        "rotation_deg": triple,
    }
    variants = {
        "box": {"dimensions_mm": triple},
        "cylinder": {"radius_mm": number, "height_mm": number},
        "cone": {"radius_mm": number, "top_radius_mm": number, "height_mm": number},
        "sphere": {"radius_mm": number},
        "tube": {"radius_mm": number, "inner_radius_mm": number, "height_mm": number},
        "profile_extrusion": {
            "profile_plane": {"type": "string", "enum": ["xy", "xz", "yz"]},
            "profile_points_mm": {
                "type": "array", "items": pair, "minItems": 3, "maxItems": 24,
            },
            "height_mm": number,
        },
        "section_loft": {
            "loft_sections_mm": {
                "type": "array", "items": loft_section, "minItems": 2, "maxItems": 12,
            },
        },
    }
    shapes = [
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                **common,
                "kind": {"type": "string", "enum": [kind]},
                **properties,
            },
            "required": ["name", "kind", *(
                name for name in properties if name != "profile_plane"
            )],
        }
        for kind, properties in variants.items()
    ]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "units": {"type": "string", "enum": ["mm"]},
            "parts": {"type": "array", "items": {"oneOf": shapes},
                      "minItems": 1, "maxItems": 32},
        },
        "required": ["title", "parts"],
    }


CAD_SYSTEM_PROMPT = """
You are a parametric CAD designer. Return ONLY the requested JSON blueprint.
Create physically plausible, separately named, correctly positioned components.
Choose geometry from the user's description; NEVER omit major requested components.
Units are millimetres. Coordinate system is Z-up. Origin should be around assembly centre.
Allowed part kinds: box, cylinder, cone, sphere, tube, profile_extrusion, section_loft.
box: dimensions_mm=[x,y,z] are full sizes.
cylinder: radius_mm and height_mm (along local Z).
cone: radius_mm (bottom), top_radius_mm (top), height_mm (local Z).
sphere: radius_mm.
tube: radius_mm outer, inner_radius_mm inner, height_mm (local Z).
profile_extrusion: profile_points_mm=[[u,v],...] in the selected
profile_plane (xy, xz or yz), height_mm = symmetric SOLID EXTRUSION DEPTH
normal to that plane. For a side silhouette needing vertical roof slopes,
choose profile_plane="xz": [u,v]=[horizontal X, vertical Z], then give
height_mm the WIDTH along Y. For front silhouettes choose "yz".
Extrusion is centered about the sketch plane. Points describe actual outer
contour with sloped sides, not a rectangular bounding box.
Use this for angled/trapezoidal/curved-like polygon silhouettes; use
rotation_deg only when the 3 named profile planes cannot represent the outline.
section_loft: loft_sections_mm=[[z,width,depth,offset_x,offset_y], ...] at 2-12
ordered stations along local Z. Adjacent rectangular sections are joined by
smoothly varying surfaces to form tapered or offset solids.
Prefer these general profile tools when plain boxes/cylinders cannot reproduce
the requested silhouette. Avoid stacks of blocky rectangles as a substitute
for a coherent profile. All sections/points are explicitly chosen from prompt
and reference understanding; never use object-name templates.
Coordinate contract: WORLD Z is vertical, WORLD X is longitudinal,
WORLD Y is lateral, subject to the reference. A profile_extrusion
with plane=xz is the MOST DIRECT way to make a sloping roof or tall
side elevation without confusing local coordinates: [x,z] coordinates,
depth along world Y. Do not substitute a full-height box when the
reference visibly has a trapezoidal/sloping outline.
A section_loft uses LOCAL Z for stations and LOCAL XY for width
and depth; rotate it if a horizontal or diagonal longitudinal shape is needed.
Think through final world-space bounds after all rotations and translations.
Every part may have position_mm=[x,y,z] and rotation_deg=[rx,ry,rz].
Critical axis math for ALL solids: cylinders and tubes initially point along Z.
To align their AXIS to world Y, use rotation_deg=[90,0,0], NOT [0,90,0].
To align their AXIS to world X, use rotation_deg=[0,90,0].
To keep axial orientation vertical, use rotation_deg=[0,0,0].
Reason explicitly about full world-space rotated dimensions before claiming
parts are mounted, co-axial, radial or attached.
Compose separate components with sane proportions and correct contact/overlap at joints.
Think in physical assembly relationships BEFORE choosing numeric positions:
- Z is UP. For hanging objects, the support must extend toward the ceiling.
- Parts described as coaxial, centered, or concentric share the same axis and x/y centre.
- A shaft, hub, bearing, or motor on a common rotational axis must remain aligned.
- For repeated radial attachments, position and rotate each instance about the SAME chosen centre.
- Joint ends should touch or intersect their mating parent; don't leave floating parts.
- Check the sign of every positional offset and re-check relative placement and orientation.
- Ensure every requested part is present; do not replace a part with a text label.
- Return separate editable parts, not one undifferentiated mesh.
Do not emit code, macros, loops, copies, material-only objects, or unsupported kinds.
For single assembled objects use assembly_mode=single and ensure all named parts form one physically connected assembly.\nFor intentionally separated independent objects ONLY use assembly_mode=independent.\nDo not use fan-specific, lamp-specific or any object templates.
"""


def build_cad(blueprint: CadBlueprint, output_dir: Path) -> dict:
    """Construct STEP/STL and a multi-node GLB on CPU; re-import STEP for QA."""
    # Optional heavy CAD packages are imported only in the opt-in experiment.
    import trimesh
    from build123d import (
        Align,
        Box,
        Compound,
        Cone,
        Cylinder,
        Plane,
        Polygon,
        Pos,
        Rectangle,
        Rot,
        Sphere,
        export_step,
        export_stl,
        extrude,
        import_step,
        loft,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    solids = []
    scene = trimesh.Scene()
    part_evidence = []
    evaluated_bounds: dict[str, list[list[float]]] = {}
    for part in blueprint.parts:
        if part.kind == "box":
            solid = Box(*part.dimensions_mm)
        elif part.kind == "cylinder":
            solid = Cylinder(part.radius_mm, part.height_mm, align=(Align.CENTER, Align.CENTER, Align.CENTER))
        elif part.kind == "cone":
            solid = Cone(part.radius_mm, part.top_radius_mm, part.height_mm,
                         align=(Align.CENTER, Align.CENTER, Align.CENTER))
        elif part.kind == "tube":
            outer = Cylinder(part.radius_mm, part.height_mm)
            bore = Cylinder(part.inner_radius_mm, part.height_mm + 2)
            solid = outer - bore
        elif part.kind == "profile_extrusion":
            # World-aligned sketch planes remove local/world orientation ambiguity.
            # Both-sided extrusion keeps position_mm at the profile's centre.
            base = Polygon(*part.profile_points_mm, align=None)
            plane = {"xy": Plane.XY, "xz": Plane.XZ, "yz": Plane.YZ}[part.profile_plane]
            sketch = plane * base
            solid = extrude(sketch, amount=part.height_mm / 2, both=True)
        elif part.kind == "section_loft":
            sections = [
                Pos(x, y, z) * Rectangle(width, depth)
                for z, width, depth, x, y in part.loft_sections_mm
            ]
            solid = loft(sections)
        else:
            solid = Sphere(part.radius_mm)
        solid = Pos(*part.position_mm) * Rot(*part.rotation_deg) * solid
        volume = float(solid.volume)
        if not solid.is_valid or not math.isfinite(volume) or volume < 1e-6:
            raise ValueError(f"Invalid or zero-volume solid: {part.name}")
        solids.append(solid)
        # Per-part STL is an intermediate triangulation, not the primary CAD output.
        part_stl = output_dir / f".part-{len(solids)}.stl"
        export_stl(solid, str(part_stl))
        mesh = trimesh.load_mesh(str(part_stl), force="mesh")
        if mesh.is_empty or not mesh.is_watertight:
            raise ValueError(f"Non-watertight tessellation: {part.name}")
        scene.add_geometry(mesh, geom_name=part.name, node_name=part.name)
        evaluated_bounds[part.name] = mesh.bounds.tolist()
        part_stl.unlink(missing_ok=True)
        part_evidence.append({"name": part.name, "kind": part.kind, "volume_mm3": volume,
                              "mesh_watertight": bool(mesh.is_watertight)})

    assembly = Compound(children=solids)
    step = output_dir / "assembly.step"
    stl = output_dir / "assembly.stl"
    glb = output_dir / "assembly.glb"
    export_step(assembly, str(step))
    export_stl(assembly, str(stl))
    glb.write_bytes(scene.export(file_type="glb"))

    imported = import_step(str(step))
    valid = bool(imported and imported.is_valid and imported.volume > 0)
    if not valid:
        raise ValueError("STEP re-import geometry validation failed")
    glb_check = trimesh.load(str(glb), force="scene")
    if len(glb_check.geometry) != len(blueprint.parts):
        raise ValueError("GLB part count differs from blueprint")

    sizes = {fmt: (output_dir / f"assembly.{fmt}").stat().st_size for fmt in ("step", "stl", "glb")}
    if min(sizes.values()) < 100:
        raise ValueError("CAD export unexpectedly small")
    from .cad_assembly_qa import evaluate_assembly_solids
    from .cad_preview import render_cad_views

    assembly_qa = evaluate_assembly_solids(
        {part.name: solid for part, solid in zip(blueprint.parts, solids, strict=True)},
        evaluated_bounds,
        assembly_mode=blueprint.assembly_mode,
    )
    render_files = render_cad_views(scene, output_dir)
    result = {
        "ok": True,
        "title": blueprint.title,
        "units": "mm",
        "part_count": len(part_evidence),
        "parts": part_evidence,
        "part_bounds_mm": evaluated_bounds,
        "step_reimport_valid": valid,
        "assembly_qa": assembly_qa,
        "renders": render_files,
        "exports": sizes,
        "quality_scope": "STEP integrity and BRep solid-distance screening; actual mating contact, visual similarity and print fit require independent review",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps({"blueprint": blueprint.model_dump(), "validation": result}, indent=2),
        encoding="utf-8",
    )
    return result
