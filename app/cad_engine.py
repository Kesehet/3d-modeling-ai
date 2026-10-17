"""Opt-in CPU-only parametric CAD backend.

The LLM supplies a declarative, bounded part graph; no LLM-generated Python runs.
Object geometry is never selected by subject name.
"""
from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CadLoftSection(BaseModel):
    """A bounded transverse rectangular profile in local coordinates."""

    at_mm: float
    width_mm: float
    depth_mm: float
    center_u_mm: float = 0
    center_v_mm: float = 0


class CadPart(BaseModel):
    name: str = Field(min_length=1, max_length=60, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    kind: Literal["box", "cylinder", "cone", "sphere", "tube", "loft"]
    position_mm: tuple[float, float, float] = (0, 0, 0)
    rotation_deg: tuple[float, float, float] = (0, 0, 0)
    dimensions_mm: tuple[float, float, float] | None = None
    radius_mm: float | None = None
    top_radius_mm: float | None = None
    inner_radius_mm: float | None = None
    height_mm: float | None = None
    loft_axis: Literal["x", "y", "z"] = "z"
    loft_sections: list[CadLoftSection] | None = None

    @model_validator(mode="after")
    def check_geometry(self) -> CadPart:
        vals = [*self.position_mm, *self.rotation_deg]
        if any(not math.isfinite(v) for v in vals):
            raise ValueError("Coordinates and angles must be finite")
        if any(abs(v) > 10_000 for v in self.position_mm):
            raise ValueError("Position exceeds 10000 mm")
        if any(abs(v) > 360 for v in self.rotation_deg):
            raise ValueError("Rotation exceeds 360 degrees")
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
        elif self.kind == "loft":
            sections = self.loft_sections or []
            if not 2 <= len(sections) <= 12:
                raise ValueError("Loft requires between two and twelve sections")
            for section in sections:
                measures = (section.at_mm, section.width_mm, section.depth_mm,
                            section.center_u_mm, section.center_v_mm)
                if not all(math.isfinite(v) and abs(v) <= 20_000 for v in measures):
                    raise ValueError("Loft section contains invalid coordinates")
                if min(section.width_mm, section.depth_mm) <= 0:
                    raise ValueError("Loft cross-section must have positive dimensions")
            if any(a.at_mm >= b.at_mm for a, b in pairwise(sections)):
                raise ValueError("Loft section positions must be strictly increasing")
            sizes = ()
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
    parts: list[CadPart] = Field(min_length=1, max_length=48)

    @model_validator(mode="after")
    def unique_part_names(self) -> CadBlueprint:
        names = [part.name.lower() for part in self.parts]
        if len(names) != len(set(names)):
            raise ValueError("Each part must have a distinct name")
        return self


CAD_SYSTEM_PROMPT = """
You are a parametric CAD designer. Return ONLY the requested JSON blueprint.
Create physically plausible, separately named, correctly positioned components.
Choose geometry from the user's description; NEVER omit major requested components.
Units are millimetres. Coordinate system is Z-up. Origin should be around assembly centre.
Allowed part kinds: box, cylinder, cone, sphere, tube.
box: dimensions_mm=[x,y,z] are full sizes.
cylinder: radius_mm and height_mm (along local Z).
cone: radius_mm (bottom), top_radius_mm (top), height_mm (local Z).
sphere: radius_mm.
tube: radius_mm outer, inner_radius_mm inner, height_mm (local Z).
loft: loft_axis=x/y/z, loft_sections=[{at_mm,width_mm,depth_mm,center_u_mm,center_v_mm}, ...].
Loft sections are transverse RECTANGLES perpendicular to chosen local axis, with strictly increasing at_mm.
Use loft for tapered/sloping/curved overall silhouettes where primitive boxes would be misleading.
Every part may have position_mm=[x,y,z] and rotation_deg=[rx,ry,rz].
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
Do not use fan-specific, lamp-specific or any object templates.
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
        Pos,
        Rectangle,
        Rot,
        Sphere,
        export_step,
        export_stl,
        import_step,
        loft,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    solids = []
    scene = trimesh.Scene()
    part_evidence = []
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
        elif part.kind == "loft":
            profiles = []
            for section in part.loft_sections:
                if part.loft_axis == "x":
                    plane = Plane(origin=(section.at_mm, section.center_u_mm,
                                          section.center_v_mm),
                                  x_dir=(0, 1, 0), z_dir=(1, 0, 0))
                elif part.loft_axis == "y":
                    plane = Plane(origin=(section.center_u_mm, section.at_mm,
                                          section.center_v_mm),
                                  x_dir=(1, 0, 0), z_dir=(0, 1, 0))
                else:
                    plane = Plane(origin=(section.center_u_mm, section.center_v_mm,
                                          section.at_mm))
                profiles.append(plane * Rectangle(section.width_mm, section.depth_mm))
            solid = loft(profiles)
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
    result = {
        "ok": True,
        "title": blueprint.title,
        "units": "mm",
        "part_count": len(part_evidence),
        "parts": part_evidence,
        "step_reimport_valid": valid,
        "exports": sizes,
        "quality_scope": "geometric integrity only; not visual/semantic correctness or print assembly",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps({"blueprint": blueprint.model_dump(), "validation": result}, indent=2),
        encoding="utf-8",
    )
    return result
