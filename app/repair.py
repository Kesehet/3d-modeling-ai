"""Experimental Blender print-repair pass.

The repair deliberately creates a new scene/export instead of mutating the source
artifact. Voxel remesh is useful for fusing intersecting visible parts, but this is
not a substitute for wall-thickness, self-intersection, tolerance, or slicer QA.
"""

from __future__ import annotations


def print_repair_script() -> str:
    return r'''
import bmesh
import bpy
import json
from pathlib import Path

SOURCE = args["source_blend"]
OUT_BLEND = args["output_blend"]
OUT_STL = args["output_stl"]
QA_PATH = args["qa_path"]
VOXEL = max(0.01, min(0.20, float(args.get("voxel_size", 0.05))))
TARGET_WIDTH_MM = args.get("target_width_mm")

bpy.ops.wm.open_mainfile(filepath=SOURCE)
scene = bpy.context.scene
scene.unit_settings.system = "METRIC"
scene.unit_settings.length_unit = "MILLIMETERS"

source_objects = [obj for obj in scene.objects if obj.type in {"MESH", "CURVE"}]
if not source_objects:
    raise RuntimeError("The source scene has no printable mesh/curve objects.")

# Convert curves and apply visible modifiers on a copy saved to a new output file.
converted = []
for obj in list(source_objects):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    if obj.type == "CURVE":
        bpy.ops.object.convert(target="MESH")
        obj = bpy.context.object
    if obj.type == "MESH":
        for modifier in list(obj.modifiers):
            try:
                bpy.ops.object.modifier_apply(modifier=modifier.name)
            except Exception:
                pass
        converted.append(obj)

if not converted:
    raise RuntimeError("No mesh objects remained after conversion.")

# Join visible parts, then voxel-remesh them into one volumetric surface.
bpy.ops.object.select_all(action="DESELECT")
for obj in converted:
    obj.select_set(True)
bpy.context.view_layer.objects.active = converted[0]
bpy.ops.object.join()
model = bpy.context.object
model.name = "Printable_Repaired_Model"

bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
model.data.remesh_mode = "VOXEL"
model.data.remesh_voxel_size = VOXEL
model.data.remesh_voxel_adaptivity = 0.0
bpy.context.view_layer.objects.active = model
model.select_set(True)
bpy.ops.object.voxel_remesh()

# Basic mesh cleanup after volumetric fusion.
bm = bmesh.new()
bm.from_mesh(model.data)
bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=max(0.001, VOXEL * 0.15))
loose = [vert for vert in bm.verts if not vert.link_edges]
if loose:
    bmesh.ops.delete(bm, geom=loose, context="VERTS")
boundary = [edge for edge in bm.edges if edge.is_boundary]
if boundary:
    try:
        bmesh.ops.holes_fill(bm, edges=boundary, sides=0)
    except Exception:
        pass
if bm.faces:
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
bm.to_mesh(model.data)
bm.free()
model.data.update()

# Optional uniform size target from the job.
if TARGET_WIDTH_MM:
    target = float(TARGET_WIDTH_MM)
    current_width = float(model.dimensions.x)
    if target > 0 and current_width > 0:
        factor = target / current_width
        model.scale = tuple(component * factor for component in model.scale)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# Final QA.
bm = bmesh.new()
bm.from_mesh(model.data)
non_manifold = sum(1 for edge in bm.edges if not edge.is_manifold)
loose_vertices = sum(1 for vert in bm.verts if not vert.link_edges)

unvisited = set(bm.verts)
components = 0
while unvisited:
    components += 1
    stack = [unvisited.pop()]
    while stack:
        vert = stack.pop()
        for edge in vert.link_edges:
            other = edge.other_vert(vert)
            if other in unvisited:
                unvisited.remove(other)
                stack.append(other)
bm.free()

dimensions = [round(float(value), 3) for value in model.dimensions]
qa = {
    "repair_method": "join + voxel_remesh + remove_doubles + hole_fill + normal_recalc",
    "source_mesh_objects": len(converted),
    "mesh_object_count": 1,
    "connected_components": components,
    "non_manifold_edges": non_manifold,
    "loose_vertices": loose_vertices,
    "dimensions_mm": dimensions if TARGET_WIDTH_MM else None,
    "scene_dimensions_blender_units": dimensions,
    "voxel_size": VOXEL,
    "print_ready": bool(components == 1 and non_manifold == 0 and loose_vertices == 0),
    "checks_not_yet_implemented": [
        "minimum wall thickness",
        "self intersections",
        "overhang/support analysis",
        "slicer validation",
        "functional clearances",
    ],
}
qa["note"] = (
    "Experimental repair pass. A clean manifold/component result is necessary but not sufficient "
    "for a safe or successful print."
)

Path(QA_PATH).write_text(json.dumps(qa, indent=2), encoding="utf-8")
bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND)

bpy.ops.object.select_all(action="DESELECT")
model.select_set(True)
bpy.context.view_layer.objects.active = model
bpy.ops.wm.stl_export(
    filepath=OUT_STL,
    export_selected_objects=True,
    apply_modifiers=True,
    use_scene_unit=False,
)

__result__ = {
    "source": SOURCE,
    "blend": OUT_BLEND,
    "stl": OUT_STL,
    "qa_path": QA_PATH,
    "qa": qa,
}
'''
