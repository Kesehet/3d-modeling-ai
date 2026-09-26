"""Controlled deterministic Blender builders used by regression and refinement jobs."""

from __future__ import annotations


def pikachu_script() -> str:
    """Return a parameter-driven Pikachu benchmark script for headless Blender."""
    return r'''
import bmesh
import bpy
import json
import math
from mathutils import Vector
from pathlib import Path

OUT = args["output_dir"]
BLEND = args["blend_path"]
EXPORTS = args.get("exports_dir", OUT)
QA_PATH = args.get("qa_path")
PREFIX = args.get("prefix", "pikachu")
P = {
    "head_width": 1.0,
    "head_height": 1.0,
    "body_width": 1.0,
    "body_height": 1.0,
    "ear_length": 1.0,
    "eye_spacing": 1.0,
    "cheek_scale": 1.0,
    "foot_scale": 1.0,
    "tail_scale": 1.0,
}
P.update(args.get("params") or {})
for key, value in list(P.items()):
    P[key] = max(0.72, min(1.32, float(value)))

Path(OUT).mkdir(parents=True, exist_ok=True)
Path(EXPORTS).mkdir(parents=True, exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)


def mat(name, color, metallic=0.0, roughness=0.45):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1.0)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
    return material


YELLOW = mat("Pikachu Yellow", (1.0, 0.72, 0.02))
BLACK = mat("Black", (0.015, 0.012, 0.01), roughness=0.3)
RED = mat("Cheek Red", (0.9, 0.025, 0.015), roughness=0.4)
BROWN = mat("Tail Brown", (0.32, 0.10, 0.025))
WHITE = mat("Eye Highlight", (1.0, 1.0, 1.0), roughness=0.2)
GROUND = mat("Ground", (0.12, 0.14, 0.17), roughness=0.7)


def smooth(obj):
    if obj.type == "MESH":
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    return obj


def uv(name, loc, scale, material, segments=48):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=24, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    smooth(obj)
    obj.data.materials.append(material)
    return obj


def cone_between(name, start, end, r1, r2, material, vertices=48):
    a, b = Vector(start), Vector(end)
    delta = b - a
    mid = (a + b) / 2
    bpy.ops.mesh.primitive_cone_add(
        vertices=vertices,
        radius1=r1,
        radius2=r2,
        depth=delta.length,
        location=mid,
    )
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = delta.to_track_quat("Z", "Y").to_euler()
    smooth(obj)
    obj.data.materials.append(material)
    return obj


# Improved baseline proportions; all key proportions remain tunable.
body_w = 1.02 * P["body_width"]
body_h = 1.18 * P["body_height"]
head_w = 1.30 * P["head_width"]
head_h = 1.02 * P["head_height"]
uv("Body", (0, 0.03, 1.62), (body_w, 0.80 * P["body_width"], body_h), YELLOW)
uv("Head", (0, -0.04, 3.12), (head_w, 0.98 * P["head_width"], head_h), YELLOW)
uv("Muzzle", (0, -0.91, 2.91), (0.60, 0.17, 0.33), YELLOW)

ear_top = 5.05 * P["ear_length"]
ear_shoulder = 3.78
cone_between("Ear.L", (-0.67, 0.02, ear_shoulder), (-0.98, 0.03, ear_top), 0.36, 0.14, YELLOW)
cone_between("Ear.R", (0.67, 0.02, ear_shoulder), (0.98, 0.03, ear_top), 0.36, 0.14, YELLOW)
tip_start = ear_shoulder + (ear_top - ear_shoulder) * 0.72
cone_between("EarTip.L", (-0.89, 0.03, tip_start), (-0.98, 0.03, ear_top + 0.02), 0.22, 0.11, BLACK)
cone_between("EarTip.R", (0.89, 0.03, tip_start), (0.98, 0.03, ear_top + 0.02), 0.22, 0.11, BLACK)

cone_between("Arm.L", (-0.72, -0.10, 2.12), (-1.10, -0.42, 1.46), 0.29, 0.18, YELLOW)
cone_between("Arm.R", (0.72, -0.10, 2.12), (1.10, -0.42, 1.46), 0.29, 0.18, YELLOW)

foot = P["foot_scale"]
uv("Foot.L", (-0.56, -0.38, 0.48), (0.50 * foot, 0.68 * foot, 0.28 * foot), YELLOW)
uv("Foot.R", (0.56, -0.38, 0.48), (0.50 * foot, 0.68 * foot, 0.28 * foot), YELLOW)

eye_x = 0.46 * P["eye_spacing"]
for x, side in [(-eye_x, "L"), (eye_x, "R")]:
    uv(f"Eye.{side}", (x, -0.95, 3.34), (0.21, 0.085, 0.27), BLACK, 32)
    highlight_x = x - 0.052 if x < 0 else x + 0.052
    uv(f"EyeHighlight.{side}", (highlight_x, -1.03, 3.43), (0.062, 0.022, 0.073), WHITE, 24)

uv("Nose", (0, -1.055, 3.00), (0.10, 0.052, 0.07), BLACK, 24)
cheek = 0.29 * P["cheek_scale"]
uv("Cheek.L", (-0.91, -0.86, 2.79), (cheek, 0.075, cheek), RED, 32)
uv("Cheek.R", (0.91, -0.86, 2.79), (cheek, 0.075, cheek), RED, 32)


def mouth_curve(name, points):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.032
    curve.bevel_resolution = 3
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for bp, co in zip(spline.bezier_points, points):
        bp.co = co
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(BLACK)
    return obj


mouth_curve("Mouth.L", [(0, -1.055, 2.90), (-0.15, -1.06, 2.80), (-0.29, -1.01, 2.84)])
mouth_curve("Mouth.R", [(0, -1.055, 2.90), (0.15, -1.06, 2.80), (0.29, -1.01, 2.84)])

tail_scale = P["tail_scale"]
verts2d = [
    (0.00, 0.00), (0.62, 0.20), (0.28, 0.60), (0.90, 0.88),
    (0.38, 1.34), (1.08, 1.70), (0.72, 2.16), (0.12, 1.80),
    (0.44, 1.42), (-0.08, 1.10), (0.27, 0.73), (-0.22, 0.45),
]
mesh = bpy.data.meshes.new("TailMesh")
verts = [(1.08 + x * tail_scale, 0.22, 1.10 + z * tail_scale) for x, z in verts2d]
mesh.from_pydata(verts, [], [list(range(len(verts)))])
mesh.update()
tail = bpy.data.objects.new("Lightning Tail", mesh)
bpy.context.collection.objects.link(tail)
tail.data.materials.append(YELLOW)
solid = tail.modifiers.new("Tail Thickness", "SOLIDIFY")
solid.thickness = 0.22
bevel = tail.modifiers.new("Tail Bevel", "BEVEL")
bevel.width = 0.065
bevel.segments = 3
cone_between("TailRoot", (0.88, 0.18, 1.24), (1.10, 0.22, 1.68), 0.18, 0.13, BROWN, 32)

# Simple back markings make rear-view QA less ambiguous.
for x in (-0.38, 0.38):
    uv("BackStripe", (x, 0.79, 2.12), (0.28, 0.055, 0.18), BROWN, 24)

bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=2.35, depth=0.16, location=(0, 0, 0.04))
ground = bpy.context.object
ground.name = "Display Base"
ground.data.materials.append(GROUND)
base_bevel = ground.modifiers.new("Base Bevel", "BEVEL")
base_bevel.width = 0.10
base_bevel.segments = 4

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "MATERIAL"
scene.display.shading.show_shadows = True
scene.display.shading.show_cavity = True
scene.display.shading.cavity_type = "WORLD"
scene.render.resolution_x = 384
scene.render.resolution_y = 384
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False

bpy.ops.object.camera_add()
camera = bpy.context.object
camera.name = "QA Camera"
camera.data.lens = 55
scene.camera = camera
target = Vector((0, 0, 2.50))
views = {
    "front": (0, -9.2, 3.15),
    "front-left": (-6.5, -6.7, 4.0),
    "left": (-8.8, 0.0, 3.15),
    "back-left": (-6.5, 6.7, 4.0),
    "back": (0, 9.2, 3.15),
    "back-right": (6.5, 6.7, 4.0),
    "right": (8.8, 0.0, 3.15),
    "front-right": (6.5, -6.7, 4.0),
    "top": (0.0, -0.2, 11.0),
}
rendered = []
for name, position in views.items():
    camera.location = position
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    path = f"{OUT}/{PREFIX}-{name}.png"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    rendered.append(path)

bpy.ops.wm.save_as_mainfile(filepath=BLEND)

exports = {}
glb_path = str(Path(EXPORTS) / f"{PREFIX}.glb")
obj_path = str(Path(EXPORTS) / f"{PREFIX}.obj")
stl_path = str(Path(EXPORTS) / f"{PREFIX}.stl")
try:
    bpy.ops.export_scene.gltf(filepath=glb_path, export_format="GLB")
    exports["glb"] = glb_path
except Exception as exc:
    exports["glb_error"] = str(exc)
try:
    bpy.ops.wm.obj_export(filepath=obj_path)
    exports["obj"] = obj_path
except Exception as exc:
    exports["obj_error"] = str(exc)
try:
    bpy.ops.wm.stl_export(filepath=stl_path)
    exports["stl"] = stl_path
except Exception as exc:
    exports["stl_error"] = str(exc)

mesh_objects = [obj for obj in scene.objects if obj.type == "MESH"]
non_manifold_edges = 0
loose_vertices = 0
for obj in mesh_objects:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    non_manifold_edges += sum(1 for edge in bm.edges if not edge.is_manifold)
    loose_vertices += sum(1 for vert in bm.verts if not vert.link_edges)
    bm.free()

bounds = []
for obj in mesh_objects:
    for corner in obj.bound_box:
        bounds.append(obj.matrix_world @ Vector(corner))
if bounds:
    minimum = Vector((min(v.x for v in bounds), min(v.y for v in bounds), min(v.z for v in bounds)))
    maximum = Vector((max(v.x for v in bounds), max(v.y for v in bounds), max(v.z for v in bounds)))
    dimensions = [round(v, 4) for v in (maximum - minimum)]
else:
    dimensions = [0.0, 0.0, 0.0]

qa = {
    "mesh_object_count": len(mesh_objects),
    "potentially_disconnected_parts": len(mesh_objects),
    "non_manifold_edges": non_manifold_edges,
    "loose_vertices": loose_vertices,
    "scene_dimensions_blender_units": dimensions,
    "print_ready": False,
    "note": "First-pass diagnostics only; separate visible parts are not yet unioned for printing.",
}
if QA_PATH:
    Path(QA_PATH).write_text(json.dumps(qa, indent=2), encoding="utf-8")

__result__ = {
    "blend_path": BLEND,
    "renders": rendered,
    "exports": exports,
    "qa": qa,
    "params": P,
    "object_count": len(scene.objects),
}
'''
