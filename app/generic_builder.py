"""Safe declarative Blender scene builder.

The reasoning model emits a constrained JSON SceneSpec. This module executes only
the supported operations; it does not execute model-generated Python.
"""

from __future__ import annotations

from .rendering import camera_framing_script


def generic_scene_script() -> str:
    return r'''
import bmesh
import bpy
import json
import math
from mathutils import Vector
from pathlib import Path

SPEC = args["spec"]
OUT = args["output_dir"]
BLEND = args["blend_path"]
EXPORTS = args["exports_dir"]
QA_PATH = args["qa_path"]
PREFIX = args.get("prefix", "model-v1")

Path(OUT).mkdir(parents=True, exist_ok=True)
Path(EXPORTS).mkdir(parents=True, exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)


def rgb(hex_color):
    value = str(hex_color or "#808080").lstrip("#")
    if len(value) != 6:
        value = "808080"
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def material_for(hex_color):
    key = "MAT_" + str(hex_color).replace("#", "").upper()
    existing = bpy.data.materials.get(key)
    if existing:
        return existing
    color = rgb(hex_color)
    material = bpy.data.materials.new(key)
    material.diffuse_color = (*color, 1.0)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.42
    return material


def add_object(item):
    shape = item.get("shape", "cube")
    location = tuple(item.get("location", [0, 0, 0]))
    if shape == "rod":
        start = Vector(item.get("start") or location)
        end = Vector(item.get("end") or [location[0], location[1], location[2] + 1.0])
        delta = end - start
        if delta.length < 0.02:
            delta = Vector((0, 0, 0.02))
            end = start + delta
        radius = max(0.02, min(5.0, float(item.get("radius") or 0.2)))
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=48,
            radius=radius,
            depth=delta.length,
            location=(start + end) / 2,
        )
        obj = bpy.context.object
        obj.rotation_euler = delta.to_track_quat("Z", "Y").to_euler()
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    elif shape == "beam":
        start = Vector(item.get("start") or location)
        end = Vector(item.get("end") or [location[0], location[1], location[2] + 1.0])
        delta = end - start
        if delta.length < 0.02:
            delta = Vector((0, 0, 0.02))
            end = start + delta
        radius = max(0.02, min(5.0, float(item.get("radius") or 0.2)))
        bpy.ops.mesh.primitive_cube_add(size=2, location=(start + end) / 2)
        obj = bpy.context.object
        obj.scale = (radius * 1.35, radius * 0.72, delta.length / 2)
        obj.rotation_euler = delta.to_track_quat("Z", "Y").to_euler()
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    elif shape == "frustum":
        bpy.ops.mesh.primitive_cone_add(
            vertices=64,
            radius1=1.0,
            radius2=0.34,
            depth=2.0,
            location=location,
        )
    elif shape == "wedge":
        verts = [
            (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1, 1), (1, -1, 0.18), (1, 1, 0.18), (-1, 1, 1),
        ]
        faces = [
            (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
            (3, 2, 6, 7), (0, 3, 7, 4), (1, 5, 6, 2),
        ]
        mesh = bpy.data.meshes.new("WedgeMesh")
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        obj = bpy.data.objects.new("Wedge", mesh)
        bpy.context.collection.objects.link(obj)
        obj.location = location
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
    elif shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=40, ring_count=20, location=location)
    elif shape == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=1, depth=2, location=location)
    elif shape == "cone":
        bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=1, radius2=0, depth=2, location=location)
    elif shape == "torus":
        bpy.ops.mesh.primitive_torus_add(
            major_radius=1,
            minor_radius=0.26,
            major_segments=48,
            minor_segments=16,
            location=location,
        )
    else:
        bpy.ops.mesh.primitive_cube_add(size=2, location=location)

    obj = bpy.context.object
    obj.name = str(item.get("name") or shape)[:80]
    if shape not in {"rod", "beam"}:
        scale = item.get("scale", [1, 1, 1])
        obj.scale = tuple(max(0.03, min(20.0, float(value))) for value in scale)
        rotation = item.get("rotation_deg", [0, 0, 0])
        obj.rotation_euler = tuple(math.radians(float(value)) for value in rotation)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material_for(item.get("color", "#808080")))

    if item.get("bevel", True):
        modifier = obj.modifiers.new("Safe Bevel", "BEVEL")
        modifier.width = max(0.01, min(0.18, min(obj.dimensions) * 0.08))
        modifier.segments = 3

    if item.get("smooth", True):
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    return obj


objects = []
for item in SPEC.get("objects", [])[:40]:
    objects.append(add_object(item))

if not objects:
    raise RuntimeError("Scene specification contains no objects.")

# A neutral presentation base is separate from model geometry.
if SPEC.get("presentation_base", True):
    all_points = []
    for obj in objects:
        all_points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    min_z = min(point.z for point in all_points)
    max_radius = max(max(abs(point.x), abs(point.y)) for point in all_points) + 0.7
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=64,
        radius=max(1.5, max_radius),
        depth=0.12,
        location=(0, 0, min_z - 0.08),
    )
    base = bpy.context.object
    base.name = "Presentation Base"
    base.data.materials.append(material_for("#303742"))

# Compute model bounds (excluding presentation base).
bpy.context.view_layer.update()
points = []
for obj in objects:
    points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
minimum = Vector((
    min(point.x for point in points),
    min(point.y for point in points),
    min(point.z for point in points),
))
maximum = Vector((
    max(point.x for point in points),
    max(point.y for point in points),
    max(point.z for point in points),
))
center = (minimum + maximum) / 2
size = maximum - minimum
span = max(size.x, size.y, size.z, 1.0)
distance = span * 2.8 + 2.0

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "MATERIAL"
scene.display.shading.background_type = "VIEWPORT"
scene.display.shading.background_color = (0.88, 0.90, 0.93)
scene.display.shading.show_shadows = True
scene.display.shading.show_cavity = True
scene.render.resolution_x = 640
scene.render.resolution_y = 640
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"

bpy.ops.object.camera_add()
camera = bpy.context.object
camera.name = "QA Camera"
camera.data.lens = 55
scene.camera = camera

z_eye = center.z + span * 0.16
views = {
    "front": (center.x, center.y - distance, z_eye),
    "front-left": (center.x - distance * 0.72, center.y - distance * 0.72, z_eye + span * 0.12),
    "left": (center.x - distance, center.y, z_eye),
    "back-left": (center.x - distance * 0.72, center.y + distance * 0.72, z_eye + span * 0.12),
    "back": (center.x, center.y + distance, z_eye),
    "back-right": (center.x + distance * 0.72, center.y + distance * 0.72, z_eye + span * 0.12),
    "right": (center.x + distance, center.y, z_eye),
    "front-right": (center.x + distance * 0.72, center.y - distance * 0.72, z_eye + span * 0.12),
    "top": (center.x, center.y - 0.01, center.z + distance),
}
rendered = []
for name, position in views.items():
    camera.location = position
# FRAME_QA_CAMERA
    path = f"{OUT}/{PREFIX}-{name}.png"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    rendered.append(path)

bpy.ops.wm.save_as_mainfile(filepath=BLEND)

# Export only authored model objects; exclude cameras and presentation base.
bpy.ops.object.select_all(action="DESELECT")
for obj in objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = objects[0]

exports = {}
glb_path = str(Path(EXPORTS) / f"{PREFIX}.glb")
obj_path = str(Path(EXPORTS) / f"{PREFIX}.obj")
stl_path = str(Path(EXPORTS) / f"{PREFIX}.stl")
try:
    bpy.ops.export_scene.gltf(filepath=glb_path, export_format="GLB", use_selection=True)
    exports["glb"] = glb_path
except Exception as exc:
    exports["glb_error"] = str(exc)
try:
    bpy.ops.wm.obj_export(filepath=obj_path, export_selected_objects=True)
    exports["obj"] = obj_path
except Exception as exc:
    exports["obj_error"] = str(exc)
try:
    bpy.ops.wm.stl_export(filepath=stl_path, export_selected_objects=True)
    exports["stl"] = stl_path
except Exception as exc:
    exports["stl_error"] = str(exc)

non_manifold = 0
loose_vertices = 0
for obj in objects:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    non_manifold += sum(1 for edge in bm.edges if not edge.is_manifold)
    loose_vertices += sum(1 for vert in bm.verts if not vert.link_edges)
    bm.free()

qa = {
    "mesh_object_count": len(objects),
    "potentially_disconnected_parts": len(objects),
    "non_manifold_edges": non_manifold,
    "loose_vertices": loose_vertices,
    "scene_dimensions_blender_units": [round(float(value), 4) for value in size],
    "print_ready": bool(len(objects) == 1 and non_manifold == 0 and loose_vertices == 0),
    "note": "Generic blockout QA. Multi-object scenes require a deliberate union/repair pass before printing.",
}
Path(QA_PATH).write_text(json.dumps(qa, indent=2), encoding="utf-8")

__result__ = {
    "blend": BLEND,
    "renders": rendered,
    "exports": exports,
    "qa": qa,
    "title": SPEC.get("title"),
    "object_count": len(objects),
}
'''.replace("# FRAME_QA_CAMERA", camera_framing_script())
