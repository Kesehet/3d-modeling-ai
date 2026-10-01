"""Install an accepted isolated component model into a parent Blender model.

The child model is treated as frozen input. This script never asks the parent
modeling worker to recreate the component; it imports the accepted geometry,
instances it at AI-approved transforms, re-renders the whole parent, and exports
the assembled candidate for regression QA.
"""

from __future__ import annotations

from .rendering import camera_framing_script


def component_assembly_script() -> str:
    return r'''
import bmesh
import bpy
import json
import math
from mathutils import Euler, Matrix, Vector
from pathlib import Path

PARENT_BLEND = args["parent_blend_path"]
COMPONENT_BLEND = args["component_blend_path"]
BLEND = args["blend_path"]
OUT = args["output_dir"]
EXPORTS = args["exports_dir"]
QA_PATH = args["qa_path"]
PREFIX = args.get("prefix", "model-v1")
COMPONENT_NAME = str(args.get("component_name") or "component")[:80]
INSTANCES = list(args.get("instances") or [])

Path(OUT).mkdir(parents=True, exist_ok=True)
Path(EXPORTS).mkdir(parents=True, exist_ok=True)

if not INSTANCES:
    raise RuntimeError("Component assembly requires at least one instance transform.")

bpy.ops.wm.open_mainfile(filepath=PARENT_BLEND)
scene = bpy.context.scene


def is_model_object(obj):
    if obj.type != "MESH":
        return False
    lowered = obj.name.lower()
    return "presentation base" not in lowered and "ground" not in lowered


parent_objects = [obj for obj in scene.objects if is_model_object(obj)]
if not parent_objects:
    raise RuntimeError("Parent model contains no mesh geometry to assemble onto.")

with bpy.data.libraries.load(COMPONENT_BLEND, link=False) as (data_from, data_to):
    data_to.objects = list(data_from.objects)

loaded_objects = [obj for obj in data_to.objects if obj is not None]
source_objects = [
    obj
    for obj in loaded_objects
    if obj.type in {"MESH", "CURVE", "SURFACE", "FONT"}
    and "presentation base" not in obj.name.lower()
    and "ground" not in obj.name.lower()
]
if not source_objects:
    raise RuntimeError("Accepted component blend contains no installable geometry.")

# Library-loaded objects are not part of a view layer yet, so matrix_world can be
# stale/identity even when the child blend stores meaningful object offsets. Stage
# the whole loaded hierarchy long enough for Blender to evaluate the frozen child's
# true world transforms, then flatten those transforms into the installed copies.
staging = bpy.data.collections.new("__component_staging__")
scene.collection.children.link(staging)
for obj in loaded_objects:
    staging.objects.link(obj)
bpy.context.view_layer.update()
source_world_matrices = {
    source.name: source.matrix_world.copy()
    for source in source_objects
}
installed_objects = []
for instance_index, instance in enumerate(INSTANCES, start=1):
    location = instance.get("location", [0.0, 0.0, 0.0])
    rotation = instance.get("rotation_deg", [0.0, 0.0, 0.0])
    scale = instance.get("scale", [1.0, 1.0, 1.0])
    location_matrix = Matrix.Translation(Vector(tuple(float(v) for v in location[:3])))
    rotation_matrix = Euler(
        tuple(math.radians(float(v)) for v in rotation[:3]),
        "XYZ",
    ).to_matrix().to_4x4()
    sx, sy, sz = (max(0.02, min(50.0, float(v))) for v in scale[:3])
    scale_matrix = Matrix.Diagonal((sx, sy, sz, 1.0))
    instance_matrix = location_matrix @ rotation_matrix @ scale_matrix

    for source in source_objects:
        clone = source.copy()
        if getattr(source, "data", None) is not None:
            clone.data = source.data.copy()
        scene.collection.objects.link(clone)
        clone.name = f"{COMPONENT_NAME}_{instance_index}_{source.name}"[:63]
        clone.parent = None
        clone.matrix_world = instance_matrix @ source_world_matrices[source.name]
        installed_objects.append(clone)

for source in loaded_objects:
    try:
        bpy.data.objects.remove(source, do_unlink=True)
    except Exception:
        pass
try:
    bpy.data.collections.remove(staging)
except Exception:
    pass

all_model_objects = [*parent_objects, *installed_objects]
bpy.context.view_layer.update()
points = []
for obj in all_model_objects:
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

for obj in list(scene.objects):
    if obj.type == "CAMERA":
        bpy.data.objects.remove(obj, do_unlink=True)

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

bpy.ops.object.select_all(action="DESELECT")
for obj in all_model_objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = all_model_objects[0]

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
for obj in all_model_objects:
    if obj.type != "MESH":
        continue
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    non_manifold += sum(1 for edge in bm.edges if not edge.is_manifold)
    loose_vertices += sum(1 for vert in bm.verts if not vert.link_edges)
    bm.free()

qa = {
    "mesh_object_count": len(all_model_objects),
    "installed_component": COMPONENT_NAME,
    "installed_instances": len(INSTANCES),
    "installed_mesh_objects": len(installed_objects),
    "non_manifold_edges": non_manifold,
    "loose_vertices": loose_vertices,
    "scene_dimensions_blender_units": [round(float(value), 4) for value in size],
    "component_source_blend": COMPONENT_BLEND,
    "parent_source_blend": PARENT_BLEND,
    "assembly_anchor": "component_global_origin",
    "print_ready": False,
    "note": "Assembly candidate requires parent-level visual acceptance before it becomes the active model.",
}
Path(QA_PATH).write_text(json.dumps(qa, indent=2), encoding="utf-8")

__result__ = {
    "blend": BLEND,
    "renders": rendered,
    "exports": exports,
    "qa": qa,
    "component_name": COMPONENT_NAME,
    "instance_count": len(INSTANCES),
    "installed_object_count": len(installed_objects),
}
'''.replace("# FRAME_QA_CAMERA", camera_framing_script())
