"""Safe declarative Blender scene builder.

The reasoning model emits a constrained JSON SceneSpec. This module executes only
the supported operations; it does not execute model-generated Python.
"""

from __future__ import annotations

from .geometry_proof import geometry_signature_script
from .mesh_parts import mesh_part_script
from .rendering import camera_framing_script


def generic_scene_script() -> str:
    return mesh_part_script() + geometry_signature_script() + r'''
import bmesh
import bpy
import json
import math
from mathutils import Matrix, Vector
from pathlib import Path

SPEC = args["spec"]
OUT = args["output_dir"]
BLEND = args["blend_path"]
EXPORTS = args["exports_dir"]
QA_PATH = args["qa_path"]
PREFIX = args.get("prefix", "model-v1")
proof_context = args.get("preservation_context") or {}
protected_names = list(proof_context.get("protected_object_names") or [])
baseline_signatures = {}
baseline_matrices = {}
if proof_context.get("baseline_blend_path") and protected_names:
    bpy.ops.wm.open_mainfile(filepath=proof_context["baseline_blend_path"])
    bpy.context.view_layer.update()
    baseline_signatures = {
        name: geometry_signature(bpy.context.scene.objects[name])
        for name in protected_names if name in bpy.context.scene.objects
    }
    baseline_matrices = {
        name: [list(row) for row in bpy.context.scene.objects[name].matrix_world]
        for name in protected_names if name in bpy.context.scene.objects
    }
    bpy.ops.wm.read_factory_settings(use_empty=True)

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


def decorate_object(obj, item):
    if item.get("bevel", True):
        modifier = obj.modifiers.new("Safe Bevel", "BEVEL")
        modifier.width = max(0.01, min(0.18, min(obj.dimensions) * 0.08))
        modifier.segments = 3

    if item.get("smooth", True):
        for polygon in obj.data.polygons:
            polygon.use_smooth = True


def add_object(item, *, decorate=True):
    shape = item.get("shape", "cube")
    location = tuple(item.get("location", [0, 0, 0]))
    if shape in {"lathe", "sweep"}:
        create_parametric_part(item, location)
    elif shape == "mesh":
        create_mesh_part(item, location)
    elif shape == "rod":
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
        # Radius is an authored physical dimension. Ignoring it while cutters
        # honor their own size can destroy the target or distort proportions.
        primitive_radius = (
            1.0
            if item.get("dimensions")
            else max(0.02, min(5.0, float(item.get("radius") or 1.0)))
        )
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=40, ring_count=20, radius=primitive_radius, location=location
        )
    elif shape == "cylinder":
        primitive_radius = (
            1.0
            if item.get("dimensions")
            else max(0.02, min(5.0, float(item.get("radius") or 1.0)))
        )
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=48,
            radius=primitive_radius,
            depth=2,
            location=location,
        )
    elif shape == "cone":
        primitive_radius = (
            1.0
            if item.get("dimensions")
            else max(0.02, min(5.0, float(item.get("radius") or 1.0)))
        )
        bpy.ops.mesh.primitive_cone_add(
            vertices=48,
            radius1=primitive_radius,
            radius2=0,
            depth=2,
            location=location,
        )
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
        apply_part_transform(obj, item)
    obj.data.materials.append(material_for(item.get("color", "#808080")))
    if decorate:
        decorate_object(obj, item)
    return obj


objects = []
object_items = []
objects_by_name = {}
for item in SPEC.get("objects", [])[:40]:
    obj = add_object(item, decorate=False)
    objects.append(obj)
    object_items.append(item)
    objects_by_name[str(item.get("name") or obj.name)] = obj

if not objects:
    raise RuntimeError("Scene specification contains no objects.")


def radial_pattern_instances(item):
    count = max(1, min(16, int(item.get("radial_repeat_count") or 1)))
    if count == 1:
        yield dict(item)
        return

    axis = str(item.get("radial_repeat_axis") or "z").lower()
    if axis not in {"x", "y", "z"}:
        axis = "z"
    center = Vector(item.get("radial_repeat_center") or [0.0, 0.0, 0.0])
    start_deg = float(item.get("radial_repeat_start_deg") or 0.0)
    base_location = Vector(item.get("location") or [0.0, 0.0, 0.0])
    base_rotation = list(item.get("rotation_deg") or [0.0, 0.0, 0.0])
    while len(base_rotation) < 3:
        base_rotation.append(0.0)
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]

    for index in range(count):
        angle_deg = start_deg + (360.0 * index / count)
        matrix = Matrix.Rotation(math.radians(angle_deg), 4, axis.upper())
        instance = dict(item)
        instance["name"] = f"{str(item.get('name') or 'cut')}-{index + 1}"
        instance["location"] = list(center + matrix @ (base_location - center))
        rotation = list(base_rotation)
        rotation[axis_index] = float(rotation[axis_index]) + angle_deg
        instance["rotation_deg"] = rotation
        instance["radial_repeat_count"] = 1
        yield instance


def apply_boolean_cutter(item):
    target_name = str(item.get("target") or "")
    target = objects_by_name.get(target_name)
    if target is None:
        raise RuntimeError(f"Boolean cutter target does not exist: {target_name!r}")

    cutter_item = dict(item)
    cutter_item["bevel"] = False
    cutter_item["smooth"] = False
    cutter = add_object(cutter_item, decorate=False)
    cutter.name = ("CUTTER_" + str(item.get("name") or "cut"))[:80]

    modifier = target.modifiers.new(("Cut " + cutter.name)[:63], "BOOLEAN")
    modifier.operation = "DIFFERENCE"
    modifier.solver = "EXACT"
    modifier.object = cutter

    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    try:
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    except Exception as exc:
        raise RuntimeError(
            f"Boolean subtraction failed for cutter {item.get('name')!r} on {target_name!r}: {exc}"
        ) from exc

    bm = bmesh.new()
    bm.from_mesh(target.data)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    # Boolean rims are real surface discontinuities. Averaging their normals
    # into the surrounding body creates long shading streaks that visual QA
    # mistakes for broken/intersecting geometry. Preserve smooth curved bands,
    # but split normals at sharp joins in the resulting subtractive mesh.
    for edge in bm.edges:
        if edge.is_manifold and edge.calc_face_angle() > math.radians(30.0):
            edge.smooth = False
    bm.to_mesh(target.data)
    target.data.update()
    bm.free()

    bpy.data.objects.remove(cutter, do_unlink=True)


applied_cutters = 0
for item in SPEC.get("cutters", [])[:24]:
    for instance in radial_pattern_instances(item):
        if applied_cutters >= 96:
            raise RuntimeError("Scene specification expands to more than 96 boolean cutters.")
        apply_boolean_cutter(instance)
        applied_cutters += 1

for obj, item in zip(objects, object_items, strict=True):
    decorate_object(obj, item)

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
# A protected evaluated mesh is authoritative. Rebuilding its declarative spec
# can change bevel evaluation or Boolean edge normals despite identical base
# vertices. Reuse the original data/material/modifier stack for scoped repairs.
if proof_context.get("reuse_protected_geometry"):
    if set(baseline_signatures) != set(protected_names):
        raise RuntimeError("Protected baseline objects are missing; scoped repair cannot proceed.")
    for obj in list(objects):
        if obj.name in protected_names:
            objects.remove(obj)
            bpy.data.objects.remove(obj, do_unlink=True)
    with bpy.data.libraries.load(proof_context["baseline_blend_path"], link=False) as (data_from, data_to):
        data_to.objects = list(protected_names)
    for name, obj in zip(protected_names, data_to.objects):
        if obj is None or obj.type != "MESH":
            raise RuntimeError("Protected baseline mesh could not be restored.")
        bpy.context.scene.collection.objects.link(obj)
        obj.name = name
        obj.parent = None
        obj.matrix_world = Matrix(baseline_matrices[name])
        objects.append(obj)
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
    "applied_cutters": applied_cutters,
    "scene_dimensions_blender_units": [round(float(value), 4) for value in size],
    "print_ready": bool(len(objects) == 1 and non_manifold == 0 and loose_vertices == 0),
    "note": "Generic blockout QA. Multi-object scenes require a deliberate union/repair pass before printing.",
}
if proof_context:
    bpy.context.view_layer.update()
    candidate_signatures = {
        obj.name: geometry_signature(obj) for obj in objects if obj.name in protected_names
    }
    matched = [name for name in protected_names if name in baseline_signatures
               and baseline_signatures[name] == candidate_signatures.get(name)]
    qa["protected_part_proof"] = {
        **proof_context,
        "verified": bool(protected_names) and len(matched) == len(protected_names),
        "matched_object_names": matched,
        "baseline_signatures": baseline_signatures,
        "candidate_signatures": candidate_signatures,
        "method": "Blender evaluated mesh vertices, topology, world matrices and material diffuse colors",
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
