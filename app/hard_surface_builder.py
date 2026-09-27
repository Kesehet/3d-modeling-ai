"""Human-style hard-surface cage builder for Blender.

Builds a low-poly half-cage, mirrors it, then applies a non-destructive-style
modifier workflow and optional boolean cutters / separate parts. The LLM emits
bounded modeling intent; Blender performs the actual geometry operations.
"""

from __future__ import annotations


def hard_surface_cage_script() -> str:
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
        bsdf.inputs["Roughness"].default_value = 0.38
    return material


def cage_point(axis, position, half_width, height):
    if axis == "x":
        return (position, half_width, height)
    return (half_width, position, height)


def build_cage():
    axis = str(SPEC.get("axis") or "y").lower()
    stations = SPEC.get("stations") or []
    if len(stations) < 4:
        raise RuntimeError("Hard-surface cage needs at least four stations.")
    profile_size = len(stations[0].get("profile") or [])
    if profile_size < 4:
        raise RuntimeError("Hard-surface cage profile needs at least four points.")
    if any(len(station.get("profile") or []) != profile_size for station in stations):
        raise RuntimeError("Every hard-surface cage station must use the same profile size.")

    vertices = []
    for station in stations:
        position = float(station["position"])
        for pair in station["profile"]:
            half_width = max(0.0, float(pair[0]))
            height = float(pair[1])
            vertices.append(cage_point(axis, position, half_width, height))

    faces = []
    for station_index in range(len(stations) - 1):
        a0 = station_index * profile_size
        b0 = (station_index + 1) * profile_size
        for point_index in range(profile_size - 1):
            faces.append((
                a0 + point_index,
                b0 + point_index,
                b0 + point_index + 1,
                a0 + point_index + 1,
            ))

    # Cap longitudinal ends. Profiles are half-sections whose first/last points
    # lie on the mirror plane, so these n-gons become closed after Mirror.
    faces.append(tuple(range(profile_size - 1, -1, -1)))
    last = (len(stations) - 1) * profile_size
    faces.append(tuple(last + index for index in range(profile_size)))

    mesh = bpy.data.meshes.new("HardSurfaceCage")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    body = bpy.data.objects.new(str(SPEC.get("title") or "Hard Surface Cage")[:80], mesh)
    bpy.context.collection.objects.link(body)
    body.data.materials.append(material_for(SPEC.get("color", "#B8BDC6")))

    mirror = body.modifiers.new("01 Mirror", "MIRROR")
    mirror.use_axis[0 if axis == "y" else 1] = True
    if axis == "y":
        mirror.use_axis[1] = False
    else:
        mirror.use_axis[0] = False
    mirror.use_clip = True
    mirror.use_mirror_merge = True
    mirror.merge_threshold = 0.002

    subdivision = max(0, min(2, int(SPEC.get("subdivision_levels") or 0)))
    if subdivision:
        subd = body.modifiers.new("02 Subdivision", "SUBSURF")
        subd.subdivision_type = "CATMULL_CLARK"
        subd.levels = subdivision
        subd.render_levels = subdivision

    bevel_width = max(0.0, min(0.3, float(SPEC.get("bevel_width") or 0.0)))
    if bevel_width > 0:
        bevel = body.modifiers.new("03 Bevel", "BEVEL")
        bevel.width = bevel_width
        bevel.segments = max(1, min(4, int(SPEC.get("bevel_segments") or 2)))
        bevel.limit_method = "ANGLE"

    for polygon in body.data.polygons:
        polygon.use_smooth = bool(SPEC.get("smooth", True))

    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    # Freeze the cage before booleans so cutters operate on the visible mirrored
    # shape, similar to applying a deliberate human blockout checkpoint.
    for modifier in list(body.modifiers):
        try:
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        except Exception:
            pass
    return body


def object_bounds(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return (
        Vector((
            min(point.x for point in corners),
            min(point.y for point in corners),
            min(point.z for point in corners),
        )),
        Vector((
            max(point.x for point in corners),
            max(point.y for point in corners),
            max(point.z for point in corners),
        )),
    )


def bounds_overlap(a_min, a_max, b_min, b_max, margin=0.01):
    return all(
        a_min[index] <= b_max[index] + margin
        and b_min[index] <= a_max[index] + margin
        for index in range(3)
    )


def cleanup_body_mesh(body):
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    if bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    # Repair duplicated/degenerate vertices created by mirrored boolean passes
    # before export/QA. This is a cleanup pass, not a substitute for visual QA.
    mesh = body.data
    mesh.validate(verbose=False, clean_customdata=True)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    if bm.verts:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0005)
    if bm.faces:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.validate(verbose=False, clean_customdata=True)
    mesh.update()


def add_cutter(body, item):
    shape = str(item.get("shape") or "cube").lower()
    location = tuple(float(v) for v in item.get("location", [0, 0, 0]))
    rotation = tuple(math.radians(float(v)) for v in item.get("rotation_deg", [0, 0, 0]))
    scale = tuple(max(0.02, min(20.0, float(v))) for v in item.get("scale", [1, 1, 1]))

    if shape == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=1.0, depth=2.0, location=location)
    elif shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, location=location)
    else:
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=location)

    cutter = bpy.context.object
    cutter.name = "CUT_" + str(item.get("name") or shape)[:70]
    cutter.scale = scale
    cutter.rotation_euler = rotation
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    body_min, body_max = object_bounds(body)
    cutter_min, cutter_max = object_bounds(cutter)
    if not bounds_overlap(body_min, body_max, cutter_min, cutter_max):
        bpy.data.objects.remove(cutter, do_unlink=True)
        return False

    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    modifier = body.modifiers.new("Boolean " + cutter.name, "BOOLEAN")
    modifier.operation = "DIFFERENCE"
    modifier.solver = "EXACT"
    modifier.object = cutter
    try:
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        body.data.validate(verbose=False, clean_customdata=True)
        body.data.update()
        return True
    finally:
        bpy.data.objects.remove(cutter, do_unlink=True)


def add_attachment(item):
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
            vertices=48, radius=radius, depth=delta.length, location=(start + end) / 2
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
        bpy.ops.mesh.primitive_cube_add(size=2, location=(start + end) / 2)
        obj = bpy.context.object
        radius = max(0.02, min(5.0, float(item.get("radius") or 0.2)))
        obj.scale = (radius * 1.35, radius * 0.72, delta.length / 2)
        obj.rotation_euler = delta.to_track_quat("Z", "Y").to_euler()
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    elif shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, location=location)
    elif shape == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=1, depth=2, location=location)
    elif shape == "cone":
        bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=1, radius2=0, depth=2, location=location)
    elif shape == "torus":
        bpy.ops.mesh.primitive_torus_add(
            major_radius=1, minor_radius=0.26, major_segments=48, minor_segments=16, location=location
        )
    elif shape == "frustum":
        bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=1.0, radius2=0.34, depth=2.0, location=location)
    else:
        bpy.ops.mesh.primitive_cube_add(size=2, location=location)

    obj = bpy.context.object
    obj.name = str(item.get("name") or shape)[:80]
    if shape not in {"rod", "beam"}:
        obj.scale = tuple(max(0.03, min(20.0, float(v))) for v in item.get("scale", [1, 1, 1]))
        obj.rotation_euler = tuple(math.radians(float(v)) for v in item.get("rotation_deg", [0, 0, 0]))
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material_for(item.get("color", "#808080")))
    if item.get("bevel", True):
        bevel = obj.modifiers.new("Attachment Bevel", "BEVEL")
        bevel.width = max(0.01, min(0.16, min(obj.dimensions) * 0.06))
        bevel.segments = 3
    if item.get("smooth", True):
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    return obj


body = build_cage()
applied_cutters = 0
for item in (SPEC.get("cutters") or [])[:16]:
    if add_cutter(body, item):
        applied_cutters += 1
cleanup_body_mesh(body)

objects = [body]
for item in (SPEC.get("attachments") or [])[:24]:
    objects.append(add_attachment(item))

if SPEC.get("presentation_base", False):
    all_points = []
    for obj in objects:
        all_points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    min_z = min(point.z for point in all_points)
    max_radius = max(max(abs(point.x), abs(point.y)) for point in all_points) + 0.7
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=64, radius=max(1.5, max_radius), depth=0.12, location=(0, 0, min_z - 0.08)
    )
    base = bpy.context.object
    base.name = "Presentation Base"
    base.data.materials.append(material_for("#303742"))

points = []
for obj in objects:
    points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
minimum = Vector((
    min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)
))
maximum = Vector((
    max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)
))
center = (minimum + maximum) / 2
size = maximum - minimum
span = max(size.x, size.y, size.z, 1.0)
distance = span * 2.8 + 2.0

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "MATERIAL"
scene.display.shading.show_shadows = True
scene.display.shading.show_cavity = True
scene.render.resolution_x = 384
scene.render.resolution_y = 384
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
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    path = f"{OUT}/{PREFIX}-{name}.png"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    rendered.append(path)

bpy.ops.wm.save_as_mainfile(filepath=BLEND)

bpy.ops.object.select_all(action="DESELECT")
for obj in objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = body

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
    "requested_cutters": len((SPEC.get("cutters") or [])[:16]),
    "applied_cutters": applied_cutters,
    "presentation_base": bool(SPEC.get("presentation_base", False)),
    "note": "Hard-surface cage QA. Separate attachments may require union/repair for printing.",
}
Path(QA_PATH).write_text(json.dumps(qa, indent=2), encoding="utf-8")

__result__ = {
    "blend": BLEND,
    "renders": rendered,
    "exports": exports,
    "qa": qa,
    "title": SPEC.get("title"),
    "object_count": len(objects),
    "strategy": "hard_surface_cage",
}
'''
