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
import hashlib
import json
import math
import struct
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

def mesh_signature(obj):
    """Exact world-transform, vertex-coordinate and topology digest for source meshes."""
    digest = hashlib.sha256()
    digest.update(obj.name.encode("utf-8"))
    for row in obj.matrix_world:
        for value in row:
            digest.update(struct.pack("<d", float(value)))
    for vertex in obj.data.vertices:
        digest.update(struct.pack("<3d", *(float(value) for value in vertex.co)))
    for polygon in obj.data.polygons:
        digest.update(struct.pack("<I", len(polygon.vertices)))
        for index in polygon.vertices:
            digest.update(struct.pack("<I", int(index)))
    return digest.hexdigest()


# Snapshot before installing any frozen child component. Structural preservation
# is measured, not inferred from camera angles or a multimodal model verdict.
parent_signatures = {obj.name: mesh_signature(obj) for obj in parent_objects}


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
def cross_section_profile(vertices, edges, *, longitudinal_axis, width_axis, outward_sign):
    """Calculate two true mesh edge/plane intersection widths along an oriented axis.

    Operates only on mesh vertices/edges in the frozen child's local frame;
    unlike a bounding box, this distinguishes taper from uniform maximum width.
    The caller must establish that the signed axis really points OUTWARD.
    """
    if not vertices or not edges or longitudinal_axis == width_axis:
        return {"measured": False, "reason": "No suitable mesh cross-section"}
    if outward_sign not in (-1, 1):
        return {"measured": False, "reason": "Outward long-axis sign unavailable"}
    coords = [float(v[longitudinal_axis]) * outward_sign for v in vertices]
    lower = min(coords)
    upper = max(coords)
    span = upper - lower
    if span <= 1e-7:
        return {"measured": False, "reason": "Near-zero longitudinal extent"}

    def width_at(fraction):
        plane = lower + fraction * span
        transverse = []
        for i, j in edges:
            start, end = coords[i], coords[j]
            if min(start, end) > plane + 1e-8 or max(start, end) < plane - 1e-8:
                continue
            if abs(start - end) < 1e-10:
                if abs(start - plane) < 1e-8:
                    transverse.extend((float(vertices[i][width_axis]), float(vertices[j][width_axis])))
                continue
            alpha = (plane - start) / (end - start)
            if -1e-8 <= alpha <= 1.0 + 1e-8:
                transverse.append(
                    float(vertices[i][width_axis]) +
                    alpha * (float(vertices[j][width_axis]) - float(vertices[i][width_axis]))
                )
        if len(transverse) < 2:
            return None
        width = max(transverse) - min(transverse)
        return width if width > 1e-7 else None

    root_width = width_at(0.20)
    distal_width = width_at(0.80)
    if root_width is None or distal_width is None:
        return {"measured": False, "reason": "Not enough mesh edges cross both slice planes"}
    ratio = distal_width / root_width
    direction = (
        "widens_outward" if ratio >= 1.10 else
        "narrows_outward" if ratio <= (1.0 / 1.10) else
        "approximately_uniform"
    )
    return {
        "measured": True,
        "method": "mesh_edge_intersections_at_20_and_80_percent_of_signed_longitudinal_extent",
        "root_width": round(root_width, 5),
        "distal_width": round(distal_width, 5),
        "distal_to_root_ratio": round(ratio, 4),
        "direction": direction,
        "long_axis": "XYZ"[longitudinal_axis],
        "width_axis": "XYZ"[width_axis],
        "outward_long_axis_sign": "positive" if outward_sign > 0 else "negative",
    }


# Extract the accepted frozen child in its native scene-global local-assembly
# frame, BEFORE rotations and translations of each installed instance. These
# widths remain valid under independent axis scaling and exact rotations.
source_vertices = []
source_edges = []
for source in source_objects:
    if source.type != "MESH":
        continue
    offset = len(source_vertices)
    source_matrix = source_world_matrices[source.name]
    source_vertices.extend(
        tuple(source_matrix @ vertex.co)
        for vertex in source.data.vertices
    )
    source_edges.extend(
        (offset + edge.vertices[0], offset + edge.vertices[1])
        for edge in source.data.edges
    )
if source_vertices:
    child_bounds = [
        max(point[axis] for point in source_vertices) -
        min(point[axis] for point in source_vertices)
        for axis in range(3)
    ]
    sorted_axes = sorted(range(3), key=lambda axis: child_bounds[axis], reverse=True)
else:
    child_bounds = []
    sorted_axes = []

repeated_profiles = []
alignment_contract = str(args.get("repeated_axis_alignment") or "free")
long_axis_sign = str(args.get("long_axis_sign") or "")
outward_sign = {"positive": 1, "negative": -1}.get(long_axis_sign)
instance_locations = [
    Vector(tuple(float(x) for x in item.get("location", [0, 0, 0])[:3]))
    for item in INSTANCES
]
placement_center = sum(instance_locations, Vector()) / len(instance_locations)
if len(INSTANCES) > 1:
    spreads = [
        max(v[a] for v in instance_locations) - min(v[a] for v in instance_locations)
        for a in range(3)
    ]
    symmetry_axis = min(range(3), key=lambda a: spreads[a])
else:
    symmetry_axis = None

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

    # Measure before rotation, retaining the child's independently scaled local
    # axes. Require the declared signed long axis to point radially away from the
    # placement centroid; ambiguous placement MUST NOT be labeled root vs tip.
    if (
        alignment_contract == "radial"
        and len(INSTANCES) > 1
        and len(sorted_axes) == 3
        and child_bounds[sorted_axes[0]] > child_bounds[sorted_axes[1]] * 1.10
        and outward_sign in (-1, 1)
        and symmetry_axis is not None
    ):
        long_axis, transverse_axis = sorted_axes[:2]
        radial = location_matrix.translation - placement_center
        radial[symmetry_axis] = 0.0
        axis_direction = rotation_matrix.to_3x3() @ Vector(
            tuple(outward_sign if axis == long_axis else 0.0 for axis in range(3))
        )
        axis_direction[symmetry_axis] = 0.0
        directional_cosine = (
            axis_direction.normalized().dot(radial.normalized())
            if radial.length > 1e-7 and axis_direction.length > 1e-7 else 0.0
        )
        if directional_cosine >= 0.85:
            scaled_vertices = [
                (point[0] * sx, point[1] * sy, point[2] * sz)
                for point in source_vertices
            ]
            profile = cross_section_profile(
                scaled_vertices, source_edges,
                longitudinal_axis=long_axis,
                width_axis=transverse_axis,
                outward_sign=outward_sign,
            )
            profile["radial_alignment_cosine"] = round(directional_cosine, 5)
        else:
            profile = {
                "measured": False,
                "reason": "Declared outward axis does not align radially with assembly placement",
                "radial_alignment_cosine": round(directional_cosine, 5),
            }
    else:
        profile = {
            "measured": False,
            "reason": "No unambiguous, elongated, repeated radial mesh with signed outward axis",
        }
    profile["instance_index"] = instance_index
    repeated_profiles.append(profile)

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
for obj in parent_objects:
    if obj.name not in scene.objects or mesh_signature(obj) != parent_signatures[obj.name]:
        raise RuntimeError("Frozen parent mesh geometry was mutated during component assembly.")
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
    "component_width_profiles": repeated_profiles,
    "component_axis_alignment": alignment_contract,
    "component_long_axis_sign": long_axis_sign,
    "measurement_description": (
        "True frozen-child mesh edge intersections at 20% and 80% of the declared outward "
        "local long axis, with per-instance radial direction verified; NOT a visual QA pass."
    ),
    "parent_geometry_preserved": True,
    "parent_mesh_count": len(parent_objects),
    "parent_mesh_signature_sha256": hashlib.sha256(
        "|".join(sorted(parent_signatures.values())).encode("utf-8")
    ).hexdigest(),
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
