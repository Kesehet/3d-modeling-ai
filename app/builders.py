"""Controlled Blender builders used by the first end-to-end generation tests.

These functions return Python source executed by the private Blender MCP worker.
The first builder is deliberately deterministic: it proves that the pipeline can
create something recognizable, persist the scene, and produce multi-view renders
before we allow the planner to synthesize arbitrary geometry programs.
"""

from __future__ import annotations


def pikachu_script() -> str:
    return r'''
import bpy
import math
from mathutils import Vector

OUT = args["output_dir"]
BLEND = args["blend_path"]

# Clean scene.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

def mat(name, color, metallic=0.0, roughness=0.45):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1.0)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
    return m

YELLOW = mat("Pikachu Yellow", (1.0, 0.72, 0.02))
BLACK = mat("Black", (0.015, 0.012, 0.01), roughness=0.3)
RED = mat("Cheek Red", (0.9, 0.025, 0.015), roughness=0.4)
BROWN = mat("Tail Brown", (0.32, 0.10, 0.025))
WHITE = mat("Eye Highlight", (1.0, 1.0, 1.0), roughness=0.2)
GROUND = mat("Ground", (0.12, 0.14, 0.17), roughness=0.7)

def smooth(obj):
    if obj.type == "MESH":
        for p in obj.data.polygons:
            p.use_smooth = True
    return obj

def uv(name, loc, scale, material, segments=48):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=24, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    smooth(o)
    o.data.materials.append(material)
    return o

def cone_between(name, a, b, r1, r2, material, vertices=48):
    a, b = Vector(a), Vector(b)
    delta = b - a
    mid = (a + b) / 2
    bpy.ops.mesh.primitive_cone_add(vertices=vertices, radius1=r1, radius2=r2, depth=delta.length, location=mid)
    o = bpy.context.object
    o.name = name
    o.rotation_euler = delta.to_track_quat("Z", "Y").to_euler()
    smooth(o)
    o.data.materials.append(material)
    return o

# Body/head: intentionally chubby classic silhouette.
uv("Body", (0, 0, 1.65), (1.05, 0.78, 1.25), YELLOW)
uv("Head", (0, -0.03, 3.15), (1.28, 0.95, 1.05), YELLOW)

# Muzzle/cheek volume.
uv("Muzzle", (0, -0.86, 2.92), (0.62, 0.18, 0.36), YELLOW)

# Ears, tilted outward. Black tips are separate cones.
cone_between("Ear.L", (-0.68, 0.0, 3.82), (-1.03, 0.02, 5.45), 0.38, 0.16, YELLOW)
cone_between("Ear.R", (0.68, 0.0, 3.82), (1.03, 0.02, 5.45), 0.38, 0.16, YELLOW)
cone_between("EarTip.L", (-0.93, 0.02, 5.00), (-1.03, 0.02, 5.47), 0.22, 0.13, BLACK)
cone_between("EarTip.R", (0.93, 0.02, 5.00), (1.03, 0.02, 5.47), 0.22, 0.13, BLACK)

# Arms.
cone_between("Arm.L", (-0.73, -0.12, 2.15), (-1.18, -0.45, 1.45), 0.30, 0.20, YELLOW)
cone_between("Arm.R", (0.73, -0.12, 2.15), (1.18, -0.45, 1.45), 0.30, 0.20, YELLOW)

# Feet.
uv("Foot.L", (-0.58, -0.34, 0.52), (0.52, 0.72, 0.30), YELLOW)
uv("Foot.R", (0.58, -0.34, 0.52), (0.52, 0.72, 0.30), YELLOW)

# Eyes facing camera (-Y).
for x, side in [(-0.48, "L"), (0.48, "R")]:
    uv(f"Eye.{side}", (x, -0.91, 3.36), (0.22, 0.09, 0.27), BLACK, 32)
    uv(f"EyeHighlight.{side}", (x - 0.055, -1.005, 3.45), (0.065, 0.025, 0.075), WHITE, 24)

# Nose.
uv("Nose", (0, -1.02, 3.02), (0.11, 0.055, 0.075), BLACK, 24)

# Red cheeks.
uv("Cheek.L", (-0.91, -0.82, 2.82), (0.31, 0.08, 0.30), RED, 32)
uv("Cheek.R", (0.91, -0.82, 2.82), (0.31, 0.08, 0.30), RED, 32)

# Tiny curved-ish mouth using two beveled curves.
def mouth_curve(name, pts):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.035
    curve.bevel_resolution = 3
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(pts) - 1)
    for bp, co in zip(spline.bezier_points, pts):
        bp.co = co
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(BLACK)
    return obj

mouth_curve("Mouth.L", [(0, -1.025, 2.91), (-0.16, -1.035, 2.80), (-0.30, -0.99, 2.84)])
mouth_curve("Mouth.R", [(0, -1.025, 2.91), (0.16, -1.035, 2.80), (0.30, -0.99, 2.84)])

# Lightning-bolt tail as a beveled 2D polygon extruded along Y.
verts2d = [
    (0.00, 0.00), (0.62, 0.22), (0.28, 0.62), (0.88, 0.88),
    (0.38, 1.34), (1.05, 1.70), (0.72, 2.12), (0.15, 1.78),
    (0.45, 1.42), (-0.05, 1.12), (0.28, 0.76), (-0.20, 0.48)
]
mesh = bpy.data.meshes.new("TailMesh")
verts = [(x + 0.95, 0.48, z + 1.20) for x, z in verts2d]
mesh.from_pydata(verts, [], [list(range(len(verts)))])
mesh.update()
tail = bpy.data.objects.new("Lightning Tail", mesh)
bpy.context.collection.objects.link(tail)
tail.data.materials.append(YELLOW)
solid = tail.modifiers.new("Tail Thickness", "SOLIDIFY")
solid.thickness = 0.20
bev = tail.modifiers.new("Tail Bevel", "BEVEL")
bev.width = 0.07
bev.segments = 3

# Brown tail root.
cone_between("TailRoot", (0.85, 0.45, 1.28), (1.05, 0.48, 1.75), 0.18, 0.14, BROWN, 32)

# Ground.
bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=2.4, depth=0.18, location=(0, 0, 0.05))
ground = bpy.context.object
ground.name = "Display Base"
ground.data.materials.append(GROUND)
bev = ground.modifiers.new("Base Bevel", "BEVEL")
bev.width = 0.12
bev.segments = 4

# Lighting.
bpy.ops.object.light_add(type="AREA", location=(4, -5, 7))
key = bpy.context.object
key.name = "Key"
key.data.energy = 900
key.data.shape = "DISK"
key.data.size = 5.0
key.rotation_euler = (math.radians(28), 0, math.radians(38))

bpy.ops.object.light_add(type="AREA", location=(-4, -2, 4))
fill = bpy.context.object
fill.name = "Fill"
fill.data.energy = 500
fill.data.size = 4.0
fill.rotation_euler = (math.radians(55), 0, math.radians(-55))

bpy.ops.object.light_add(type="AREA", location=(0, 4, 6))
rim = bpy.context.object
rim.name = "Rim"
rim.data.energy = 700
rim.data.size = 3.0
rim.rotation_euler = (math.radians(-35), 0, math.radians(180))

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"\nscene.display.shading.light = "STUDIO"\nscene.display.shading.color_type = "MATERIAL"\nscene.display.shading.show_shadows = True\nscene.display.shading.show_cavity = True
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
scene.world.color = (0.035, 0.045, 0.065)

# Camera and multi-view render.
bpy.ops.object.camera_add()
camera = bpy.context.object
camera.name = "QA Camera"
camera.data.lens = 55
scene.camera = camera

target = Vector((0, 0, 2.55))
views = {
    "front": (0, -9.5, 3.2),
    "front-left": (-6.8, -7.0, 4.2),
    "left": (-9.0, 0.0, 3.2),
    "back-left": (-6.8, 7.0, 4.2),
    "back": (0, 9.5, 3.2),
    "back-right": (6.8, 7.0, 4.2),
    "right": (9.0, 0.0, 3.2),
    "front-right": (6.8, -7.0, 4.2),
    "top": (0.0, -0.2, 11.5),
}
rendered = []
for name, position in views.items():
    camera.location = position
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    path = f"{OUT}/pikachu-{name}.png"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    rendered.append(path)

bpy.ops.wm.save_as_mainfile(filepath=BLEND)
__result__ = {
    "blend_path": BLEND,
    "renders": rendered,
    "object_count": len(bpy.context.scene.objects),
}
'''
