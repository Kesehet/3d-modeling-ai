"""Shared declarative polygon geometry for panels and detailed component shapes."""

import math

from pydantic import BaseModel, Field, model_validator


class ParametricGeometry(BaseModel):
    dimensions: list[float] | None = Field(
        default=None, min_length=3, max_length=3,
        description="Full local XYZ bounding-box dimensions BEFORE rotation, overriding scale. Prefer this for primitives; a box of height H has dimensions[2]=H, not 2H.",
    )
    profile: list[tuple[float, float]] = Field(
        default_factory=list, max_length=96,
        description="For shape=lathe: ordered [radius, local Z] points around a CLOSED material cross-section, revolved about local Z. Include outer and inner walls for a hollow form; radius=0 closes on the axis. No duplicate closing point needed.",
    )
    path: list[tuple[float, float, float]] = Field(
        default_factory=list, max_length=64,
        description="For shape=sweep: local XYZ centerline points. A round tube of the supplied radius follows this smooth open path, with closed end caps.",
    )
    segments: int = Field(default=64, ge=8, le=128, description="Angular resolution of a lathe.")

    @model_validator(mode="after")
    def validate_parametric_geometry(self):
        if self.dimensions is not None and any(
            not math.isfinite(v) or v <= 0 or v > 40 for v in self.dimensions
        ):
            raise ValueError("Full dimensions must be finite, positive and at most 40.")
        shape = getattr(self, "shape", None)
        if shape == "lathe":
            if len(self.profile) < 3 or len(set(self.profile)) < 3:
                raise ValueError("A lathe needs at least three distinct profile points.")
            if any(not math.isfinite(v) or abs(v) > 20 for p in self.profile for v in p):
                raise ValueError("Lathe profile coordinates must be finite and within +/-20.")
            if any(p[0] < 0 for p in self.profile) or max(p[0] for p in self.profile) <= 0:
                raise ValueError("Lathe radii must be nonnegative with positive outer radius.")
            area = sum(a[0]*b[1] - b[0]*a[1] for a, b in zip(self.profile, self.profile[1:]+self.profile[:1]))
            if abs(area) < 1e-8:
                raise ValueError("Lathe profile must enclose material area.")
        if shape == "sweep":
            if len(self.path) < 2 or any(a == b for a, b in zip(self.path, self.path[1:])):
                raise ValueError("Sweep needs a nondegenerate centerline with at least two points.")
            if any(not math.isfinite(v) or abs(v) > 20 for p in self.path for v in p):
                raise ValueError("Sweep path coordinates must be finite and within +/-20.")
            if not getattr(self, "radius", None):
                raise ValueError("Sweep requires an explicit tube radius.")
        return self


def validate_mesh_part(shape, vertices, faces):
    if shape != "mesh":
        return
    if len(vertices) < 3 or not faces:
        raise ValueError("Mesh parts require vertices and polygon faces.")
    if any(not math.isfinite(v) or abs(v) > 20 for point in vertices for v in point):
        raise ValueError("Mesh coordinates must be finite and within +/-20 local units.")
    if any(len(set(face)) != len(face) or any(i < 0 or i >= len(vertices) for i in face) for face in faces):
        raise ValueError("Mesh faces must use distinct, valid vertex indices.")


def geometry_context(value):
    """Keep dense mesh data out of prompts while preserving editable part identity and bounds."""
    if isinstance(value, list):
        return [geometry_context(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: geometry_context(item) for key, item in value.items() if key not in {"vertices", "faces"}}
    if value.get("shape") == "mesh" and value.get("vertices"):
        vertices = value["vertices"]
        result["mesh_summary"] = {
            "vertex_count": len(vertices), "face_count": len(value.get("faces") or []),
            "local_min": [min(p[i] for p in vertices) for i in range(3)],
            "local_max": [max(p[i] for p in vertices) for i in range(3)],
        }
    return result


def mesh_part_script() -> str:
    return r'''
def create_parametric_part(item, location):
    if item["shape"] == "lathe":
        profile = list(item["profile"])
        if profile[0] == profile[-1]:
            profile.pop()
        count = int(item.get("segments", 64))
        vertices, rings, faces = [], [], []
        for radius, z in profile:
            if radius == 0:
                rings.append([len(vertices)])
                vertices.append((0, 0, z))
            else:
                rings.append(list(range(len(vertices), len(vertices) + count)))
                vertices.extend((radius*math.cos(2*math.pi*i/count), radius*math.sin(2*math.pi*i/count), z)
                                for i in range(count))
        for a, b in zip(rings, rings[1:] + rings[:1]):
            if len(a) == len(b) == 1:
                continue
            for i in range(count):
                n = (i + 1) % count
                if len(a) == 1:
                    faces.append((a[0], b[n], b[i]))
                elif len(b) == 1:
                    faces.append((a[i], a[n], b[0]))
                else:
                    faces.append((a[i], a[n], b[n], b[i]))
        return create_mesh_part({**item, "vertices": vertices, "faces": faces}, location)
    curve = bpy.data.curves.new(str(item.get("name") or "Sweep"), "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 12
    curve.bevel_depth = float(item["radius"])
    curve.bevel_resolution = 4
    curve.use_fill_caps = True
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(item["path"]) - 1)
    for point, coordinate in zip(spline.bezier_points, item["path"]):
        point.co = coordinate
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(curve.name, curve)
    bpy.context.collection.objects.link(obj)
    bpy.ops.object.select_all(action="DESELECT")
    obj.location = location
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.convert(target="MESH")
    return bpy.context.object


def apply_part_transform(obj, item):
    dimensions = item.get("dimensions")
    if dimensions:
        native = [max(v.co[i] for v in obj.data.vertices) - min(v.co[i] for v in obj.data.vertices)
                  for i in range(3)]
        if any(v < 1e-9 for v in native):
            raise ValueError("Cannot size a flat part in all three dimensions; use scale for a surface mesh.")
        obj.scale = tuple(float(dimensions[i]) / native[i] for i in range(3))
    else:
        obj.scale = tuple(max(0.03, min(20.0, float(v))) for v in item.get("scale", [1, 1, 1]))
    obj.rotation_euler = tuple(math.radians(float(v)) for v in item.get("rotation_deg", [0, 0, 0]))
    # Apply only to this part; previously selected parts must never be rescaled.
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def create_mesh_part(item, location):
    mesh = bpy.data.meshes.new(str(item.get("name") or "Mesh part"))
    mesh.from_pydata(item["vertices"], [], item["faces"])
    mesh.validate(verbose=False, clean_customdata=True)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.ops.object.select_all(action="DESELECT")
    obj.location = location
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return obj

'''
