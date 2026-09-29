"""Shared declarative polygon geometry for panels and detailed component shapes."""

import math


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
