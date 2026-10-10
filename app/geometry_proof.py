"""Blender-derived identity evidence for explicitly protected scene objects."""

import inspect


def geometry_signature(obj):
    import hashlib
    import struct

    import bpy

    digest = hashlib.sha256()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        for row in evaluated.matrix_world:
            for value in row:
                digest.update(struct.pack("<d", float(value)))
        for vertex in mesh.vertices:
            digest.update(struct.pack("<3d", *(float(value) for value in vertex.co)))
        for polygon in mesh.polygons:
            digest.update(struct.pack("<II", len(polygon.vertices), polygon.material_index))
            for index in polygon.vertices:
                digest.update(struct.pack("<I", int(index)))
        for material in mesh.materials:
            if material is not None:
                digest.update(struct.pack("<4d", *material.diffuse_color))
    finally:
        evaluated.to_mesh_clear()
    return digest.hexdigest()


def geometry_signature_script():
    return inspect.getsource(geometry_signature)
