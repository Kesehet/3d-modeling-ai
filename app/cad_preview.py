"""Headless, GPU-free shaded diagnostic renders of actual exported CAD meshes.

Views are independent of Blender and use only Pillow/numpy (already pulled
in by the CAD engine). They are evidence for review, not proof of accuracy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def render_cad_views(scene, output_dir: Path) -> dict[str, str]:
    """Orthographic PNGs from evaluated per-part geometry, with stable framing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras = {
        "front-left": (1.0, -1.2, 0.75),
        "back-right": (-1.0, 1.2, 0.75),
        "left": (0.0, -1.0, 0.08),
        "right": (0.0, 1.0, 0.08),
        "top": (0.15, -0.15, 1.0),
    }
    results: dict[str, str] = {}
    meshes = [(name, mesh) for name, mesh in scene.geometry.items() if len(mesh.faces)]
    if not meshes:
        return results
    points = np.concatenate([mesh.vertices for _, mesh in meshes])
    mid = (points.min(axis=0) + points.max(axis=0)) * 0.5
    pal = [(83, 142, 207), (215, 149, 92), (96, 181, 166),
           (178, 151, 209), (211, 182, 96), (116, 155, 179)]

    for view_name, vector in cameras.items():
        eye = np.asarray(vector, dtype=float)
        eye /= np.linalg.norm(eye)
        world_up = np.array([0., 0., 1.])
        if abs(float(np.dot(eye, world_up))) > 0.95:
            world_up = np.array([0., 1., 0.])
        horizontal = np.cross(world_up, eye)
        horizontal /= np.linalg.norm(horizontal)
        vertical = np.cross(eye, horizontal)
        basis = np.vstack((horizontal, vertical, eye)).T
        projected_global = (points - mid) @ basis
        extent = np.ptp(projected_global[:, :2], axis=0)
        size = 700
        scale = 590.0 / max(float(extent.max()), 1.0)
        canvas = Image.new("RGB", (size, size), "#f8fafc")
        draw = ImageDraw.Draw(canvas)
        faces_to_draw = []
        for index, (_, mesh) in enumerate(meshes):
            verts = (mesh.vertices - mid) @ basis
            faces = mesh.faces
            stride = max(1, len(faces) // 4500)
            color = np.array(pal[index % len(pal)], dtype=float)
            for triangle in faces[::stride]:
                coords = verts[triangle]
                a, b, c = coords
                normal = np.cross(b - a, c - a)
                norm = float(np.linalg.norm(normal))
                if norm < 1e-12:
                    continue
                shade = 0.68 + 0.28 * abs(float(normal[2])) / norm
                fill = tuple(int(min(255, max(0, item * shade))) for item in color)
                pixel = [(
                    int(size / 2 + x * scale),
                    int(size / 2 - y * scale),
                ) for x, y in coords[:, :2]]
                faces_to_draw.append((float(coords[:, 2].mean()), pixel, fill))
        # Orthographic painter's order: farthest surfaces first.
        for _, coords, fill in sorted(faces_to_draw, key=lambda entry: entry[0]):
            draw.polygon(coords, fill=fill)
        destination = output_dir / f"assembly-{view_name}.png"
        canvas.save(destination, optimize=True)
        results[view_name] = destination.name
    return results
