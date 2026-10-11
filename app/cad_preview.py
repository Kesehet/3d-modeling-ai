"""Faithful headless CPU CAD renders with a real software depth buffer.

Never drop or stride dense BRep tessellation triangles: sparse sampling of a
closed surface creates fake gaps and misleads the autonomous visual reviewer.
This is a CPU-only diagnostic renderer, not photorealistic rendering.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


CAMERAS = {
    "front-left": (1.0, -1.2, 0.75),
    "back-right": (-1.0, 1.2, 0.75),
    "left": (0.0, -1.0, 0.08),
    "right": (0.0, 1.0, 0.08),
    "top": (0.15, -0.15, 1.0),
}
PALETTE = (
    (83, 142, 207), (215, 149, 92), (96, 181, 166),
    (178, 151, 209), (211, 182, 96), (116, 155, 179),
)
BACKGROUND = (248, 250, 252)


def _rasterize_mesh(
    pixels: np.ndarray,
    zbuffer: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    color: np.ndarray,
    *,
    epsilon: float,
) -> None:
    """Rasterize every triangle using barycentric interpolation and z-buffer."""
    size_y, size_x = zbuffer.shape
    for face in faces:
        p0, p1, p2 = vertices[face]
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        left = max(0, int(np.floor(min(x0, x1, x2))))
        right = min(size_x - 1, int(np.ceil(max(x0, x1, x2))))
        top = max(0, int(np.floor(min(y0, y1, y2))))
        bottom = min(size_y - 1, int(np.ceil(max(y0, y1, y2))))
        if left > right or top > bottom:
            continue
        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(det) < 1e-6:
            continue
        xs = np.arange(left, right + 1, dtype=np.float32)[None, :] + 0.5
        ys = np.arange(top, bottom + 1, dtype=np.float32)[:, None] + 0.5
        a = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / det
        b = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / det
        c = 1 - a - b
        inside = (a >= -1e-5) & (b >= -1e-5) & (c >= -1e-5)
        if not inside.any():
            continue
        depth = a * z0 + b * z1 + c * z2
        buffer = zbuffer[top:bottom + 1, left:right + 1]
        # A small world-space bias makes near-coplanar mating surfaces stable
        # instead of producing multicolored z-fighting stripes.
        write = inside & (depth > buffer + epsilon)
        if not write.any():
            continue
        buffer[write] = depth[write]
        screen_normal = np.cross(p1 - p0, p2 - p0)
        length = float(np.linalg.norm(screen_normal))
        shade = 0.68 + 0.28 * abs(screen_normal[2]) / length if length > 1e-9 else 0.68
        pixels[top:bottom + 1, left:right + 1][write] = np.clip(
            color * shade, 0, 255
        ).astype(np.uint8)


def render_cad_views(scene, output_dir: Path) -> dict[str, str]:
    """Orthographic CPU PNGs of every evaluated triangle and every requested view."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meshes = [(name, mesh) for name, mesh in scene.geometry.items() if len(mesh.faces)]
    if not meshes:
        return {}
    points = np.concatenate([mesh.vertices for _, mesh in meshes])
    mid = (points.min(axis=0) + points.max(axis=0)) * 0.5
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    epsilon = min(0.25, max(0.001, diagonal * 0.00002))
    size = 700
    results: dict[str, str] = {}

    for name, vector in CAMERAS.items():
        eye = np.asarray(vector, dtype=float)
        eye /= np.linalg.norm(eye)
        up = np.array([0., 0., 1.])
        if abs(float(np.dot(eye, up))) > 0.95:
            up = np.array([0., 1., 0.])
        horizontal = np.cross(up, eye)
        horizontal /= np.linalg.norm(horizontal)
        vertical = np.cross(eye, horizontal)
        basis = np.vstack((horizontal, vertical, eye)).T
        overall = (points - mid) @ basis
        scale = 590.0 / max(float(np.ptp(overall[:, :2], axis=0).max()), 1.0)

        pixels = np.full((size, size, 3), BACKGROUND, dtype=np.uint8)
        zbuffer = np.full((size, size), -np.inf, dtype=np.float32)
        for index, (_, mesh) in enumerate(meshes):
            projected = (mesh.vertices - mid) @ basis
            vertices = np.empty_like(projected)
            vertices[:, 0] = size / 2 + projected[:, 0] * scale
            vertices[:, 1] = size / 2 - projected[:, 1] * scale
            vertices[:, 2] = projected[:, 2]
            _rasterize_mesh(
                pixels, zbuffer, vertices, mesh.faces,
                np.asarray(PALETTE[index % len(PALETTE)], dtype=float),
                epsilon=epsilon,
            )
        destination = output_dir / f"assembly-{name}.png"
        Image.fromarray(pixels).save(destination, optimize=True)
        results[name] = destination.name
    return results
