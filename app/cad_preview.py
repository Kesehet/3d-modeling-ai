"""Deterministic headless orthographic previews for CPU CAD.

Uses real exported GLB triangles, not a generated or illustrative image.
This is a lightweight painter's-order preview, not a physically based renderer.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length < 1e-8:
        raise ValueError("Invalid camera direction")
    return vector / length


def render_cad_views(glb: Path, output_dir: Path, *,
                     size: int = 480, max_faces: int = 16000) -> list[Path]:
    """Create three labeled CPU previews with shared framing from the true mesh."""
    scene = trimesh.load(str(glb), force="scene")
    meshes = [mesh for mesh in scene.dump() if not mesh.is_empty]
    if not meshes:
        raise ValueError("Cannot render an empty CAD scene")
    all_bounds = np.asarray([mesh.bounds for mesh in meshes], dtype=float)
    low = all_bounds[:, 0, :].min(axis=0)
    high = all_bounds[:, 1, :].max(axis=0)
    center = (low + high) / 2
    diagonal = float(np.linalg.norm(high - low))
    if not math.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("Cannot frame a degenerate CAD scene")
    if size < 128 or size > 1024 or max_faces < 100:
        raise ValueError("Unsupported preview bounds")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    cameras = (
        ("isometric", (1.0, -1.2, 0.7)),
        ("side", (0.01, -1.0, 0.12)),
        ("top", (0.35, -0.25, 1.0)),
    )
    for label, direction in cameras:
        toward_eye = _unit(np.array(direction, dtype=float))
        up_hint = np.array((0.0, 0.0, 1.0))
        if abs(float(np.dot(up_hint, toward_eye))) > 0.94:
            up_hint = np.array((0.0, 1.0, 0.0))
        right = _unit(np.cross(up_hint, toward_eye))
        up = _unit(np.cross(toward_eye, right))
        extent = max(
            sum(abs(v) * abs(w) for v, w in zip((high - low) / 2, right)),
            sum(abs(v) * abs(w) for v, w in zip((high - low) / 2, up)),
            diagonal * 0.2,
        ) * 1.20
        scale = (size * 0.43) / extent
        polys = []
        face_count = sum(len(m.faces) for m in meshes)
        stride = max(1, math.ceil(face_count / max_faces))
        for mesh in meshes:
            verts = np.asarray(mesh.vertices, dtype=float)
            faces = np.asarray(mesh.faces, dtype=int)
            for face in faces[::stride]:
                tri = verts[face] - center
                xyz = np.column_stack((tri @ right, tri @ up, tri @ toward_eye))
                a, b, c = tri
                normal = np.cross(b - a, c - a)
                if float(np.linalg.norm(normal)) < 1e-8:
                    continue
                normal = _unit(normal)
                # Diffuse neutral material. No semantic color assumptions.
                light = max(0.0, float(np.dot(normal, _unit(np.array([0.45, -0.65, 1.0])))))
                tone = int(115 + light * 95)
                pts = [(float(size / 2 + row[0] * scale),
                        float(size / 2 - row[1] * scale)) for row in xyz]
                polys.append((float(xyz[:, 2].mean()), pts, tone))
        if not polys:
            raise ValueError("Preview contains no visible triangles")
        # Draw far faces first; approximate occlusion for review, not true z-buffering.
        polys.sort(key=lambda item: item[0])
        frame = Image.new("RGB", (size, size), (247, 248, 251))
        draw = ImageDraw.Draw(frame)
        for _, points, tone in polys:
            draw.polygon(points, fill=(tone, tone + 3, min(255, tone + 10)))
        draw.rectangle((0, 0, size - 1, size - 1), outline=(190, 198, 207), width=1)
        draw.text((12, 12), f"CAD export: {label}", fill=(32, 38, 45))
        path = output_dir / f"{label}.png"
        frame.save(path, optimize=True)
        files.append(path)
    return files
