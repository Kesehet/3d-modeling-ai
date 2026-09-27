"""Immutable model version allocation, including interrupted builds."""

import re
from pathlib import Path


def reserve_model_version(root: Path) -> int:
    scene = root / "scene"
    scene.mkdir(parents=True, exist_ok=True)
    versions = [0]
    for folder in (root, scene, root / "renders", root / "exports"):
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            match = re.search(r"(?:model|cage-spec|mesh-spec|scene-spec)-v(\d+)(?:\D|$)", path.name)
            if match:
                versions.append(int(match.group(1)))
    version = max(versions) + 1
    while True:
        try:
            # Exclusive creation also protects concurrent allocators. Keep the
            # reservation after failure: partial exports must never be reused.
            (scene / f".model-v{version}.reserve").touch(exist_ok=False)
            return version
        except FileExistsError:
            version += 1


def require_unused_version(root: Path, version: int) -> None:
    for folder in (root / "scene", root / "renders", root / "exports"):
        if any(folder.glob(f"model-v{version}.*")) or any(folder.glob(f"model-v{version}-*")):
            raise ValueError(f"Model v{version} already has artifacts; refusing to overwrite a checkpoint.")
