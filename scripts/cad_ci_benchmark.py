"""Export inspectable real CAD artifacts without external LLM credentials.

This offline benchmark uses fixed model responses to validate the full CAD execution
path. It does NOT establish autonomous prompt comprehension.
Set --live to call the configured Ollama proxy and test actual generation.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

from app import cad_experiment
from app.cad_spatial import evaluate_spatial_contract
from app.ollama import OllamaProxyClient

OFFLINE_CASES = {
    "multi_part_mechanism": {
        "title": "Generic rotating mechanical assembly",
        "units": "mm",
        "parts": [
            {"name": "housing", "kind": "cylinder", "radius_mm": 22, "height_mm": 20},
            {"name": "hub", "kind": "cylinder", "radius_mm": 12, "height_mm": 12},
            {"name": "upright_rod", "kind": "cylinder", "radius_mm": 8, "height_mm": 70,
             "position_mm": [0, 0, 45]},
            {"name": "radial_arm_1", "kind": "box", "dimensions_mm": [95, 19, 5],
             "position_mm": [55, 0, 0]},
            {"name": "radial_arm_2", "kind": "box", "dimensions_mm": [95, 19, 5],
             "position_mm": [-27.5, 47.631, 0], "rotation_deg": [0, 0, 120]},
            {"name": "radial_arm_3", "kind": "box", "dimensions_mm": [95, 19, 5],
             "position_mm": [-27.5, -47.631, 0], "rotation_deg": [0, 0, 240]},
        ],
    },
    "hollow_component": {
        "title": "Hollow cylindrical spacer",
        "units": "mm",
        "parts": [{"name": "spacer", "kind": "tube", "radius_mm": 30,
                   "inner_radius_mm": 12, "height_mm": 60}],
    },
}



def _inspect_exports(directory: Path, blueprint) -> dict:
    """Independent BRep and world-space GLB inspection, not export-exists assertions."""
    import math

    import trimesh
    from build123d import import_step

    problems = []
    try:
        brep = import_step(str(directory / "assembly.step"))
        if not brep.is_valid or not math.isfinite(brep.volume) or brep.volume <= 0:
            problems.append("STEP re-import is empty or invalid")
        scene = trimesh.load(directory / "assembly.glb", force="scene")
        meshes = list(scene.dump())
        if len(meshes) != len(blueprint.parts):
            problems.append(f"GLB part count {len(meshes)} != {len(blueprint.parts)}")
        if len(meshes) == len(blueprint.parts):
            centers = [tuple(part.position_mm) for part in blueprint.parts]
            for mesh in meshes:
                if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
                    problems.append("GLB has a non-manifold/inverted/empty part")
                middle = (mesh.bounds[0] + mesh.bounds[1]) / 2
                nearest = min(centers, key=lambda pos: math.dist(middle, pos))
                if math.dist(middle, nearest) > 1.5:
                    problems.append(f"GLB part is misplaced by {math.dist(middle, nearest):.2f} mm")
                centers.remove(nearest)
        manifest = json.loads((directory / "manifest.json").read_text())
        if len(manifest["blueprint"]["parts"]) != len(blueprint.parts):
            problems.append("Manifest part count mismatch")
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        problems.append(f"Could not independently parse CAD exports: {exc}")
    return {"pass": not problems, "failures": problems}


class OfflineClient:
    def __init__(self, data):
        self.data = data

    async def chat_json(self, **kwargs):
        return SimpleNamespace(data=self.data)


async def evaluate(out: Path, live: bool):
    out.mkdir(parents=True, exist_ok=True)
    cad_experiment.JOBS_ROOT = out / "_temporary_jobs"
    cases = (
        {"live_fan": None} if live else OFFLINE_CASES
    )
    report = []
    for name, blueprint in cases.items():
        prompt = ("A physically plausible ceiling fan suspended from an overhead ceiling: "
                  "central motor housing and concentric rotor hub, "
                  "a straight downrod extending UP from the motor toward the ceiling (Z-up), "
                  "and three distinct long thin horizontal blades spaced 120 degrees apart "
                  "radially around the SAME hub centre. Preserve all part connections."
                  if live else "Build a parametric multi-part mechanical test object.")
        client = OllamaProxyClient() if live else OfflineClient(blueprint)
        plan, result, working = await cad_experiment.generate_cad(
            prompt, os.getenv("REASONING_MODEL", "gpt-oss:120b"), client=client,
        )
        final = out / name
        final.mkdir(exist_ok=True)
        for file in working.iterdir():
            if file.is_file():
                shutil.copy2(file, final / file.name)
        result["geometry_benchmark"] = _inspect_exports(final, plan)
        if live:
            result["semantic_benchmark"] = evaluate_spatial_contract(
                plan,
                coaxial_terms=("motor", "hub", "rod"),
                vertical_support_term="rod",
                supported_term="motor",
                radial_term="blade",
                radial_count=3,
            )
        elif name == "multi_part_mechanism":
            result["semantic_benchmark"] = evaluate_spatial_contract(
                plan,
                coaxial_terms=("housing", "hub", "rod"),
                vertical_support_term="rod",
                supported_term="housing",
                radial_term="radial_arm",
                radial_count=3,
            )
        result["visual_reference_verified"] = False
        result["case"] = name
        result["inference"] = "live Ollama proxy" if live else "deterministic mock response"
        report.append(result)
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if any(not item["geometry_benchmark"]["pass"] or
           not item.get("semantic_benchmark", {"pass": True})["pass"] for item in report):
        raise SystemExit("CAD benchmark failed; inspect summary.json and exported geometry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="cad-artifacts")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    asyncio.run(evaluate(Path(args.output), args.live))
