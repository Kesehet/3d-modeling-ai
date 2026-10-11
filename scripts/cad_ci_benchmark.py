"""Export inspectable real CAD artifacts without external LLM credentials.

This offline benchmark uses fixed model responses to validate the full CAD execution
path. It does NOT establish autonomous prompt comprehension.
Set --live to call the configured Ollama proxy and test actual generation.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

from app import cad_experiment
from app.ollama import OllamaProxyClient

OFFLINE_CASES = {
    "multi_part_mechanism": {
        "title": "Generic rotating mechanical assembly",
        "units": "mm",
        "parts": [
            {"name": "shaft", "kind": "cylinder", "radius_mm": 8, "height_mm": 70},
            {"name": "housing", "kind": "cylinder", "radius_mm": 22, "height_mm": 20},
            {"name": "radial_arm_1", "kind": "box", "dimensions_mm": [95, 19, 5],
             "position_mm": [55, 0, 0]},
            {"name": "radial_arm_2", "kind": "box", "dimensions_mm": [95, 19, 5],
             "position_mm": [-55, 0, 0]},
            {"name": "radial_arm_3", "kind": "box", "dimensions_mm": [95, 19, 5],
             "position_mm": [0, 55, 0], "rotation_deg": [0, 0, 90]},
        ],
    },
    "hollow_component": {
        "title": "Hollow cylindrical spacer",
        "units": "mm",
        "parts": [{"name": "spacer", "kind": "tube", "radius_mm": 30,
                   "inner_radius_mm": 12, "height_mm": 60}],
    },
}


class OfflineClient:
    def __init__(self, data):
        self.data = data

    async def chat_json(self, **kwargs):
        return SimpleNamespace(data=self.data)


def evaluate_spatial_contract(
    blueprint,
    *,
    coaxial_terms: tuple[str, ...],
    vertical_support_term: str,
    supported_term: str,
    radial_term: str,
    radial_count: int,
) -> dict:
    """Benchmark-only spatial QA. Never changes CAD geometry."""
    parts = blueprint.parts
    findings = []
    centers = {}
    for term in coaxial_terms:
        matches = [p for p in parts if term in p.name.lower()]
        if len(matches) != 1:
            findings.append(f"Expected one part containing {term!r}, got {len(matches)}")
        else:
            centers[term] = matches[0]
    if len(centers) == len(coaxial_terms):
        anchor = centers[coaxial_terms[0]].position_mm
        for term, part in centers.items():
            xy_offset = math.hypot(part.position_mm[0] - anchor[0],
                                   part.position_mm[1] - anchor[1])
            if xy_offset > 5:
                findings.append(f"{term!r} is {xy_offset:.1f} mm off central axis")
    supports = [p for p in parts if vertical_support_term in p.name.lower()]
    parents = [p for p in parts if supported_term in p.name.lower()]
    if len(supports) == 1 and len(parents) == 1:
        support, parent = supports[0], parents[0]
        if support.height_mm is None or support.position_mm[2] - support.height_mm / 2 < parent.position_mm[2]:
            findings.append("Vertical support does not extend upward from parent")
    arms = [p for p in parts if radial_term in p.name.lower()]
    if len(arms) != radial_count:
        findings.append(f"Expected {radial_count} radial parts, got {len(arms)}")
    elif coaxial_terms[0] in centers:
        origin = centers[coaxial_terms[0]].position_mm
        angles = []
        for part in arms:
            dx = part.position_mm[0] - origin[0]
            dy = part.position_mm[1] - origin[1]
            if math.hypot(dx, dy) < 5:
                findings.append(f"Radial part {part.name} lies on central axis")
            else:
                angles.append(math.atan2(dy, dx) % (2 * math.pi))
        if len(angles) == radial_count:
            angles.sort()
            gaps = [(angles[(i + 1) % radial_count] - angles[i]) % (2 * math.pi)
                    for i in range(radial_count)]
            if any(abs(gap - 2 * math.pi / radial_count) > 0.14 for gap in gaps):
                findings.append("Radial parts are not evenly spaced")
    return {"pass": not findings, "failures": findings}


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
        if live:
            result["semantic_benchmark"] = evaluate_spatial_contract(
                plan,
                coaxial_terms=("motor", "hub", "rod"),
                vertical_support_term="rod",
                supported_term="motor",
                radial_term="blade",
                radial_count=3,
            )
        result["case"] = name
        result["inference"] = "live Ollama proxy" if live else "deterministic mock response"
        report.append(result)
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if live and any(not item["semantic_benchmark"]["pass"] for item in report):
        raise SystemExit("Spatial benchmark failed; see uploaded artifact for diagnostics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="cad-artifacts")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    asyncio.run(evaluate(Path(args.output), args.live))
