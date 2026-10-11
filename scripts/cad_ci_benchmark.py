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


async def evaluate(out: Path, live: bool):
    out.mkdir(parents=True, exist_ok=True)
    cad_experiment.JOBS_ROOT = out / "_temporary_jobs"
    cases = (
        {"live_fan": None} if live else OFFLINE_CASES
    )
    report = []
    for name, blueprint in cases.items():
        prompt = ("A complete ceiling fan with motor housing, a vertical mounting rod, "
                  "and three distinct long blades attached around the hub."
                  if live else "Build a parametric multi-part mechanical test object.")
        client = OllamaProxyClient() if live else OfflineClient(blueprint)
        _, result, working = await cad_experiment.generate_cad(
            prompt, os.getenv("REASONING_MODEL", "gpt-oss:120b"), client=client,
        )
        final = out / name
        final.mkdir(exist_ok=True)
        for file in working.iterdir():
            if file.is_file():
                shutil.copy2(file, final / file.name)
        result["case"] = name
        result["inference"] = "live Ollama proxy" if live else "deterministic mock response"
        report.append(result)
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="cad-artifacts")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    asyncio.run(evaluate(Path(args.output), args.live))
