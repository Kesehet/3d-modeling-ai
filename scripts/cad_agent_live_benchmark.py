"""Single, bounded, unmocked research-first Ollama CAD benchmark.

Run in GitHub Actions only with explicit [cad-live] opt-in.
A CI success requires actual vision-model subject acceptance, not just STEP export.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.cad_agents import generate_agent_cad


async def run(prompt: str, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "prompt": prompt, "started_at_utc": datetime.now(UTC).isoformat(),
        "benchmark": "unmocked Ollama research-first multi-agent CAD",
    }
    try:
        _, result, root = await generate_agent_cad(
            prompt, root=output / "candidate", max_parallel=2,
        )
        report.update({
            "passed": bool(result["ok"]),
            "component_jobs": result["agent_workflow"]["component_jobs"],
            "rounds": result["agent_workflow"]["rounds"],
            "assembly": result["assembly_validation"],
            "visual_acceptance_verified": result["visual_acceptance_verified"],
            "subject_score": result["visual_model_score"],
            "exports_present": [
                name for name in ("assembly.step", "assembly.glb", "assembly.stl")
                if (root / name).is_file()
            ],
        })
    except (ValueError, RuntimeError, OSError) as exc:
        report.update({"passed": False, "failure_type": type(exc).__name__,
                       "failure": str(exc)[:700]})
    report["finished_at_utc"] = datetime.now(UTC).isoformat()
    (output / "benchmark-summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Prius")
    parser.add_argument("--output", type=Path, default=Path("cad-agent-live-artifacts"))
    arguments = parser.parse_args()
    sys.exit(asyncio.run(run(arguments.prompt, arguments.output)))
