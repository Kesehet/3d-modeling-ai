"""Opt-in CAD experiment API using the existing authenticated Ollama proxy."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError

from .cad_engine import CAD_SYSTEM_PROMPT, CadBlueprint, build_cad
from .config import JOBS_ROOT, REASONING_MODEL, VISION_MODEL
from .ollama import OllamaProxyClient, OllamaProxyError
from .security import require_api_token

router = APIRouter(prefix="/v1/cad-experiments", tags=["cad-experiments"],
                   dependencies=[Depends(require_api_token)])
CAD_SEMAPHORE = asyncio.Semaphore(1)
EXPORTS = {"step": "assembly.step", "stl": "assembly.stl",
           "glb": "assembly.glb", "json": "manifest.json",
           "front-left": "assembly-front-left.png", "top": "assembly-top.png"}


class CadRequest(BaseModel):
    prompt: str = Field(min_length=10, max_length=4000)
    model: str | None = Field(default=None, min_length=2, max_length=120)
    reference_images: list[str] = Field(default_factory=list, max_length=3)
    reference_mode: str = Field(default="shape_guidance", pattern="^(exact_match|shape_guidance|inspiration)$")


class CadVisualReview(BaseModel):
    passed: bool
    subject_recognizable: bool
    reference_match_score: float = Field(ge=0, le=1)
    findings: list[str] = Field(default_factory=list, max_length=8)


async def _review_reference_match(
    client: OllamaProxyClient,
    prompt: str,
    references: list[str],
    render_dir: Path,
    mode: str,
) -> dict:
    if not references:
        return {"status": "not_run_no_reference", "passed": None}
    previews = [
        base64.b64encode((render_dir / filename).read_bytes()).decode("ascii")
        for filename in ("assembly-front-left.png", "assembly-top.png")
    ]
    instructions = (
        f"User request: {prompt}\nComparison policy: {mode}.\n"
        f"The FIRST {len(references)} images are original reference photographs. "
        "The FINAL TWO are actual CAD isometric and top renders, respectively. "
        "Do not confuse them. Judge geometry, part count, silhouette, attachments "
        "and proportions. Do not use the label or prompt as evidence that a "
        "visually unrecognizable model is correct. For inspiration mode ignore "
        "cosmetic differences but reject missing required structural features."
    )
    try:
        raw = await client.chat_json(
            model=VISION_MODEL,
            system="Independent CAD comparison judge; use the supplied pixels, not imagined geometry.",
            prompt=instructions,
            images=[*references, *previews],
            schema=CadVisualReview.model_json_schema(),
            temperature=0.0,
        )
        review = CadVisualReview.model_validate(raw.data).model_dump()
        review["status"] = "passed" if review["passed"] else "failed"
        return review
    except (OllamaProxyError, ValueError, TypeError) as exc:
        return {"status": "unverified", "passed": None, "error": str(exc)[:300]}



def _enabled() -> None:
    if os.environ.get("CAD_EXPERIMENT_ENABLED") != "1":
        raise HTTPException(404, "Experimental CPU CAD backend disabled")


def _root() -> Path:
    return JOBS_ROOT / ".cad-experiments"


async def generate_cad(
    prompt: str,
    model: str,
    *,
    client: OllamaProxyClient | None = None,
    reference_images: list[str] | None = None,
    reference_mode: str = "shape_guidance",
) -> tuple[CadBlueprint, dict, Path]:
    """Repair invalid assemblies with measured feedback; retain the last real export."""
    client = client or OllamaProxyClient()
    reference_images = reference_images or []
    experiment_id = uuid.uuid4().hex
    final_dir = _root() / experiment_id
    final_dir.mkdir(parents=True, exist_ok=False)
    error = ""
    last_export: tuple[CadBlueprint, dict, Path] | None = None
    try:
        for attempt in range(1, 4):
            query = prompt if not error else (
                f"{prompt}\n\nPrevious candidate failed independent QA. "
                f"Return an improved complete JSON blueprint fixing measured issues: {error[:1200]}"
            )
            if reference_images:
                query += (
                    f"\nThere are {len(reference_images)} original reference images. "
                    f"Reference policy: {reference_mode}. Infer dimensions, layout and "
                    "silhouette from visible evidence; do not invent unseen details."
                )
            response = await client.chat_json(
                model=VISION_MODEL if reference_images else model,
                system=CAD_SYSTEM_PROMPT,
                prompt=query,
                images=reference_images or None,
                schema=CadBlueprint.model_json_schema(),
                temperature=0.15,
            )
            try:
                blueprint = CadBlueprint.model_validate(response.data)
            except (ValidationError, ValueError, TypeError) as exc:
                error = f"Blueprint schema: {exc}"
                continue
            attempt_dir = final_dir / f"attempt-{attempt}"
            try:
                report = await asyncio.to_thread(build_cad, blueprint, attempt_dir)
            except (ValueError, RuntimeError, OSError) as exc:
                error = f"CAD construction/export: {exc}"
                shutil.rmtree(attempt_dir, ignore_errors=True)
                continue
            last_export = (blueprint, report, attempt_dir)
            if not report["assembly_qa"]["pass"]:
                error = "; ".join(report["assembly_qa"]["failures"])
                continue
            review = await _review_reference_match(
                client, prompt, reference_images, attempt_dir, reference_mode
            )
            report["reference_visual_qa"] = review
            if review["status"] == "failed":
                error = "Reference visual review: " + "; ".join(review.get("findings") or [
                    "Shape/subject does not sufficiently match reference"
                ])
                continue
            break

        if last_export is None:
            raise ValueError(f"No valid CAD model after 3 attempts: {error[:800]}")
        blueprint, report, chosen_dir = last_export
        for file in chosen_dir.iterdir():
            if file.is_file():
                shutil.copy2(file, final_dir / file.name)
        report["attempts"] = attempt
        report["id"] = experiment_id
        report["repair_exhausted"] = bool(
            not report["assembly_qa"]["pass"]
            or report.get("reference_visual_qa", {}).get("status") == "failed"
        )
        report.setdefault("reference_visual_qa", {"status": "not_run_due_to_assembly", "passed": None})
        (final_dir / "manifest.json").write_text(
            json.dumps({"blueprint": blueprint.model_dump(), "validation": report}, indent=2),
            encoding="utf-8",
        )
        for directory in final_dir.glob("attempt-*"):
            if directory.is_dir():
                shutil.rmtree(directory)
        return blueprint, report, final_dir
    except BaseException:
        shutil.rmtree(final_dir, ignore_errors=True)
        raise


@router.post("")
async def create_experiment(payload: CadRequest) -> dict:
    _enabled()
    if any(len(item) > 600_000 for item in payload.reference_images):
        raise HTTPException(413, "Reference image exceeds the compressed image limit")
    if CAD_SEMAPHORE.locked():
        raise HTTPException(429, "CPU CAD worker busy; retry later")
    async with CAD_SEMAPHORE:
        try:
            _, report, _ = await generate_cad(
                payload.prompt, payload.model or REASONING_MODEL,
                reference_images=payload.reference_images, reference_mode=payload.reference_mode,
            )
        except (OllamaProxyError, ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(422, detail=str(exc)[:1000]) from exc
    ident = report["id"]
    report["files"] = {extension: f"/v1/cad-experiments/{ident}/files/{extension}"
                       for extension in EXPORTS}
    return report


@router.get("/{experiment_id}/files/{extension}")
async def get_export(experiment_id: uuid.UUID, extension: str):
    _enabled()
    if extension not in EXPORTS:
        raise HTTPException(404, "Export format not supported")
    file = _root() / experiment_id.hex / EXPORTS[extension]
    if not file.is_file():
        raise HTTPException(404, "Export not found")
    return FileResponse(file, filename=f"cad-{experiment_id.hex}.{extension}",
                        media_type="application/octet-stream")


@router.delete("/{experiment_id}")
async def delete_experiment(experiment_id: uuid.UUID) -> dict:
    _enabled()
    path = _root() / experiment_id.hex
    if path.is_dir():
        shutil.rmtree(path)
    return {"deleted": True}
