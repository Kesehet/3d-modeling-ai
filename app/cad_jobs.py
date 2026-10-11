"""Connect the CPU-only CAD sidecar to regular 3D AI dashboard jobs.

The public API does not import OpenCascade or execute LLM-generated Python.
All CAD processing happens inside the optional, CPU-limited internal service.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from io import BytesIO
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from .artifacts import reserve_model_version
from .config import API_TOKEN
from .history import append_history
from .security import require_api_token

router = APIRouter(tags=["cpu-cad"])
CAD_JOBS: dict[str, asyncio.Task[None]] = {}
CAD_WORKER_URL = os.getenv("CAD_WORKER_URL", "http://cad:8084").rstrip("/")
CAD_MAX_WAIT_SECONDS = 840.0


class CadGenerateRequest(BaseModel):
    model: str | None = Field(default=None, min_length=2, max_length=120)
    reference_mode: str = Field(default="shape_guidance", pattern="^(exact_match|shape_guidance|inspiration)$")
    assembly_mode: Literal["single", "independent"] = "single"


def _load_reference_images(root) -> list[str]:
    # Main API already owns reference upload, identity checks and provenance.
    from . import main as api

    images: list[str] = []
    for record in api._usable_reference_index(root)[:3]:
        stored_name = str(record.get("stored_name") or "")
        source = root / "references" / stored_name
        if source.name != stored_name or not source.is_file():
            continue
        try:
            with Image.open(source) as original:
                frame = original.convert("RGB")
                frame.thumbnail((640, 640))
                buf = BytesIO()
                frame.save(buf, format="JPEG", quality=65, optimize=True)
            if buf.tell() <= 425_000:
                images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        except (OSError, ValueError):
            continue
    return images



async def run_cad_job(job_id: str, version: int, *, model: str | None = None,
                      reference_mode: str = "shape_guidance",
                      assembly_mode: str = "single") -> None:
    from . import main as api

    root = api._require_job(job_id)
    export_dir = root / "exports"
    artifact_id: str | None = None
    try:
        request = json.loads((root / "request.json").read_text(encoding="utf-8"))
        api._write_status(root, state="running", stage="cad_reasoning", modeling_strategy="cpu_cad")
        append_history(root, "cad_generation_started", version=version, backend="build123d")
        auth = {"Authorization": f"Bearer {API_TOKEN}"}
        references = _load_reference_images(root)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(CAD_MAX_WAIT_SECONDS, connect=10.0)
        ) as client:
            response = await client.post(
                f"{CAD_WORKER_URL}/v1/cad-experiments",
                json={"prompt": request["prompt"], "reference_images": references,
                      "reference_mode": reference_mode, "assembly_mode": assembly_mode,
                      **({"model": model} if model else {})},
                headers=auth,
            )
            if not response.is_success:
                raise RuntimeError(f"CAD worker returned {response.status_code}: {response.text[:350]}")
            payload = response.json()
            if not payload.get("ok") or not payload.get("step_reimport_valid"):
                raise RuntimeError("CAD worker reported invalid STEP geometry")
            artifact_id = uuid.UUID(str(payload["id"])).hex
            api._write_status(root, state="running", stage="cad_exporting")
            exports = {"glb": "assembly.glb", "step": "assembly.step", "stl": "assembly.stl",
                       "json": "manifest.json"}
            download_files: dict[str, bytes] = {}
            if payload.get("renders"):
                exports.update({view: f"assembly-{view}.png" for view in (
                    "front-left", "back-right", "left", "right", "top"
                )})
            for extension in exports:
                response = await client.get(
                    f"{CAD_WORKER_URL}/v1/cad-experiments/{artifact_id}/files/{extension}",
                    headers=auth,
                    timeout=90.0,
                )
                if not response.is_success or not 100 <= len(response.content) <= 25_000_000:
                    raise RuntimeError(f"CAD worker export was missing or invalid: {extension}")
                download_files[extension] = response.content
        # Transfer verified worker artifacts to the normal job before promotion.
        for extension, content in download_files.items():
            suffix = f".{extension}" if extension not in ("json", "front-left", "back-right", "left", "right", "top") else (
                "-cad.json" if extension == "json" else None
            )
            if suffix is None:
                dest = root / "renders" / f"model-v{version}-{extension}.png"
            else:
                dest = export_dir / f"model-v{version}{suffix}"
            temp = dest.with_name(f".{dest.name}.tmp")
            temp.write_bytes(content)
            temp.replace(dest)
        assembly_qa = payload.get("assembly_qa") or {"pass": None, "measurement": "not_run"}
        reference_qa = payload.get("reference_visual_qa") or {"status": "not_run_no_reference"}
        assembly_failed = assembly_qa.get("pass") is False
        visual_failed = reference_qa.get("status") == "failed"
        review_passed = reference_qa.get("status") == "passed" and not assembly_failed
        quality = {
            "status": ("failed_assembly_qa" if assembly_failed else
                       "failed_reference_visual_qa" if visual_failed else
                       "reference_visual_passed_pending_mechanical" if review_passed else
                       "geometric_integrity_only"),
            "recognizable": True if review_passed else False if visual_failed else None,
            "summary": (
                "; ".join(assembly_qa.get("failures") or []) if assembly_failed else
                "; ".join(reference_qa.get("findings") or []) if visual_failed else
                "Reference comparison passed; physical contact/print fit still requires review." if review_passed else
                "STEP and AABB checks passed; visual subject match or physical contact unverified."
            ),
        }
        qa = {
            "engine": "cpu_cad",
            "version": version,
            "part_count": payload["part_count"],
            "parts": payload["parts"],
            "step_reimport_valid": True,
            "review_required": True,
            "assembly_qa": assembly_qa,
            "reference_visual_qa": reference_qa,
            "note": payload.get("quality_scope"),
        }
        (export_dir / f"model-v{version}-qa.json").write_text(
            json.dumps(qa, indent=2), encoding="utf-8"
        )
        api._write_status(
            root,
            state="failed" if assembly_failed or visual_failed else "ready",
            stage=("cad_needs_assembly_repair" if assembly_failed else
                   "cad_needs_reference_repair" if visual_failed else
                   "cad_needs_visual_review"),
            modeling_strategy="cpu_cad",
            cad={"version": version, "part_count": payload["part_count"], "attempts": payload.get("attempts"),
                 "engine": "build123d", "visual_review_required": True,
                 "assembly_qa": assembly_qa, "reference_visual_qa": reference_qa},
            generic_model={"version": version},
            quality_gate=quality,
            error=None,
        )
        append_history(root, "cad_generation_exported", version=version,
                       part_count=payload["part_count"], formats=list(exports))
    except (httpx.HTTPError, ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        # Do not override a prior accepted model when a new CAD attempt fails.
        api._write_status(root, state="failed", stage="cad_failed",
                          modeling_strategy="cpu_cad", error=str(exc)[:600])
        append_history(root, "cad_generation_failed", version=version,
                       reason=str(exc)[:600])
    finally:
        if artifact_id:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.delete(
                        f"{CAD_WORKER_URL}/v1/cad-experiments/{artifact_id}",
                        headers={"Authorization": f"Bearer {API_TOKEN}"},
                    )
            except (httpx.HTTPError, OSError):
                pass


async def start_cad_generation(job_id: str, request: CadGenerateRequest) -> dict:
    from . import main as api

    root = api._require_job(job_id)
    existing = CAD_JOBS.get(job_id)
    if existing and not existing.done():
        raise HTTPException(status_code=409, detail="CAD generation already running")
    status = api._read_status(root)
    if status.get("state") == "running":
        raise HTTPException(status_code=409, detail="Job already running another model-generation action")
    source_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    if not 10 <= len(str(source_request.get("prompt") or "")) <= 4000:
        raise HTTPException(status_code=422, detail="CPU CAD requires a 10-4000 character text prompt")
    version = reserve_model_version(root)
    api._write_status(root, state="running", stage="cad_queued",
                      modeling_strategy="cpu_cad", error=None)
    task = asyncio.create_task(run_cad_job(
        job_id, version, model=request.model, reference_mode=request.reference_mode,
        assembly_mode=request.assembly_mode
    ))
    CAD_JOBS[job_id] = task
    task.add_done_callback(lambda done, job_id=job_id: CAD_JOBS.pop(job_id, None)
                           if CAD_JOBS.get(job_id) is done else None)
    return {"job_id": job_id, "version": version, "started": True, "engine": "cpu_cad",
            "status": api._read_status(root)}


@router.post("/dashboard/jobs/{job_id}/cad", include_in_schema=False)
async def dashboard_generate_cad(job_id: str, request: CadGenerateRequest) -> dict:
    return await start_cad_generation(job_id, request)


@router.post("/v1/jobs/{job_id}/cad", dependencies=[Depends(require_api_token)])
async def api_generate_cad(job_id: str, request: CadGenerateRequest) -> dict:
    return await start_cad_generation(job_id, request)
