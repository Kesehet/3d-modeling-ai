"""Opt-in CAD experiment API using the existing authenticated Ollama proxy."""
from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError

from .cad_engine import CAD_SYSTEM_PROMPT, CadBlueprint, build_cad
from .config import JOBS_ROOT, REASONING_MODEL
from .ollama import OllamaProxyClient, OllamaProxyError
from .security import require_api_token

router = APIRouter(prefix="/v1/cad-experiments", tags=["cad-experiments"],
                   dependencies=[Depends(require_api_token)])
CAD_SEMAPHORE = asyncio.Semaphore(1)
EXPORTS = {"step": "assembly.step", "stl": "assembly.stl",
           "glb": "assembly.glb", "json": "manifest.json"}


class CadRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    model: str | None = Field(default=None, min_length=2, max_length=120)
    agent_mode: bool = False


def _enabled() -> None:
    if os.environ.get("CAD_EXPERIMENT_ENABLED") != "1":
        raise HTTPException(404, "Experimental CPU CAD backend disabled")


def _root() -> Path:
    return JOBS_ROOT / ".cad-experiments"


async def generate_cad(prompt: str, model: str, *, client: OllamaProxyClient | None = None) -> tuple[CadBlueprint, dict, Path]:
    """Generate and validate a CAD assembly; errors become model retry context."""
    client = client or OllamaProxyClient()
    experiment_id = uuid.uuid4().hex
    final_dir = _root() / experiment_id
    final_dir.mkdir(parents=True, exist_ok=False)
    error = ""
    try:
        for attempt in range(1, 4):
            query = prompt if not error else (
                f"{prompt}\n\nPrevious attempt failed. Correct your entire JSON blueprint: {error[:700]}"
            )
            response = await client.chat_json(
                model=model,
                system=CAD_SYSTEM_PROMPT,
                prompt=query,
                schema=CadBlueprint.model_json_schema(),
                temperature=0.15,
            )
            try:
                blueprint = CadBlueprint.model_validate(response.data)
            except (ValidationError, ValueError, TypeError) as exc:
                error = f"Blueprint schema: {exc}"
                continue
            try:
                # Build on CPU. This deliberately does NOT execute LLM-written Python.
                result = await asyncio.to_thread(build_cad, blueprint, final_dir)
            except (ValueError, RuntimeError, OSError) as exc:
                error = f"CAD construction/export: {exc}"
                for file in final_dir.iterdir():
                    if file.is_file():
                        file.unlink(missing_ok=True)
                continue
            result["attempts"] = attempt
            result["id"] = experiment_id
            return blueprint, result, final_dir
        raise ValueError(f"No valid CAD model after 3 attempts: {error[:800]}")
    except BaseException:
        shutil.rmtree(final_dir, ignore_errors=True)
        raise


@router.post("")
async def create_experiment(payload: CadRequest) -> dict:
    _enabled()
    if CAD_SEMAPHORE.locked():
        raise HTTPException(429, "CPU CAD worker busy; retry later")
    async with CAD_SEMAPHORE:
        try:
            if payload.agent_mode and os.getenv("CAD_AGENT_WORKFLOW_ENABLED") != "1":
                raise HTTPException(404, "Research-first CAD agents are disabled")
            if payload.agent_mode:
                from .cad_agents import generate_agent_cad

                ident = uuid.uuid4().hex
                _, report, _ = await generate_agent_cad(
                    payload.prompt, payload.model or REASONING_MODEL,
                    root=_root() / ident,
                )
                report["id"] = ident
            else:
                _, report, _ = await generate_cad(payload.prompt, payload.model or REASONING_MODEL)
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
