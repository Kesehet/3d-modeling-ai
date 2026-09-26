from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import JOBS_ROOT, REASONING_MODEL, VISION_MODEL, WORKER_URL
from .security import require_api_token

app = FastAPI(title="3D Modeling AI", version="0.1.0")


class JobCreate(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    intended_use: str = Field(default="3d_printing", pattern="^(3d_printing|rendering|game_asset)$")
    target_width_mm: float | None = Field(default=None, gt=0, le=10_000)


def _job_dir(job_id: str) -> Path:
    if not job_id or any(ch not in "0123456789abcdef-" for ch in job_id.lower()):
        raise HTTPException(status_code=400, detail="Invalid job id")
    path = (JOBS_ROOT / job_id).resolve()
    if JOBS_ROOT not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid job path")
    return path


def _write_status(path: Path, **values: object) -> dict:
    status_file = path / "status.json"
    current: dict = {}
    if status_file.exists():
        current = json.loads(status_file.read_text(encoding="utf-8"))
    current.update(values)
    current["updated_at"] = datetime.now(UTC).isoformat()
    status_file.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current


@app.get("/health")
async def health() -> dict:
    worker = {"ok": False}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{WORKER_URL}/health")
            response.raise_for_status()
            worker = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        worker = {"ok": False, "error": str(exc)}
    return {"ok": bool(worker.get("ok")), "service": "3d-modeling-ai", "worker": worker}


@app.get("/v1/capabilities", dependencies=[Depends(require_api_token)])
async def capabilities() -> dict:
    return {
        "reasoning_model": REASONING_MODEL,
        "vision_model": VISION_MODEL,
        "blender_mcp": "djeada/blender-mcp-server@428f60cdb819c55c69d67eef681f0318e464e0e9",
        "stage": "bootstrap",
        "features": ["job-workspaces", "headless-blender", "mcp-smoke-test"],
    }


@app.post("/v1/jobs", dependencies=[Depends(require_api_token)])
async def create_job(payload: JobCreate) -> dict:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    root = _job_dir(job_id)
    for child in ("references", "scene", "renders", "exports", "logs"):
        (root / child).mkdir(parents=True, exist_ok=True)

    request = payload.model_dump()
    request["job_id"] = job_id
    request["created_at"] = datetime.now(UTC).isoformat()
    (root / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    status = _write_status(root, job_id=job_id, state="created", stage="waiting_for_orchestrator")
    return {"job_id": job_id, "status": status}


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_api_token)])
async def get_job(job_id: str) -> dict:
    root = _job_dir(job_id)
    if not root.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads((root / "status.json").read_text(encoding="utf-8"))


@app.post("/v1/jobs/{job_id}/smoke-test", dependencies=[Depends(require_api_token)])
async def smoke_test(job_id: str) -> dict:
    root = _job_dir(job_id)
    if not root.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    blend_path = str(root / "scene" / "smoke-test.blend")
    render_path = str(root / "renders" / "smoke-test.png")

    code = r"""
import bpy
from mathutils import Vector

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
cube = bpy.context.object
cube.name = "MCP_Smoke_Test_Cube"

bpy.ops.object.camera_add(location=(5, -5, 4))
camera = bpy.context.object
camera.rotation_euler = ((Vector((0, 0, 0)) - camera.location).to_track_quat("-Z", "Y")).to_euler()
bpy.context.scene.camera = camera

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.filepath = args["render_path"]

bpy.ops.wm.save_as_mainfile(filepath=args["blend_path"])
bpy.ops.render.render(write_still=True)
__result__ = {"blend_path": args["blend_path"], "render_path": args["render_path"], "object": cube.name}
"""

    _write_status(root, state="running", stage="mcp_smoke_test")
    request_payload = {
        "tool": "blender_python_exec",
        "arguments": {
            "code": code,
            "args": {"blend_path": blend_path, "render_path": render_path},
            "transport": "headless",
            "factory_startup": True,
            "timeout_seconds": 180,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=240) as client:
            response = await client.post(f"{WORKER_URL}/v1/mcp/call", json=request_payload)
            response.raise_for_status()
            result = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        _write_status(root, state="failed", stage="mcp_smoke_test", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Blender worker failed: {exc}") from exc

    status = _write_status(
        root,
        state="ready",
        stage="mcp_smoke_test_complete",
        smoke_test={"blend": blend_path, "render": render_path},
    )
    return {"job_id": job_id, "status": status, "worker_result": result}
