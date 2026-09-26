from __future__ import annotations

import base64
import hashlib
import json
import uuid
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError

from .builders import pikachu_script
from .config import (
    JOBS_ROOT,
    OLLAMA_PROXY_BASE_URL,
    REASONING_MODEL,
    VISION_MODEL,
    WORKER_URL,
)
from .dashboard import dashboard_page, jobs_snapshot, public_artifact, public_render
from .ollama import OllamaProxyClient, OllamaProxyError
from .security import require_api_token

app = FastAPI(title="3D Modeling AI", version="0.2.0")

Image.MAX_IMAGE_PIXELS = 40_000_000

MAX_REFERENCE_FILES = 8
MAX_REFERENCE_BYTES = 12 * 1024 * 1024
MAX_REFERENCE_TOTAL_BYTES = 48 * 1024 * 1024
IMAGE_FORMATS = {
    "JPEG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
    "WEBP": ("image/webp", ".webp"),
}
ARTIFACT_CATEGORIES = {"references", "scene", "renders", "exports", "logs"}


class JobCreate(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    intended_use: str = Field(default="3d_printing", pattern="^(3d_printing|rendering|game_asset)$")
    target_width_mm: float | None = Field(default=None, gt=0, le=10_000)


class VisionAnalyzeRequest(BaseModel):
    stage: str = Field(default="reference_analysis", min_length=1, max_length=80)
    include_references: bool = True
    include_renders: bool = True
    max_images: int = Field(default=10, ge=1, le=16)
    instruction: str | None = Field(default=None, max_length=4000)


class VisionIssue(BaseModel):
    object: str = ""
    issue: str
    severity: Literal["low", "medium", "high"] = "medium"
    suggested_change: str


class VisionReport(BaseModel):
    summary: str
    recommended_modeling_strategy: Literal["procedural", "base_mesh", "hybrid"]
    observations: list[str] = Field(default_factory=list)
    issues: list[VisionIssue] = Field(default_factory=list)
    priority_actions: list[str] = Field(default_factory=list)


class PlanRequest(BaseModel):
    instruction: str | None = Field(default=None, max_length=4000)


class ModelingStage(BaseModel):
    name: str
    objective: str
    success_criteria: list[str] = Field(default_factory=list)


class ModelingPlan(BaseModel):
    workflow: Literal["procedural", "base_mesh", "hybrid"]
    units: Literal["mm", "cm", "m"] = "mm"
    assumptions: list[str] = Field(default_factory=list)
    stages: list[ModelingStage]
    final_checks: list[str] = Field(default_factory=list)


def _job_dir(job_id: str) -> Path:
    if not job_id or any(ch not in "0123456789abcdef-" for ch in job_id.lower()):
        raise HTTPException(status_code=400, detail="Invalid job id")
    path = (JOBS_ROOT / job_id).resolve()
    if JOBS_ROOT not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid job path")
    return path


def _require_job(job_id: str) -> Path:
    root = _job_dir(job_id)
    if not root.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return root


def _write_status(path: Path, **values: object) -> dict:
    status_file = path / "status.json"
    current: dict = {}
    if status_file.exists():
        current = json.loads(status_file.read_text(encoding="utf-8"))
    current.update(values)
    current["updated_at"] = datetime.now(UTC).isoformat()
    status_file.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current


def _write_llm_log(root: Path, prefix: str, payload: dict) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"{prefix}-{stamp}.json"
    (root / "logs" / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return filename


def _worker_blender_error(result: dict) -> str | None:
    if result.get("is_error"):
        return "Blender MCP reported a tool error."
    for item in result.get("content", []):
        if not isinstance(item, str):
            continue
        try:
            payload = json.loads(item)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("error"):
            return str(payload["error"])
    return None


def _image_info(data: bytes) -> tuple[str, str, int, int]:
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            fmt = (image.format or "").upper()
            width, height = image.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="One uploaded file is not a supported image.") from exc

    if fmt not in IMAGE_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {fmt or 'unknown'}")
    mime, extension = IMAGE_FORMATS[fmt]
    return mime, extension, width, height


def _load_reference_index(root: Path) -> list[dict]:
    path = root / "references.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _collect_images(root: Path, request: VisionAnalyzeRequest) -> tuple[list[str], list[str]]:
    candidates: list[Path] = []
    if request.include_references:
        candidates.extend(sorted((root / "references").glob("*")))
    if request.include_renders:
        candidates.extend(sorted((root / "renders").glob("*"), key=lambda item: item.stat().st_mtime))

    image_paths: list[Path] = []
    for path in candidates:
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        image_paths.append(path)
    image_paths = image_paths[-request.max_images :]

    encoded = [base64.b64encode(path.read_bytes()).decode("ascii") for path in image_paths]
    labels = [f"{path.parent.name}/{path.name}" for path in image_paths]
    return encoded, labels


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    return dashboard_page()


@app.get("/dashboard/api", include_in_schema=False)
async def dashboard_api() -> dict:
    return jobs_snapshot()


@app.get("/dashboard/renders/{job_id}/{filename}", include_in_schema=False)
async def dashboard_render(job_id: str, filename: str) -> FileResponse:
    return FileResponse(public_render(job_id, filename))


@app.get("/dashboard/artifacts/{job_id}/{category}/{filename}", include_in_schema=False)
async def dashboard_artifact(job_id: str, category: str, filename: str) -> FileResponse:
    path = public_artifact(job_id, category, filename)
    return FileResponse(path, filename=path.name)


@app.post("/dashboard/jobs", include_in_schema=False)
async def dashboard_create_job(payload: JobCreate) -> dict:
    return await create_job(payload)


@app.post("/dashboard/jobs/{job_id}/pikachu", include_in_schema=False)
async def dashboard_run_pikachu(job_id: str) -> dict:
    return await generate_pikachu_test(job_id)


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
    return {
        "ok": bool(worker.get("ok")),
        "service": "3d-modeling-ai",
        "worker": worker,
        "ollama_proxy": OLLAMA_PROXY_BASE_URL,
    }


@app.get("/v1/capabilities", dependencies=[Depends(require_api_token)])
async def capabilities() -> dict:
    return {
        "reasoning_model": REASONING_MODEL,
        "vision_model": VISION_MODEL,
        "ollama_proxy": OLLAMA_PROXY_BASE_URL,
        "blender_mcp": "djeada/blender-mcp-server@428f60cdb819c55c69d67eef681f0318e464e0e9",
        "stage": "vision-and-planning",
        "features": [
            "job-workspaces",
            "reference-image-upload",
            "multi-image-vision",
            "modeling-plan",
            "headless-blender",
            "mcp-smoke-test",
            "artifact-download",
        ],
    }


@app.post("/v1/jobs", dependencies=[Depends(require_api_token)])
async def create_job(payload: JobCreate) -> dict:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    root = _job_dir(job_id)
    for child in ARTIFACT_CATEGORIES:
        (root / child).mkdir(parents=True, exist_ok=True)

    request = payload.model_dump()
    request["job_id"] = job_id
    request["created_at"] = datetime.now(UTC).isoformat()
    (root / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    status = _write_status(root, job_id=job_id, state="created", stage="waiting_for_input")
    return {"job_id": job_id, "status": status}


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_api_token)])
async def get_job(job_id: str) -> dict:
    root = _require_job(job_id)
    return json.loads((root / "status.json").read_text(encoding="utf-8"))


@app.post("/v1/jobs/{job_id}/references", dependencies=[Depends(require_api_token)])
async def upload_references(
    job_id: str,
    files: Annotated[list[UploadFile], File()],
) -> dict:
    root = _require_job(job_id)
    if not files or len(files) > MAX_REFERENCE_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Upload between 1 and {MAX_REFERENCE_FILES} reference images at once.",
        )

    index = _load_reference_index(root)
    known_hashes = {str(item.get("sha256")) for item in index}
    added: list[dict] = []
    total = 0

    for upload in files:
        data = await upload.read(MAX_REFERENCE_BYTES + 1)
        await upload.close()
        if len(data) > MAX_REFERENCE_BYTES:
            raise HTTPException(status_code=413, detail="A reference image exceeds the 12 MB limit.")
        total += len(data)
        if total > MAX_REFERENCE_TOTAL_BYTES:
            raise HTTPException(status_code=413, detail="Reference upload exceeds the 48 MB batch limit.")

        mime, extension, width, height = _image_info(data)
        digest = hashlib.sha256(data).hexdigest()
        if digest in known_hashes:
            continue

        stored_name = f"ref-{digest[:16]}{extension}"
        destination = root / "references" / stored_name
        destination.write_bytes(data)

        record = {
            "stored_name": stored_name,
            "original_name": Path(upload.filename or "reference").name[:200],
            "sha256": digest,
            "mime": mime,
            "bytes": len(data),
            "width": width,
            "height": height,
            "uploaded_at": datetime.now(UTC).isoformat(),
        }
        index.append(record)
        added.append(record)
        known_hashes.add(digest)

    (root / "references.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    status = _write_status(
        root,
        state="ready",
        stage="references_uploaded",
        reference_count=len(index),
    )
    return {"job_id": job_id, "added": added, "references": index, "status": status}


@app.get("/v1/jobs/{job_id}/artifacts", dependencies=[Depends(require_api_token)])
async def list_artifacts(job_id: str) -> dict:
    root = _require_job(job_id)
    result: dict[str, list[dict]] = {}
    for category in sorted(ARTIFACT_CATEGORIES):
        entries = []
        for path in sorted((root / category).glob("*")):
            if path.is_file():
                entries.append({"name": path.name, "bytes": path.stat().st_size})
        result[category] = entries
    return {"job_id": job_id, "artifacts": result}


@app.get(
    "/v1/jobs/{job_id}/artifacts/{category}/{filename}",
    dependencies=[Depends(require_api_token)],
)
async def download_artifact(job_id: str, category: str, filename: str) -> FileResponse:
    root = _require_job(job_id)
    if category not in ARTIFACT_CATEGORIES or Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    path = (root / category / filename).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path)


@app.post("/v1/jobs/{job_id}/vision/analyze", dependencies=[Depends(require_api_token)])
async def analyze_vision(job_id: str, request: VisionAnalyzeRequest) -> dict:
    root = _require_job(job_id)
    images, labels = _collect_images(root, request)
    if not images:
        raise HTTPException(status_code=400, detail="No reference or render images are available.")

    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    system = (
        "You are the visual QA component of an autonomous Blender 3D modeling system. "
        "Compare all supplied reference/current-render images together. Focus on geometry, silhouette, "
        "proportions, spatial relationships, missing features, and whether procedural modeling, a generated "
        "base mesh, or a hybrid workflow is appropriate. Return only JSON matching the supplied schema. "
        "Do not claim details that cannot be seen."
    )
    prompt = (
        f"Job: {job_request.get('prompt', '')}\n"
        f"Intended use: {job_request.get('intended_use', '')}\n"
        f"Target width mm: {job_request.get('target_width_mm')}\n"
        f"Current stage: {request.stage}\n"
        f"Images in order: {labels}\n"
        f"Extra instruction: {request.instruction or 'None'}\n"
        "Give concrete changes that the Blender planner can act on."
    )

    try:
        ollama_result = await OllamaProxyClient().chat_json(
            model=VISION_MODEL,
            system=system,
            prompt=prompt,
            images=images,
            schema=VisionReport.model_json_schema(),
            temperature=0.1,
        )
        report = VisionReport.model_validate(ollama_result.data)
    except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Vision analysis failed: {exc}") from exc

    payload = {
        "job_id": job_id,
        "model": VISION_MODEL,
        "endpoint": ollama_result.endpoint,
        "stage": request.stage,
        "images": labels,
        "usage": ollama_result.usage,
        "report": report.model_dump(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    log_name = _write_llm_log(root, "vision", payload)
    (root / "vision-latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_status(root, state="ready", stage="vision_analyzed", latest_vision_log=log_name)
    return payload


@app.post("/v1/jobs/{job_id}/plan", dependencies=[Depends(require_api_token)])
async def build_plan(job_id: str, request: PlanRequest) -> dict:
    root = _require_job(job_id)
    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    latest_vision: dict = {}
    vision_path = root / "vision-latest.json"
    if vision_path.exists():
        latest_vision = json.loads(vision_path.read_text(encoding="utf-8")).get("report") or {}

    system = (
        "You are the planning component of an autonomous Blender modeling system. "
        "Create an executable modeling plan that can be implemented with Blender/Python/MCP. "
        "Prefer deterministic procedural operations for functional/geometric objects and use base-mesh or "
        "hybrid approaches only when organic geometry justifies it. Return only JSON matching the schema."
    )
    prompt = (
        f"User request: {job_request.get('prompt', '')}\n"
        f"Intended use: {job_request.get('intended_use', '')}\n"
        f"Target width mm: {job_request.get('target_width_mm')}\n"
        f"Latest vision analysis: {json.dumps(latest_vision, ensure_ascii=False)}\n"
        f"Additional instruction: {request.instruction or 'None'}\n"
        "Break the work into checkpoint-sized stages. Each stage needs objective success criteria that can "
        "be checked from Blender scene data and/or rendered images."
    )

    try:
        ollama_result = await OllamaProxyClient().chat_json(
            model=REASONING_MODEL,
            system=system,
            prompt=prompt,
            schema=ModelingPlan.model_json_schema(),
            temperature=0.1,
        )
        plan = ModelingPlan.model_validate(ollama_result.data)
    except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Modeling plan failed: {exc}") from exc

    payload = {
        "job_id": job_id,
        "model": REASONING_MODEL,
        "endpoint": ollama_result.endpoint,
        "usage": ollama_result.usage,
        "plan": plan.model_dump(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    log_name = _write_llm_log(root, "plan", payload)
    (root / "plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_status(root, state="ready", stage="planned", latest_plan_log=log_name)
    return payload


@app.post("/v1/jobs/{job_id}/tests/pikachu", dependencies=[Depends(require_api_token)])
async def generate_pikachu_test(job_id: str) -> dict:
    """Generate the first recognizable multi-view character through Blender MCP."""
    root = _require_job(job_id)
    blend_path = str(root / "scene" / "pikachu-v1.blend")
    render_dir = str(root / "renders")

    _write_status(root, state="running", stage="pikachu_build")
    request_payload = {
        "tool": "blender_python_exec",
        "arguments": {
            "code": pikachu_script(),
            "args": {"blend_path": blend_path, "output_dir": render_dir},
            "transport": "headless",
            "factory_startup": True,
            "timeout_seconds": 420,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=480) as client:
            response = await client.post(f"{WORKER_URL}/v1/mcp/call", json=request_payload)
            response.raise_for_status()
            result = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        _write_status(root, state="failed", stage="pikachu_build", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Pikachu Blender build failed: {exc}") from exc

    blender_error = _worker_blender_error(result)
    if blender_error:
        _write_status(root, state="failed", stage="pikachu_build", error=blender_error)
        raise HTTPException(status_code=502, detail=f"Blender script failed: {blender_error}")

    expected = [
        "pikachu-front.png",
        "pikachu-front-left.png",
        "pikachu-left.png",
        "pikachu-back-left.png",
        "pikachu-back.png",
        "pikachu-back-right.png",
        "pikachu-right.png",
        "pikachu-front-right.png",
        "pikachu-top.png",
    ]
    missing = [name for name in expected if not (root / "renders" / name).is_file()]
    if missing or not (root / "scene" / "pikachu-v1.blend").is_file():
        _write_status(root, state="failed", stage="pikachu_build", error=f"Missing artifacts: {missing}")
        raise HTTPException(status_code=502, detail=f"Blender completed but expected artifacts are missing: {missing}")

    status = _write_status(
        root,
        state="ready",
        stage="pikachu_rendered",
        pikachu={"blend": "pikachu-v1.blend", "renders": expected},
    )
    return {"job_id": job_id, "status": status, "worker_result": result, "renders": expected}


@app.post("/v1/jobs/{job_id}/smoke-test", dependencies=[Depends(require_api_token)])
async def smoke_test(job_id: str) -> dict:
    root = _require_job(job_id)

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

    blender_error = _worker_blender_error(result)
    if blender_error:
        _write_status(root, state="failed", stage="mcp_smoke_test", error=blender_error)
        raise HTTPException(status_code=502, detail=f"Blender script failed: {blender_error}")

    status = _write_status(
        root,
        state="ready",
        stage="mcp_smoke_test_complete",
        smoke_test={"blend": blend_path, "render": render_path},
    )
    return {"job_id": job_id, "status": status, "worker_result": result}
