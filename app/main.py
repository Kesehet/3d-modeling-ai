from __future__ import annotations

import base64
import hashlib
import json
import shutil
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
    VISION_MODELS,
    WORKER_URL,
)
from .dashboard import dashboard_page, jobs_snapshot, public_artifact, public_render
from .generic_builder import generic_scene_script
from .history import append_history, load_history
from .mesh_builder import adaptive_loft_script
from .ollama import OllamaProxyClient, OllamaProxyError
from .quality import evaluate_scene_spec_structural, get_benchmark
from .repair import print_repair_script
from .research import research_web_references, write_research_manifest
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
    recognizable: bool | None = None
    subject_match_score: float | None = Field(default=None, ge=0.0, le=1.0)
    observations: list[str] = Field(default_factory=list)
    issues: list[VisionIssue] = Field(default_factory=list)
    priority_actions: list[str] = Field(default_factory=list)


class ModelingDirectorDecision(BaseModel):
    action: Literal[
        "accept",
        "build_procedural",
        "revise_procedural",
        "build_mesh",
        "rebuild_mesh",
    ]
    subject_match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = Field(default="", max_length=2400)
    instructions: list[str] = Field(default_factory=list, max_length=16)
    major_problems: list[str] = Field(default_factory=list, max_length=20)


class RefinementComparison(BaseModel):
    candidate_is_better: bool
    summary: str = ""
    improvements: list[str] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)


class QualityBenchmarkRequest(BaseModel):
    key: Literal["pikachu", "desk-lamp", "sneaker", "office-chair", "quadruped-robot"]


class BenchmarkVisualReport(BaseModel):
    pass_benchmark: bool
    recognizable: bool
    summary: str = ""
    required_features_visible: dict[str, bool] = Field(default_factory=dict)
    major_failures: list[str] = Field(default_factory=list)


def _normalize_benchmark_visual_payload(
    data: object,
    required_features: tuple[str, ...],
) -> dict:
    if not isinstance(data, dict):
        raise TypeError("Benchmark visual response is not a JSON object.")

    def as_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {
            "1", "true", "yes", "pass", "passed", "visible", "present", "met", "recognizable",
        }

    def text_value(value: object) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return "; ".join(text_value(item) for item in value if text_value(item))
        if isinstance(value, dict):
            return "; ".join(
                f"{key}: {text_value(item)}"
                for key, item in value.items()
                if text_value(item)
            )
        if value is None:
            return ""
        return str(value)

    def find_value(keys: tuple[str, ...]) -> object | None:
        queue: list[dict] = [data]
        seen: set[int] = set()
        while queue:
            node = queue.pop(0)
            node_id = id(node)
            if node_id in seen:
                continue
            seen.add(node_id)
            for key in keys:
                if key in node:
                    return node[key]
            for key in ("analysis", "evaluation", "assessment", "result", "overall", "visual_quality"):
                child = node.get(key)
                if isinstance(child, dict):
                    queue.append(child)
        return None

    def canonical_text(value: object) -> str:
        return " ".join(
            "".join(
                character if character.isalnum() else " "
                for character in str(value).lower()
            ).split()
        )

    pass_raw = find_value(
        ("pass_benchmark", "benchmark_pass", "quality_gate_pass", "overall_pass", "passed", "pass")
    )
    recognizable_raw = find_value(
        ("recognizable", "recognisable", "is_recognizable", "subject_recognizable", "recognition")
    )
    summary_raw = find_value(
        ("summary", "overall_summary", "overall_assessment", "assessment_summary", "verdict", "critique")
    )
    failures_raw = find_value(
        ("major_failures", "failures", "missing_features", "major_issues", "blocking_issues", "problems")
    )
    raw_features = find_value(
        (
            "required_features_visible",
            "feature_results",
            "feature_checks",
            "required_features",
            "features",
            "checks",
        )
    )

    explicit_pass = pass_raw is not None
    explicit_recognizable = recognizable_raw is not None
    pass_value = as_bool(pass_raw) if explicit_pass else False
    recognizable = as_bool(recognizable_raw) if explicit_recognizable else pass_value

    returned_features: dict[str, bool] = {}
    if isinstance(raw_features, dict):
        for key, value in raw_features.items():
            if isinstance(value, dict):
                visible = (
                    value.get("visible")
                    if "visible" in value
                    else value.get("present", value.get("passed", value.get("met", False)))
                )
            else:
                visible = value
            returned_features[canonical_text(key)] = as_bool(visible)
    elif isinstance(raw_features, list):
        for item in raw_features:
            if isinstance(item, dict):
                feature_name = (
                    item.get("feature")
                    or item.get("name")
                    or item.get("requirement")
                    or item.get("part")
                )
                if not feature_name:
                    continue
                visible = (
                    item.get("visible")
                    if "visible" in item
                    else item.get("present", item.get("passed", item.get("met", False)))
                )
                returned_features[canonical_text(feature_name)] = as_bool(visible)
            elif isinstance(item, str):
                lowered = item.lower()
                if ":" in item:
                    key, value = item.rsplit(":", 1)
                    returned_features[canonical_text(key)] = as_bool(value)
                elif any(word in lowered for word in ("visible", "present", "pass", "met")):
                    returned_features[canonical_text(item)] = True

    feature_map: dict[str, bool] = {}
    for feature in required_features:
        target = canonical_text(feature)
        if target in returned_features:
            feature_map[feature] = returned_features[target]
            continue

        target_tokens = set(target.split())
        best_score = 0.0
        best_value = False
        for candidate, value in returned_features.items():
            candidate_tokens = set(candidate.split())
            if not target_tokens or not candidate_tokens:
                continue
            overlap = len(target_tokens & candidate_tokens)
            score = overlap / max(len(target_tokens), len(candidate_tokens))
            if target in candidate or candidate in target:
                score = max(score, 0.9)
            if score > best_score:
                best_score = score
                best_value = value
        feature_map[feature] = best_value if best_score >= 0.5 else False

    failures_text = text_value(failures_raw)
    failures: list[str] = []
    if isinstance(failures_raw, list):
        failures = [text_value(item) for item in failures_raw if text_value(item)]
    elif failures_text:
        failures = [failures_text]

    # Some vision models give only an overall pass/recognizable verdict and prose instead
    # of repeating the exact feature-key map. Respect an explicit positive verdict unless
    # they also report missing/blocking features. This prevents schema-format mismatch from
    # turning every otherwise valid model into a false regression.
    if not returned_features and explicit_pass and pass_value and recognizable and not failures:
        feature_map = {feature: True for feature in required_features}

    if not explicit_pass:
        pass_value = recognizable and bool(feature_map) and all(feature_map.values()) and not failures
    if not explicit_recognizable and pass_value:
        recognizable = True

    normalized = {
        "pass_benchmark": bool(pass_value),
        "recognizable": bool(recognizable),
        "summary": text_value(summary_raw),
        "required_features_visible": feature_map,
        "major_failures": failures,
    }
    if not all(feature_map.values()):
        normalized["pass_benchmark"] = False
    if not normalized["recognizable"]:
        normalized["pass_benchmark"] = False
    if failures:
        normalized["pass_benchmark"] = False
    return normalized


def _normalize_refinement_comparison_payload(data: object) -> dict:
    if not isinstance(data, dict):
        raise TypeError("Refinement comparison response is not a JSON object.")
    normalized = dict(data)
    if "candidate_is_better" not in normalized:
        for key in ("candidate_better", "is_better", "improved", "accept_candidate", "accepted"):
            if key not in normalized:
                continue
            value = normalized[key]
            if isinstance(value, bool):
                normalized["candidate_is_better"] = value
            else:
                normalized["candidate_is_better"] = str(value).strip().lower() in {
                    "1", "true", "yes", "better", "improved", "accept", "accepted"
                }
            break
    normalized.setdefault("candidate_is_better", False)
    normalized.setdefault("summary", "")
    for key in ("improvements", "regressions"):
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = [value]
        elif not isinstance(value, list):
            normalized[key] = []
    return normalized


def _normalize_vision_report_payload(data: object) -> dict:
    if not isinstance(data, dict):
        raise TypeError("Vision response is not a JSON object.")

    def text_value(value: object) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return "; ".join(text_value(item) for item in value if text_value(item))
        if isinstance(value, dict):
            return "; ".join(
                f"{key}: {text_value(item)}"
                for key, item in value.items()
                if text_value(item)
            )
        if value is None:
            return ""
        return str(value)

    def find_value(keys: tuple[str, ...]) -> object | None:
        queue: list[dict] = [data]
        seen: set[int] = set()
        while queue:
            node = queue.pop(0)
            node_id = id(node)
            if node_id in seen:
                continue
            seen.add(node_id)
            for key in keys:
                if key in node:
                    return node[key]
            for key in ("analysis", "evaluation", "assessment", "result", "overall", "visual_quality"):
                child = node.get(key)
                if isinstance(child, dict):
                    queue.append(child)
        return None

    def as_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {
            "1", "true", "yes", "recognizable", "recognisable", "match", "matched", "pass", "passed",
        }

    recognizable_raw = find_value(
        ("recognizable", "recognisable", "is_recognizable", "subject_recognizable", "subject_match")
    )
    recognizable = as_bool(recognizable_raw) if recognizable_raw is not None else None

    score_raw = find_value(
        ("subject_match_score", "match_score", "recognizability_score", "recognition_score")
    )
    subject_match_score: float | None = None
    if score_raw is not None:
        try:
            subject_match_score = float(score_raw)
            if subject_match_score > 1.0 and subject_match_score <= 100.0:
                subject_match_score /= 100.0
            subject_match_score = max(0.0, min(1.0, subject_match_score))
        except (TypeError, ValueError):
            subject_match_score = None

    observations: list[str] = []
    raw_observations = data.get("observations")
    if isinstance(raw_observations, list):
        observations.extend(text_value(item) for item in raw_observations if text_value(item))

    analysis = data.get("analysis")
    if isinstance(analysis, dict):
        for key, value in analysis.items():
            if key in {"issues", "priority_actions", "recommendations"}:
                continue
            text = text_value(value)
            if text:
                observations.append(f"{str(key).replace('_', ' ')}: {text}")
    elif analysis:
        text = text_value(analysis)
        if text:
            observations.append(text)

    raw_issues = data.get("issues")
    if not isinstance(raw_issues, list) and isinstance(analysis, dict):
        raw_issues = analysis.get("issues")
    if not isinstance(raw_issues, list):
        raw_issues = []

    issues: list[dict] = []
    severity_aliases = {
        "critical": "high",
        "major": "high",
        "high": "high",
        "moderate": "medium",
        "medium": "medium",
        "minor": "low",
        "low": "low",
        "info": "low",
    }
    for raw in raw_issues[:30]:
        if isinstance(raw, str):
            issue_text = raw.strip()
            if issue_text:
                issues.append(
                    {
                        "object": "",
                        "issue": issue_text,
                        "severity": "medium",
                        "suggested_change": issue_text,
                    }
                )
            continue
        if not isinstance(raw, dict):
            continue
        issue_text = text_value(
            raw.get("issue")
            or raw.get("problem")
            or raw.get("description")
            or raw.get("finding")
        )
        if not issue_text:
            continue
        severity_key = str(raw.get("severity") or raw.get("priority") or "medium").lower()
        suggestion = text_value(
            raw.get("suggested_change")
            or raw.get("recommendation")
            or raw.get("fix")
            or raw.get("action")
        ) or issue_text
        issues.append(
            {
                "object": text_value(raw.get("object") or raw.get("part") or raw.get("area")),
                "issue": issue_text,
                "severity": severity_aliases.get(severity_key, "medium"),
                "suggested_change": suggestion,
            }
        )

    # Some multimodal models return prose grouped by visual category instead of an issue array.
    # Preserve that critique as actionable medium-priority issues rather than discarding it.
    if not issues and isinstance(analysis, dict):
        for key in (
            "geometry",
            "silhouette",
            "proportions",
            "spatial_relationships",
            "missing_features",
            "placement",
            "colors",
            "materials",
            "accuracy",
            "problems",
            "recommendations",
        ):
            text = text_value(analysis.get(key))
            if not text:
                continue
            issues.append(
                {
                    "object": str(key).replace("_", " "),
                    "issue": text,
                    "severity": "medium",
                    "suggested_change": text,
                }
            )

    priority_actions: list[str] = []
    raw_actions = data.get("priority_actions") or data.get("recommendations")
    if isinstance(raw_actions, list):
        priority_actions.extend(text_value(item) for item in raw_actions if text_value(item))
    elif raw_actions:
        action_text = text_value(raw_actions)
        if action_text:
            priority_actions.append(action_text)
    if not priority_actions:
        priority_actions = [item["suggested_change"] for item in issues[:6]]

    summary = text_value(
        find_value(("summary", "overall_summary", "overall_assessment", "assessment_summary", "verdict"))
    )
    if not summary:
        summary = " ".join(observations[:3]).strip()[:1800] or "Visual analysis completed."

    strategy_raw = str(
        find_value(("recommended_modeling_strategy", "modeling_strategy", "recommended_strategy"))
        or "procedural"
    ).lower()
    if "hybrid" in strategy_raw:
        strategy = "hybrid"
    elif "base" in strategy_raw or "sculpt" in strategy_raw:
        strategy = "base_mesh"
    else:
        strategy = "procedural"

    return {
        "summary": summary,
        "recommended_modeling_strategy": strategy,
        "recognizable": recognizable,
        "subject_match_score": subject_match_score,
        "observations": observations[:30],
        "issues": issues[:30],
        "priority_actions": priority_actions[:10],
    }


class PlanRequest(BaseModel):
    instruction: str | None = Field(default=None, max_length=4000)


class ResearchRequest(BaseModel):
    query: str | None = Field(default=None, max_length=500)
    max_images: int = Field(default=6, ge=1, le=8)


class PikachuRefineRequest(BaseModel):
    iterations: int = Field(default=2, ge=1, le=3)
    auto_research: bool = True


class PrintRepairRequest(BaseModel):
    voxel_size: float = Field(default=0.05, ge=0.01, le=0.20)
    target_width_mm: float | None = Field(default=None, gt=0, le=10_000)


class PikachuTuning(BaseModel):
    head_width: float = Field(default=1.0, ge=0.72, le=1.32)
    head_height: float = Field(default=1.0, ge=0.72, le=1.32)
    body_width: float = Field(default=1.0, ge=0.72, le=1.32)
    body_height: float = Field(default=1.0, ge=0.72, le=1.32)
    ear_length: float = Field(default=1.0, ge=0.72, le=1.32)
    eye_spacing: float = Field(default=1.0, ge=0.72, le=1.32)
    cheek_scale: float = Field(default=1.0, ge=0.72, le=1.32)
    foot_scale: float = Field(default=1.0, ge=0.72, le=1.32)
    tail_scale: float = Field(default=1.0, ge=0.72, le=1.32)


class GenericGenerateRequest(BaseModel):
    auto_research: bool = True


class GenericRefineRequest(BaseModel):
    iterations: int = Field(default=1, ge=1, le=3)


class SubjectPartSpec(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    count: int = Field(default=1, ge=1, le=12)
    importance: Literal["required", "important", "detail"] = "required"
    shape_hint: str = Field(default="", max_length=160)
    placement_hint: str = Field(default="", max_length=240)


class SubjectInventory(BaseModel):
    subject_family: str = Field(default="unknown", max_length=100)
    silhouette_summary: str = Field(default="", max_length=1200)
    complexity: Literal["simple", "moderate", "complex"] = "moderate"
    recommended_strategy: Literal["procedural", "base_mesh", "hybrid"] = "procedural"
    minimum_distinct_parts: int = Field(default=4, ge=2, le=32)
    major_parts: list[SubjectPartSpec] = Field(min_length=2, max_length=24)


class SceneObjectSpec(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    shape: Literal["sphere", "cube", "cylinder", "cone", "torus", "rod", "beam", "frustum", "wedge"]
    location: list[float] = Field(min_length=3, max_length=3)
    scale: list[float] = Field(min_length=3, max_length=3)
    rotation_deg: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)
    start: list[float] | None = Field(default=None, min_length=3, max_length=3)
    end: list[float] | None = Field(default=None, min_length=3, max_length=3)
    radius: float | None = Field(default=None, gt=0.01, le=5.0)
    color: str = Field(default="#808080", pattern=r"^#[0-9A-Fa-f]{6}$")
    bevel: bool = True
    smooth: bool = True


class GenericSceneSpec(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    rationale: str = Field(default="", max_length=2000)
    presentation_base: bool = True
    objects: list[SceneObjectSpec] = Field(min_length=1, max_length=40)


class LoftSection(BaseModel):
    position: float = Field(ge=-10.0, le=10.0)
    contour: list[tuple[float, float]] = Field(min_length=8, max_length=8)


class AdaptiveLoftSpec(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    rationale: str = Field(default="", max_length=2400)
    axis: Literal["x", "y", "z"] = "y"
    color: str = Field(default="#B8BDC6", pattern=r"^#[0-9A-Fa-f]{6}$")
    subdivision_levels: int = Field(default=1, ge=0, le=2)
    smooth: bool = True
    presentation_base: bool = True
    sections: list[LoftSection] = Field(min_length=4, max_length=12)
    attachments: list[SceneObjectSpec] = Field(default_factory=list, max_length=24)


async def _build_subject_inventory(
    root: Path,
    job_request: dict,
    visual_context: dict,
    research_context: dict,
) -> SubjectInventory | None:
    system = (
        "You create a subject-part inventory for a general 3D modeling agent. Return JSON only matching "
        "the schema. Identify the minimum set of visually distinct major parts needed for an unfamiliar "
        "viewer to recognize the requested subject. This is not a benchmark-specific checklist. Use the "
        "prompt, reference analysis and research evidence together. Group micro-details, but keep separate "
        "silhouette-defining masses, repeated structural parts, openings/windows/screens, wheels/feet/legs, "
        "handles/appendages and other identity-critical features. Set counts for repeated parts when visible "
        "identity depends on them. Recommend base_mesh or hybrid when rearranging simple primitives is unlikely "
        "to reproduce the subject's silhouette accurately."
    )
    prompt = (
        f"User request: {job_request.get('prompt', '')}\n"
        f"Intended use: {job_request.get('intended_use', '')}\n"
        f"Reference analysis: {json.dumps(visual_context, ensure_ascii=False)}\n"
        f"Research context: {json.dumps(research_context, ensure_ascii=False)}\n"
        "Return a conservative major-part inventory. Do not add brand trivia or invisible internal components."
    )
    reference_images, reference_labels = _collect_images(
        root,
        VisionAnalyzeRequest(
            stage="subject_inventory",
            include_references=True,
            include_renders=False,
            max_images=8,
        ),
    )
    prompt += f"\nReference images in order: {reference_labels}\n"
    candidate_models = (*VISION_MODELS, REASONING_MODEL)
    result = None
    inventory = None
    selected_model = None
    errors: list[str] = []
    client = OllamaProxyClient()
    for candidate_model in candidate_models:
        try:
            candidate_result = await client.chat_json(
                model=candidate_model,
                system=system,
                prompt=prompt,
                images=reference_images or None,
                schema=SubjectInventory.model_json_schema(),
                temperature=0.0,
                num_predict=4096,
            )
            candidate_inventory = SubjectInventory.model_validate(candidate_result.data)
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            errors.append(f"{candidate_model}: {exc}")
            continue
        result = candidate_result
        inventory = candidate_inventory
        selected_model = candidate_model
        break

    if result is None or inventory is None or selected_model is None:
        append_history(root, "subject_inventory_failed", error=" | ".join(errors[-4:]))
        return None

    payload = {
        "model": selected_model,
        "reference_images": reference_labels,
        "inventory": inventory.model_dump(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    (root / "subject-inventory.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    append_history(
        root,
        "subject_inventory",
        family=inventory.subject_family,
        complexity=inventory.complexity,
        strategy=inventory.recommended_strategy,
        minimum_distinct_parts=inventory.minimum_distinct_parts,
        major_parts=len(inventory.major_parts),
    )
    return inventory


def _normalize_scene_spec_payload(data: object, fallback_title: str) -> dict:
    if not isinstance(data, dict):
        raise TypeError("SceneSpec response is not a JSON object.")

    normalized = dict(data)
    normalized.setdefault("title", (fallback_title.strip() or "Generated model")[:120])
    normalized.setdefault("rationale", "")
    normalized.setdefault("presentation_base", True)

    allowed_shapes = {"sphere", "cube", "cylinder", "cone", "torus", "rod", "beam", "frustum", "wedge"}
    shape_aliases = {
        "ellipsoid": "sphere",
        "ball": "sphere",
        "box": "cube",
        "rounded_cube": "cube",
        "disc": "cylinder",
        "disk": "cylinder",
        "tube": "cylinder",
        "ring": "torus",
        "strut": "rod",
        "beam": "beam",
        "limb": "rod",
        "bar": "rod",
        "box_beam": "beam",
        "rectangular_beam": "beam",
        "truncated_cone": "frustum",
        "dome_shade": "frustum",
        "shoe_wedge": "wedge",
    }

    raw_objects = normalized.get("objects")
    if not isinstance(raw_objects, list):
        raise TypeError("SceneSpec objects must be a list.")

    objects = []
    for index, raw in enumerate(raw_objects[:40]):
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        shape = item.get("shape", item.get("type", "cube"))
        shape = str(shape).lower().strip()
        shape = shape_aliases.get(shape, shape)
        if shape not in allowed_shapes:
            shape = "cube"

        location = item.get("location", item.get("position", item.get("center", [0, 0, 0])))
        scale = item.get("scale", item.get("size", [1, 1, 1]))
        rotation = item.get("rotation_deg", item.get("rotation", item.get("rotation_degrees", [0, 0, 0])))

        def vec3(value: object, default: list[float]) -> list[float]:
            if isinstance(value, (int, float)):
                return [float(value), float(value), float(value)]
            if isinstance(value, list) and len(value) >= 3:
                return [float(value[0]), float(value[1]), float(value[2])]
            return default

        def color_hex(value: object) -> str:
            named = {
                "white": "#F4F4F2",
                "black": "#111111",
                "gray": "#808080",
                "grey": "#808080",
                "darkgray": "#3B3F46",
                "darkgrey": "#3B3F46",
                "lightgray": "#C9CDD3",
                "lightgrey": "#C9CDD3",
                "orange": "#F97316",
                "safetyorange": "#FF6700",
                "red": "#DC2626",
                "green": "#16A34A",
                "blue": "#2563EB",
                "yellow": "#FACC15",
                "brown": "#7C4A2D",
                "silver": "#A8AFB8",
                "metal": "#737A84",
            }
            if isinstance(value, list) and len(value) >= 3:
                components = [float(value[0]), float(value[1]), float(value[2])]
                if max(components) <= 1.0:
                    components = [component * 255 for component in components]
                rgb = [max(0, min(255, round(component))) for component in components]
                return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
            text = str(value or "").strip()
            if (
                len(text) == 7
                and text.startswith("#")
                and all(character in "0123456789abcdefABCDEF" for character in text[1:])
            ):
                return text.upper()
            key = "".join(character for character in text.lower() if character.isalnum())
            return named.get(key, "#808080")

        object_name = str(item.get("name") or f"{shape}-{index + 1}")[:80]
        normalized_location = vec3(location, [0.0, 0.0, 0.0])
        normalized_scale = vec3(scale, [1.0, 1.0, 1.0])
        normalized_rotation = vec3(rotation, [0.0, 0.0, 0.0])
        has_connector_endpoints = item.get("start") is not None and item.get("end") is not None

        # Rods/beams are endpoint-defined geometry in the Blender executor.
        # If the model omits endpoints, preserve its location/scale but fall back to a
        # generic solid. Do not infer what the object "should" be from its semantic name.
        if shape in {"rod", "beam"} and not has_connector_endpoints:
            shape = "cube" if shape == "beam" else "cylinder"

        objects.append(
            {
                "name": object_name,
                "shape": shape,
                "location": normalized_location,
                "scale": normalized_scale,
                "rotation_deg": normalized_rotation,
                "start": (
                    vec3(item.get("start"), [0.0, 0.0, 0.0])
                    if item.get("start") is not None
                    else None
                ),
                "end": (
                    vec3(item.get("end"), [0.0, 0.0, 1.0])
                    if item.get("end") is not None
                    else None
                ),
                "radius": (
                    max(0.02, min(5.0, float(item.get("radius"))))
                    if item.get("radius") is not None
                    else None
                ),
                "color": color_hex(item.get("color", "#808080")),
                "bevel": bool(item.get("bevel", True)),
                "smooth": bool(item.get("smooth", True)),
            }
        )

    normalized["objects"] = objects
    return normalized


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
    previous_state = current.get("state")
    previous_stage = current.get("stage")
    current.update(values)
    current["updated_at"] = datetime.now(UTC).isoformat()
    status_file.write_text(json.dumps(current, indent=2), encoding="utf-8")
    if current.get("state") != previous_state or current.get("stage") != previous_stage:
        append_history(
            path,
            "status",
            state=current.get("state"),
            stage=current.get("stage"),
            previous_state=previous_state,
            previous_stage=previous_stage,
        )
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


def _encode_vision_image(path: Path, *, max_side: int = 640, quality: int = 70) -> str:
    """Encode a compact vision-only copy without modifying the persisted artifact."""
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=max(45, min(85, quality)),
                optimize=True,
                progressive=True,
            )
            payload = buffer.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Could not prepare vision image {path.name}: {exc}") from exc
    return base64.b64encode(payload).decode("ascii")


def _collect_images(root: Path, request: VisionAnalyzeRequest) -> tuple[list[str], list[str]]:
    def images_in(folder: str) -> list[Path]:
        paths = []
        for path in (root / folder).glob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                paths.append(path)
        return sorted(paths, key=lambda item: item.stat().st_mtime)

    references = images_in("references") if request.include_references else []
    renders = images_in("renders") if request.include_renders else []

    # Generic refinement must critique one coherent model version. Mixing older model-vN
    # renders into the current set can make the vision model "fix" geometry that no longer exists.
    if request.stage.startswith("generic_") and renders:
        versioned: list[tuple[int, Path]] = []
        for path in renders:
            stem = path.stem
            if not stem.startswith("model-v"):
                continue
            remainder = stem[len("model-v"):]
            number_text = remainder.split("-", 1)[0]
            if number_text.isdigit():
                versioned.append((int(number_text), path))
        if versioned:
            accepted_version: int | None = None
            status_path = root / "status.json"
            if status_path.exists():
                try:
                    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
                    generic_model = status_payload.get("generic_model")
                    if isinstance(generic_model, dict) and isinstance(generic_model.get("version"), int):
                        accepted_version = generic_model["version"]
                except (OSError, json.JSONDecodeError):
                    accepted_version = None
            target_version = accepted_version
            if target_version is None or not any(version == target_version for version, _ in versioned):
                target_version = max(version for version, _ in versioned)
            renders = [path for version, path in versioned if version == target_version]

    if references and renders:
        reference_budget = min(len(references), max(2, request.max_images // 3))
        render_budget = max(1, request.max_images - reference_budget)
        image_paths = references[-reference_budget:] + renders[-render_budget:]
    else:
        image_paths = (references or renders)[-request.max_images :]

    encoded = [_encode_vision_image(path) for path in image_paths]
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


@app.post("/dashboard/jobs/{job_id}/research", include_in_schema=False)
async def dashboard_research(job_id: str, request: ResearchRequest) -> dict:
    return await research_job(job_id, request)


@app.post("/dashboard/jobs/{job_id}/generate", include_in_schema=False)
async def dashboard_generate_generic(job_id: str, request: GenericGenerateRequest) -> dict:
    return await generate_generic_scene(job_id, request)


@app.post("/dashboard/jobs/{job_id}/improve", include_in_schema=False)
async def dashboard_improve_generic(job_id: str, request: GenericRefineRequest) -> dict:
    return await refine_generic_scene(job_id, request)


@app.post("/dashboard/jobs/{job_id}/quality-benchmark", include_in_schema=False)
async def dashboard_quality_benchmark(job_id: str, request: QualityBenchmarkRequest) -> dict:
    return await evaluate_quality_benchmark(job_id, request)


@app.delete("/dashboard/jobs/{job_id}", include_in_schema=False)
async def dashboard_delete_job(job_id: str) -> dict:
    return await delete_job(job_id)


@app.post("/dashboard/jobs/{job_id}/refine-pikachu", include_in_schema=False)
async def dashboard_refine_pikachu(job_id: str, request: PikachuRefineRequest) -> dict:
    return await refine_pikachu(job_id, request)


@app.get("/dashboard/jobs/{job_id}/history", include_in_schema=False)
async def dashboard_history(job_id: str) -> dict:
    root = _require_job(job_id)
    return {"job_id": job_id, "history": load_history(root)}


@app.post("/dashboard/jobs/{job_id}/repair-print", include_in_schema=False)
async def dashboard_repair_print(job_id: str, request: PrintRepairRequest) -> dict:
    return await repair_print_model(job_id, request)


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
        "vision_models": list(VISION_MODELS),
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
            "web-reference-research",
            "iteration-history",
            "parametric-pikachu-refinement",
            "glb-obj-stl-export-attempts",
            "mesh-qa-report",
            "experimental-voxel-print-repair",
            "safe-declarative-generic-scene-builder",
            "generic-prompt-to-primitive-blockout",
            "generic-visual-refinement",
            "quality-regression-benchmark-suite",
            "permanent-job-delete",
        ],
    }


async def delete_job(job_id: str) -> dict:
    root = _require_job(job_id)
    status_path = root / "status.json"
    status: dict = {}
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status = {}
    if status.get("state") == "running":
        raise HTTPException(
            status_code=409,
            detail="This job is still running. Wait for it to finish before deleting it.",
        )

    file_count = sum(1 for path in root.rglob("*") if path.is_file())
    byte_count = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    shutil.rmtree(root)
    return {
        "job_id": job_id,
        "deleted": True,
        "files_deleted": file_count,
        "bytes_deleted": byte_count,
    }


@app.delete("/v1/jobs/{job_id}", dependencies=[Depends(require_api_token)])
async def delete_job_api(job_id: str) -> dict:
    return await delete_job(job_id)


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


async def research_job(job_id: str, request: ResearchRequest) -> dict:
    root = _require_job(job_id)
    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    query = (request.query or job_request.get("prompt") or "").strip()
    _write_status(root, state="running", stage="researching_references")
    try:
        payload = await research_web_references(
            query,
            root / "references",
            max_images=request.max_images,
        )
    except (httpx.HTTPError, ValueError) as exc:
        _write_status(root, state="failed", stage="researching_references", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Reference research failed: {exc}") from exc

    index = _load_reference_index(root)
    known_hashes = {str(item.get("sha256")) for item in index}
    added = []
    for record in payload.get("references", []):
        if record.get("sha256") in known_hashes:
            continue
        index.append(record)
        added.append(record)
        known_hashes.add(str(record.get("sha256")))

    (root / "references.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    write_research_manifest(root / "research.json", payload)
    append_history(root, "research", query=query, added=len(added), provider=payload.get("provider"))
    status = _write_status(
        root,
        state="ready",
        stage="references_researched",
        reference_count=len(index),
    )
    return {
        "job_id": job_id,
        "query": query,
        "added": added,
        "pages": payload.get("pages", []),
        "status": status,
    }


@app.post("/v1/jobs/{job_id}/research", dependencies=[Depends(require_api_token)])
async def research_job_api(job_id: str, request: ResearchRequest) -> dict:
    return await research_job(job_id, request)


@app.get("/v1/jobs/{job_id}/history", dependencies=[Depends(require_api_token)])
async def job_history(job_id: str) -> dict:
    root = _require_job(job_id)
    return {"job_id": job_id, "history": load_history(root)}


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
        "base mesh, or a hybrid workflow is appropriate. When current renders are supplied, explicitly set "
        "recognizable=true only if the rendered model clearly reads as the user's requested subject without "
        "needing the filename or prompt to explain what it is; otherwise set recognizable=false. Set "
        "subject_match_score from 0.0 to 1.0 when possible. Return only JSON matching the supplied schema. "
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

    ollama_result = None
    report = None
    selected_model = None
    errors: list[str] = []
    client = OllamaProxyClient()
    for candidate_model in VISION_MODELS:
        try:
            candidate_result = await client.chat_json(
                model=candidate_model,
                system=system,
                prompt=prompt,
                images=images,
                schema=VisionReport.model_json_schema(),
                temperature=0.0,
                num_predict=4096,
            )
            normalized_report = _normalize_vision_report_payload(candidate_result.data)
            candidate_report = VisionReport.model_validate(normalized_report)
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            errors.append(f"{candidate_model}: {exc}")
            continue
        ollama_result = candidate_result
        report = candidate_report
        selected_model = candidate_model
        break

    if ollama_result is None or report is None or selected_model is None:
        detail = " | ".join(errors[-4:])
        raise HTTPException(
            status_code=502,
            detail=f"Vision analysis failed across configured models: {detail}",
        )

    if selected_model != VISION_MODEL:
        append_history(
            root,
            "vision_model_fallback",
            requested=VISION_MODEL,
            selected=selected_model,
            errors=errors,
        )

    payload = {
        "job_id": job_id,
        "model": selected_model,
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


async def _evaluate_benchmark_visual(
    root: Path,
    *,
    version: int,
    key: str,
) -> dict:
    profile = get_benchmark(key)
    views = ("front", "front-left", "left", "back", "right", "front-right")
    render_paths = [root / "renders" / f"model-v{version}-{view}.png" for view in views]
    if not all(path.is_file() for path in render_paths):
        return {
            "pass_benchmark": False,
            "recognizable": False,
            "summary": "Required benchmark renders are missing.",
            "required_features_visible": {
                feature: False for feature in profile.visual_requirements
            },
            "major_failures": ["missing benchmark renders"],
            "model": None,
        }

    reference_paths = sorted(
        [
            path
            for path in (root / "references").glob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ],
        key=lambda path: path.stat().st_mtime,
    )[-2:]
    image_paths = reference_paths + render_paths
    images = [base64.b64encode(path.read_bytes()).decode("ascii") for path in image_paths]
    labels = [f"{path.parent.name}/{path.name}" for path in image_paths]

    feature_keys = list(profile.visual_requirements)
    system = (
        "You are a strict visual quality gate for a 3D-model regression suite. "
        "Judge the rendered model against the exact benchmark prompt and required features. "
        "A model passes only if it is immediately recognizable as the requested subject and every "
        "required feature is visibly represented across the supplied views. Floating major parts, missing "
        "appendages, generic stacked blobs, or incorrect object structure are failures. Return JSON only "
        "matching the supplied schema. In required_features_visible, use the exact required-feature strings "
        "provided by the user as keys."
    )
    prompt = (
        f"Benchmark: {profile.title}\n"
        f"Exact prompt: {profile.prompt}\n"
        f"Required feature keys: {json.dumps(feature_keys, ensure_ascii=False)}\n"
        f"Images in order: {labels}\n"
        "Set pass_benchmark=true only if the subject is recognizable and all required features are visible. "
        "Do not give credit merely because the colors or rough category are correct."
    )

    client = OllamaProxyClient()
    errors: list[str] = []
    for candidate_model in VISION_MODELS:
        try:
            result = await client.chat_json(
                model=candidate_model,
                system=system,
                prompt=prompt,
                images=images,
                schema=BenchmarkVisualReport.model_json_schema(),
                temperature=0.0,
            )
            normalized = _normalize_benchmark_visual_payload(
                result.data,
                profile.visual_requirements,
            )
            report = BenchmarkVisualReport.model_validate(normalized)
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            errors.append(f"{candidate_model}: {exc}")
            continue

        return {
            **report.model_dump(),
            "model": candidate_model,
            "version": version,
            "images": labels,
            "raw_response": result.data,
        }

    return {
        "pass_benchmark": False,
        "recognizable": False,
        "summary": "Visual benchmark evaluation failed across configured vision models.",
        "required_features_visible": {
            feature: False for feature in profile.visual_requirements
        },
        "major_failures": errors[-4:] or ["visual benchmark unavailable"],
        "model": None,
        "version": version,
        "images": labels,
    }


async def evaluate_quality_benchmark(job_id: str, request: QualityBenchmarkRequest) -> dict:
    root = _require_job(job_id)
    profile = get_benchmark(request.key)
    status_path = root / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=409, detail="Job has no generated model yet.")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    generic_model = status.get("generic_model")
    if not isinstance(generic_model, dict) or not isinstance(generic_model.get("version"), int):
        raise HTTPException(status_code=409, detail="Job has no accepted generic model yet.")

    version = int(generic_model["version"])
    spec_path = root / f"scene-spec-v{version}.json"
    if not spec_path.is_file():
        raise HTTPException(status_code=409, detail="Accepted SceneSpec is missing.")
    spec_payload = json.loads(spec_path.read_text(encoding="utf-8"))
    spec = spec_payload.get("spec") or {}

    structural = evaluate_scene_spec_structural(spec, profile)
    visual = await _evaluate_benchmark_visual(root, version=version, key=request.key)
    passed = bool(structural.get("passed")) and bool(visual.get("pass_benchmark"))
    payload = {
        "job_id": job_id,
        "benchmark": request.key,
        "title": profile.title,
        "version": version,
        "passed": passed,
        "structural": structural,
        "visual": visual,
        "created_at": datetime.now(UTC).isoformat(),
    }
    report_path = root / "exports" / f"benchmark-{request.key}-quality.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    append_history(
        root,
        "quality_benchmark",
        benchmark=request.key,
        version=version,
        passed=passed,
        structural_passed=structural.get("passed"),
        visual_passed=visual.get("pass_benchmark"),
    )
    return payload


async def _compare_generic_versions(
    root: Path,
    *,
    baseline_version: int,
    candidate_version: int,
) -> dict:
    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    views = ("front", "front-left", "left", "back", "right", "front-right")
    reference_paths = sorted(
        [
            path
            for path in (root / "references").glob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ],
        key=lambda path: path.stat().st_mtime,
    )[-4:]
    baseline_paths = [
        root / "renders" / f"model-v{baseline_version}-{view}.png"
        for view in views
    ]
    candidate_paths = [
        root / "renders" / f"model-v{candidate_version}-{view}.png"
        for view in views
    ]
    if not all(path.is_file() for path in baseline_paths + candidate_paths):
        return {
            "candidate_is_better": False,
            "summary": "Visual regression comparison could not run because one or more comparison renders are missing.",
            "improvements": [],
            "regressions": ["missing comparison renders"],
            "model": None,
        }

    image_paths = reference_paths + baseline_paths + candidate_paths
    images = [base64.b64encode(path.read_bytes()).decode("ascii") for path in image_paths]
    labels = [f"{path.parent.name}/{path.name}" for path in image_paths]
    system = (
        "You are a strict visual regression gate for an autonomous 3D modeling system. "
        "Compare the BASELINE model against the CANDIDATE model for the user's exact request. "
        "Set candidate_is_better=true only when the candidate preserves all important recognizable parts "
        "and makes a clear net improvement in silhouette, proportions, connectivity or requested features. "
        "Reject the candidate if it loses a major part, turns detailed geometry into generic blobs, creates "
        "floating/disconnected parts, or is merely different without being clearly better. If uncertain, "
        "set candidate_is_better=false. Return only JSON matching the schema."
    )
    prompt = (
        f"User request: {job_request.get('prompt', '')}\n"
        f"Intended use: {job_request.get('intended_use', '')}\n"
        f"Image order/labels: {labels}\n"
        f"Reference images (if any) come first. Next are BASELINE v{baseline_version} views in this order: "
        f"{list(views)}. Last are CANDIDATE v{candidate_version} views in the same order.\n"
        "Judge recognizability and requested-part preservation before cosmetic changes."
    )

    client = OllamaProxyClient()
    errors: list[str] = []
    for candidate_model in VISION_MODELS:
        try:
            result = await client.chat_json(
                model=candidate_model,
                system=system,
                prompt=prompt,
                images=images,
                schema=RefinementComparison.model_json_schema(),
                temperature=0.0,
            )
            normalized = _normalize_refinement_comparison_payload(result.data)
            comparison = RefinementComparison.model_validate(normalized)
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            errors.append(f"{candidate_model}: {exc}")
            continue

        payload = {
            **comparison.model_dump(),
            "model": candidate_model,
            "baseline_version": baseline_version,
            "candidate_version": candidate_version,
            "images": labels,
            "created_at": datetime.now(UTC).isoformat(),
        }
        _write_llm_log(root, "generic-version-compare", payload)
        return payload

    return {
        "candidate_is_better": False,
        "summary": "Visual regression comparison failed across configured vision models; preserving the previous version.",
        "improvements": [],
        "regressions": errors[-4:] or ["visual comparison unavailable"],
        "model": None,
        "baseline_version": baseline_version,
        "candidate_version": candidate_version,
    }


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


def _generic_spatial_guidance(_: str) -> str:
    return (
        "- Coordinate convention: X is left/right, Y is depth, Z is up.\n"
        "- The front camera sits on negative Y and looks toward positive Y.\n"
        "- Visible details should sit on the intended surface instead of being buried inside another part.\n"
        "- Connected parts should touch or overlap when the real object is physically connected.\n"
        "- Use rod/beam only when start and end are explicitly defined; otherwise use a solid primitive.\n"
        "- Use the reference images, not object-name heuristics, to decide proportions, silhouette and part placement."
    )


async def _build_generic_scene_spec(job_id: str, auto_research: bool) -> GenericSceneSpec:
    root = _require_job(job_id)
    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))

    if auto_research and not _load_reference_index(root):
        try:
            await research_job(job_id, ResearchRequest(max_images=5))
        except HTTPException:
            append_history(root, "research_skipped", reason="generic generation could not obtain web references")

    visual_context: dict = {}
    if _load_reference_index(root):
        try:
            visual = await analyze_vision(
                job_id,
                VisionAnalyzeRequest(
                    stage="generic_reference_analysis",
                    include_references=True,
                    include_renders=False,
                    max_images=8,
                    instruction=(
                        "Analyze the references for a general 3D reconstruction. State whether the subject is "
                        "recognizable, describe its overall silhouette and relative dimensions, and explicitly "
                        "name every major visually distinct part needed for recognition, including repeated "
                        "structural parts and openings/windows/screens where relevant. Do not optimize the "
                        "description around the primitive builder; describe the subject faithfully first."
                    ),
                ),
            )
            visual_context = visual.get("report") or {}
        except HTTPException:
            append_history(root, "vision_skipped", reason="reference analysis was unavailable")

    research_context: dict = {}
    research_path = root / "research.json"
    if research_path.exists():
        try:
            research_payload = json.loads(research_path.read_text(encoding="utf-8"))
            research_context = {
                "pages": [
                    {
                        "title": page.get("title"),
                        "extract": (page.get("extract") or "")[:1200],
                    }
                    for page in (research_payload.get("pages") or [])[:4]
                ]
            }
        except (OSError, json.JSONDecodeError):
            research_context = {}

    inventory = await _build_subject_inventory(
        root,
        job_request,
        visual_context,
        research_context,
    )
    inventory_context = inventory.model_dump() if inventory is not None else {}

    system = (
        "You are the modeling agent for Blender. Inspect the user request and supplied reference images, then "
        "design the best safe declarative SceneSpec you can with the available primitive vocabulary. You own the "
        "geometry decisions: silhouette, proportions, primitive choice, number of parts, placement, rotation, "
        "connectivity and colors. Do not rely on hard-coded object-family templates. Return JSON only matching "
        "the supplied schema. Use rod or beam only when both start and end are explicitly provided. Coordinates "
        "should normally stay within -8..8, with Z up and negative Y facing the front camera. If primitives are "
        "a poor fit, still produce the strongest honest blockout you can; the AI director can switch the next "
        "pass to the continuous-mesh strategy."
    )
    prompt = (
        f"User request: {job_request.get('prompt', '')}\n"
        f"Intended use: {job_request.get('intended_use', '')}\n"
        f"Target width mm: {job_request.get('target_width_mm')}\n"
        f"Visual reference analysis: {json.dumps(visual_context, ensure_ascii=False)}\n"
        f"Web research context: {json.dumps(research_context, ensure_ascii=False)}\n"
        f"Required subject inventory: {json.dumps(inventory_context, ensure_ascii=False)}\n"
        f"Spatial/modeling guidance:\n{_generic_spatial_guidance(str(job_request.get('prompt') or ''))}\n"
        "Create the complete SceneSpec. Favor visual recognizability and the reference evidence over a "
        "small object count. You may use as much of the available object budget as the subject genuinely needs."
    )
    reference_images, reference_labels = _collect_images(
        root,
        VisionAnalyzeRequest(
            stage="generic_scene_planning",
            include_references=True,
            include_renders=False,
            max_images=8,
        ),
    )
    prompt += f"\nReference images supplied directly to the planner in this order: {reference_labels}\n"
    planner_models = (*VISION_MODELS, REASONING_MODEL)
    result = None
    spec = None
    selected_planner_model = None
    planning_errors: list[str] = []
    client = OllamaProxyClient()
    for candidate_model in planner_models:
        try:
            candidate_result = await client.chat_json(
                model=candidate_model,
                system=system,
                prompt=prompt,
                images=reference_images or None,
                schema=GenericSceneSpec.model_json_schema(),
                temperature=0.0,
                num_predict=8192,
            )
            normalized = _normalize_scene_spec_payload(
                candidate_result.data,
                str(job_request.get("prompt") or "Generated model"),
            )
            candidate_spec = GenericSceneSpec.model_validate(normalized)
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            planning_errors.append(f"{candidate_model}: {exc}")
            continue
        result = candidate_result
        spec = candidate_spec
        selected_planner_model = candidate_model
        break

    if result is None or spec is None or selected_planner_model is None:
        raise HTTPException(
            status_code=502,
            detail=f"Generic scene planning failed across configured models: {' | '.join(planning_errors[-4:])}",
        )

    coverage = None

    payload = {
        "job_id": job_id,
        "model": selected_planner_model,
        "endpoint": result.endpoint,
        "usage": result.usage,
        "reference_images": reference_labels,
        "inventory": inventory.model_dump() if inventory is not None else None,
        "inventory_coverage": coverage,
        "spec": spec.model_dump(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    (root / "scene-spec.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_llm_log(root, "scene-spec", payload)
    append_history(
        root,
        "scene_spec",
        title=spec.title,
        object_count=len(spec.objects),
        inventory_coverage=coverage.get("required_coverage") if coverage else None,
    )
    return spec


async def _execute_generic_spec(
    job_id: str,
    spec: GenericSceneSpec,
    *,
    version: int,
) -> dict:
    root = _require_job(job_id)
    prefix = f"model-v{version}"
    blend_path = root / "scene" / f"{prefix}.blend"
    qa_path = root / "exports" / f"{prefix}-qa.json"

    _write_status(root, state="running", stage=f"generic_build_v{version}")
    payload = {
        "tool": "blender_python_exec",
        "arguments": {
            "code": generic_scene_script(),
            "args": {
                "spec": spec.model_dump(),
                "blend_path": str(blend_path),
                "output_dir": str(root / "renders"),
                "exports_dir": str(root / "exports"),
                "qa_path": str(qa_path),
                "prefix": prefix,
            },
            "transport": "headless",
            "factory_startup": True,
            "timeout_seconds": 300,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=360) as client:
            response = await client.post(f"{WORKER_URL}/v1/mcp/call", json=payload)
            response.raise_for_status()
            result = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        _write_status(root, state="failed", stage=f"generic_build_v{version}", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Generic Blender build failed: {exc}") from exc

    blender_error = _worker_blender_error(result)
    if blender_error:
        _write_status(root, state="failed", stage=f"generic_build_v{version}", error=blender_error)
        raise HTTPException(status_code=502, detail=f"Generic Blender script failed: {blender_error}")

    expected = [f"{prefix}-{view}.png" for view in (
        "front", "front-left", "left", "back-left", "back",
        "back-right", "right", "front-right", "top"
    )]
    missing = [name for name in expected if not (root / "renders" / name).is_file()]
    if missing or not blend_path.is_file():
        _write_status(root, state="failed", stage=f"generic_build_v{version}", error=f"Missing artifacts: {missing}")
        raise HTTPException(status_code=502, detail=f"Generic build completed but artifacts are missing: {missing}")

    spec_payload = {
        "job_id": job_id,
        "version": version,
        "spec": spec.model_dump(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    (root / f"scene-spec-v{version}.json").write_text(
        json.dumps(spec_payload, indent=2),
        encoding="utf-8",
    )
    append_history(
        root,
        "generic_generation",
        version=version,
        title=spec.title,
        object_count=len(spec.objects),
        renders=expected,
        blend=blend_path.name,
        qa=qa_path.name,
    )
    status = _write_status(
        root,
        state="ready",
        stage=f"generic_rendered_v{version}",
        generic_model={
            "version": version,
            "title": spec.title,
            "blend": blend_path.name,
            "renders": expected,
            "qa": qa_path.name,
        },
    )
    return {
        "job_id": job_id,
        "status": status,
        "spec": spec.model_dump(),
        "renders": expected,
        "worker_result": result,
    }


def _load_subject_inventory(root: Path) -> SubjectInventory | None:
    path = root / "subject-inventory.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("inventory") if isinstance(payload, dict) else None
        return SubjectInventory.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError, TypeError):
        return None


def _normalize_adaptive_loft_payload(data: object, fallback_title: str) -> dict:
    if not isinstance(data, dict):
        raise TypeError("Adaptive loft response is not a JSON object.")

    normalized = dict(data)
    # Multimodal models sometimes wrap the schema even when a JSON schema is supplied.
    for wrapper in (
        "adaptive_loft_spec",
        "adaptive_loft",
        "loft_spec",
        "loft",
        "mesh_spec",
        "mesh",
        "spec",
        "result",
    ):
        candidate = normalized.get(wrapper)
        if isinstance(candidate, dict) and any(
            key in candidate
            for key in ("sections", "cross_sections", "profiles", "slices")
        ):
            normalized = dict(candidate)
            break

    normalized["title"] = str(normalized.get("title") or fallback_title or "Adaptive mesh")[:120]
    normalized["rationale"] = str(
        normalized.get("rationale")
        or normalized.get("reasoning")
        or normalized.get("description")
        or ""
    )[:2400]
    axis = str(normalized.get("axis") or normalized.get("loft_axis") or "y").lower()
    normalized["axis"] = axis if axis in {"x", "y", "z"} else "y"
    color = str(normalized.get("color") or normalized.get("base_color") or "#B8BDC6")
    if not (
        len(color) == 7
        and color.startswith("#")
        and all(character in "0123456789abcdefABCDEF" for character in color[1:])
    ):
        color = "#B8BDC6"
    normalized["color"] = color.upper()
    raw_subdivision = (
        normalized.get("subdivision_levels")
        if normalized.get("subdivision_levels") is not None
        else normalized.get("subdivision", 0)
    )
    try:
        subdivision = int(raw_subdivision or 0)
    except (TypeError, ValueError):
        subdivision = 0
    normalized["subdivision_levels"] = max(0, min(2, subdivision))
    normalized["smooth"] = bool(normalized.get("smooth", True))
    normalized["presentation_base"] = bool(normalized.get("presentation_base", True))

    sections = []
    raw_sections = (
        normalized.get("sections")
        or normalized.get("cross_sections")
        or normalized.get("profiles")
        or normalized.get("slices")
    )
    if isinstance(raw_sections, dict):
        for container_key in ("items", "sections", "cross_sections", "profiles", "slices"):
            candidate = raw_sections.get(container_key)
            if isinstance(candidate, list):
                raw_sections = candidate
                break
        else:
            dict_values = list(raw_sections.values())
            if dict_values and all(isinstance(item, dict) for item in dict_values):
                raw_sections = dict_values
    if not isinstance(raw_sections, list):
        raise TypeError("Adaptive loft sections could not be normalized to a list.")

    for raw in raw_sections[:12]:
        if not isinstance(raw, dict):
            continue
        raw_contour = (
            raw.get("contour")
            or raw.get("points")
            or raw.get("perimeter")
            or raw.get("profile")
        )
        if isinstance(raw_contour, dict):
            raw_contour = (
                raw_contour.get("points")
                or raw_contour.get("contour")
                or raw_contour.get("vertices")
            )
        if not isinstance(raw_contour, list) or len(raw_contour) != 8:
            continue
        contour: list[list[float]] = []
        valid = True
        for point in raw_contour:
            if isinstance(point, dict):
                if "u" in point and "v" in point:
                    pair = (point.get("u"), point.get("v"))
                elif "x" in point and "z" in point:
                    pair = (point.get("x"), point.get("z"))
                elif "x" in point and "y" in point:
                    pair = (point.get("x"), point.get("y"))
                else:
                    valid = False
                    break
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                pair = (point[0], point[1])
            else:
                valid = False
                break
            try:
                u = max(-10.0, min(10.0, float(pair[0])))
                v = max(-10.0, min(10.0, float(pair[1])))
            except (TypeError, ValueError):
                valid = False
                break
            contour.append([u, v])
        if not valid:
            continue
        raw_position = raw.get("position")
        if raw_position is None:
            raw_position = (
                raw.get("axis_position")
                if raw.get("axis_position") is not None
                else raw.get("offset")
            )
        if raw_position is None:
            raw_position = raw.get(normalized["axis"])
        try:
            position = max(-10.0, min(10.0, float(raw_position)))
        except (TypeError, ValueError):
            continue
        sections.append({"position": position, "contour": contour})

    sections.sort(key=lambda section: section["position"])
    deduped = []
    for section in sections:
        if deduped and abs(section["position"] - deduped[-1]["position"]) < 0.05:
            continue
        deduped.append(section)
    if len(deduped) < 4:
        raise ValueError("Adaptive loft needs at least four distinct valid cross sections.")
    normalized["sections"] = deduped

    attachments = (
        normalized.get("attachments")
        or normalized.get("separate_parts")
        or normalized.get("details")
        or []
    )
    if isinstance(attachments, dict):
        attachments = list(attachments.values())
    if not isinstance(attachments, list):
        attachments = []
    normalized_attachments = _normalize_scene_spec_payload(
        {"title": normalized["title"], "objects": attachments[:24]},
        normalized["title"],
    )
    normalized["attachments"] = normalized_attachments["objects"]
    return normalized


async def _build_adaptive_loft_spec(job_id: str, *, reason: str) -> AdaptiveLoftSpec:
    root = _require_job(job_id)
    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    inventory = _load_subject_inventory(root)

    latest_vision: dict = {}
    vision_path = root / "vision-latest.json"
    if vision_path.exists():
        try:
            vision_payload = json.loads(vision_path.read_text(encoding="utf-8"))
            if isinstance(vision_payload, dict):
                latest_vision = vision_payload.get("report") or {}
        except (OSError, json.JSONDecodeError):
            latest_vision = {}

    images, labels = _collect_images(
        root,
        VisionAnalyzeRequest(
            stage="generic_mesh_reference_reconstruction",
            include_references=True,
            include_renders=True,
            max_images=14,
        ),
    )
    system = (
        "You are the mesh-reconstruction stage of an autonomous Blender system. The primitive blockout "
        "was not recognizable enough, so create a SAFE DECLARATIVE LOFT MESH instead of more cubes/wedges. "
        "Return JSON only matching the supplied AdaptiveLoftSpec schema. Choose the axis that best follows "
        "the subject's main length. Each cross section MUST contain exactly 8 perimeter points in consistent "
        "clockwise order when viewed along the positive axis; use the same semantic perimeter order in every "
        "section. Use 5-10 sections to capture major silhouette changes. Coordinates are Blender units and "
        "must remain roughly within -8..8. The loft is the main continuous body mass. Put visually separate "
        "identity-critical parts that should not be fused into the main silhouette into attachments using the "
        "safe primitive schema (for example wheels, handles, lenses, windows, feet, knobs, antennas). "
        "Do not approximate the whole subject with attachments; the loft must carry the primary silhouette. "
        "Front-facing subject direction is negative Y and Z is up. Favor reference geometry and recognizability "
        "over cosmetic detail."
    )
    prompt = (
        f"User request: {job_request.get('prompt', '')}\n"
        f"Intended use: {job_request.get('intended_use', '')}\n"
        f"Target width mm: {job_request.get('target_width_mm')}\n"
        f"Reason for strategy switch: {reason}\n"
        f"Subject inventory: {json.dumps(inventory.model_dump() if inventory else {}, ensure_ascii=False)}\n"
        f"Latest visual critique: {json.dumps(latest_vision, ensure_ascii=False)}\n"
        f"Images in order: {labels}\n"
        "Create a substantially more recognizable continuous base mesh. For an 8-point cross section, a useful "
        "order is around the perimeter from lower-left -> mid-left -> upper-left/shoulder -> top-left -> "
        "top-right -> upper-right/shoulder -> mid-right -> lower-right, adjusted to the actual subject."
    )

    client = OllamaProxyClient()
    errors: list[str] = []
    for candidate_model in VISION_MODELS:
        try:
            result = await client.chat_json(
                model=candidate_model,
                system=system,
                prompt=prompt,
                images=images or None,
                schema=AdaptiveLoftSpec.model_json_schema(),
                temperature=0.0,
                num_predict=8192,
            )
            normalized = _normalize_adaptive_loft_payload(
                result.data,
                str(job_request.get("prompt") or "Adaptive mesh"),
            )
            spec = AdaptiveLoftSpec.model_validate(normalized)
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            errors.append(f"{candidate_model}: {exc}")
            continue

        payload = {
            "job_id": job_id,
            "model": candidate_model,
            "endpoint": result.endpoint,
            "usage": result.usage,
            "images": labels,
            "reason": reason,
            "spec": spec.model_dump(),
            "created_at": datetime.now(UTC).isoformat(),
        }
        _write_llm_log(root, "adaptive-loft-spec", payload)
        append_history(
            root,
            "adaptive_loft_planned",
            model=candidate_model,
            axis=spec.axis,
            sections=len(spec.sections),
            attachments=len(spec.attachments),
            reason=reason,
        )
        return spec

    detail = " | ".join(errors[-4:])
    raise HTTPException(
        status_code=502,
        detail=f"Adaptive mesh planning failed across configured vision models: {detail}",
    )


async def _execute_adaptive_loft(
    job_id: str,
    spec: AdaptiveLoftSpec,
    *,
    version: int,
) -> dict:
    root = _require_job(job_id)
    prefix = f"model-v{version}"
    blend_path = root / "scene" / f"{prefix}.blend"
    qa_path = root / "exports" / f"{prefix}-qa.json"

    _write_status(root, state="running", stage=f"adaptive_mesh_build_v{version}")
    payload = {
        "tool": "blender_python_exec",
        "arguments": {
            "code": adaptive_loft_script(),
            "args": {
                "spec": spec.model_dump(),
                "blend_path": str(blend_path),
                "output_dir": str(root / "renders"),
                "exports_dir": str(root / "exports"),
                "qa_path": str(qa_path),
                "prefix": prefix,
            },
            "transport": "headless",
            "factory_startup": True,
            "timeout_seconds": 300,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=360) as client:
            response = await client.post(f"{WORKER_URL}/v1/mcp/call", json=payload)
            response.raise_for_status()
            result = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        _write_status(root, state="failed", stage=f"adaptive_mesh_build_v{version}", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Adaptive mesh Blender build failed: {exc}") from exc

    blender_error = _worker_blender_error(result)
    if blender_error:
        _write_status(root, state="failed", stage=f"adaptive_mesh_build_v{version}", error=blender_error)
        raise HTTPException(status_code=502, detail=f"Adaptive mesh Blender script failed: {blender_error}")

    views = (
        "front", "front-left", "left", "back-left", "back",
        "back-right", "right", "front-right", "top",
    )
    expected = [f"{prefix}-{view}.png" for view in views]
    missing = [name for name in expected if not (root / "renders" / name).is_file()]
    if missing or not blend_path.is_file():
        _write_status(
            root,
            state="failed",
            stage=f"adaptive_mesh_build_v{version}",
            error=f"Missing artifacts: {missing}",
        )
        raise HTTPException(
            status_code=502,
            detail=f"Adaptive mesh build completed but artifacts are missing: {missing}",
        )

    spec_payload = {
        "job_id": job_id,
        "version": version,
        "strategy": "adaptive_loft",
        "spec": spec.model_dump(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    (root / f"mesh-spec-v{version}.json").write_text(
        json.dumps(spec_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    append_history(
        root,
        "adaptive_mesh_generation",
        version=version,
        title=spec.title,
        sections=len(spec.sections),
        attachments=len(spec.attachments),
        renders=expected,
        blend=blend_path.name,
        qa=qa_path.name,
    )
    status = _write_status(
        root,
        state="ready",
        stage=f"adaptive_mesh_rendered_v{version}",
        modeling_strategy="adaptive_loft",
        generic_model={
            "version": version,
            "title": spec.title,
            "blend": blend_path.name,
            "renders": expected,
            "qa": qa_path.name,
            "strategy": "adaptive_loft",
        },
    )
    return {
        "job_id": job_id,
        "status": status,
        "strategy": "adaptive_loft",
        "spec": spec.model_dump(),
        "renders": expected,
        "worker_result": result,
    }


async def _generate_adaptive_mesh_fallback(job_id: str, *, reason: str) -> dict:
    root = _require_job(job_id)
    status_path = root / "status.json"
    previous_status: dict = {}
    if status_path.exists():
        try:
            previous_status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_status = {}

    previous_model = (
        dict(previous_status.get("generic_model"))
        if isinstance(previous_status.get("generic_model"), dict)
        else None
    )
    baseline_version = (
        previous_model.get("version")
        if isinstance(previous_model, dict) and isinstance(previous_model.get("version"), int)
        else None
    )

    _write_status(root, state="running", stage="adaptive_mesh_planning")
    spec = await _build_adaptive_loft_spec(job_id, reason=reason)
    version = 1 + len(list((root / "scene").glob("model-v*.blend")))
    build = await _execute_adaptive_loft(job_id, spec, version=version)

    comparison: dict | None = None
    if baseline_version is not None and baseline_version != version:
        comparison = await _compare_generic_versions(
            root,
            baseline_version=baseline_version,
            candidate_version=version,
        )

    try:
        quality = await _generic_recognizability_check(
            job_id,
            stage="generic_mesh_fallback_quality",
        )
    except HTTPException as exc:
        append_history(root, "adaptive_mesh_quality_unavailable", error=str(exc.detail))
        quality = {
            "recognizable": None,
            "subject_match_score": None,
            "recommended_strategy": "hybrid",
            "summary": str(exc.detail),
        }

    recognizable = quality.get("recognizable")
    better = bool(comparison and comparison.get("candidate_is_better"))
    accept_candidate = recognizable is True or baseline_version is None or better

    if not accept_candidate and previous_model is not None:
        active_renders = previous_model.get("renders")
        if not isinstance(active_renders, list):
            active_renders = [
                f"model-v{baseline_version}-{view}.png"
                for view in (
                    "front", "front-left", "left", "back-left", "back",
                    "back-right", "right", "front-right", "top",
                )
            ]
        status = _write_status(
            root,
            state="ready",
            stage="generic_needs_strategy_switch",
            modeling_strategy=previous_status.get("modeling_strategy") or "procedural",
            generic_model=previous_model,
            quality_gate={
                "recognizable": False,
                "subject_match_score": quality.get("subject_match_score"),
                "recommended_strategy": quality.get("recommended_strategy"),
                "summary": quality.get("summary"),
                "adaptive_mesh_attempted": True,
                "adaptive_mesh_candidate_version": version,
            },
        )
        append_history(
            root,
            "adaptive_mesh_rejected",
            baseline_version=baseline_version,
            candidate_version=version,
            comparison_summary=comparison.get("summary") if comparison else None,
        )
    else:
        if recognizable is True:
            stage = "adaptive_mesh_recognizable"
        elif recognizable is None:
            stage = "generic_quality_unverified"
        else:
            stage = "adaptive_mesh_needs_refinement"
        status = _write_status(
            root,
            state="ready",
            stage=stage,
            modeling_strategy="adaptive_loft",
            quality_gate={
                "recognizable": recognizable,
                "subject_match_score": quality.get("subject_match_score"),
                "recommended_strategy": quality.get("recommended_strategy"),
                "summary": quality.get("summary"),
                "adaptive_mesh_attempted": True,
                "adaptive_mesh_candidate_version": version,
                "better_than_previous": better if comparison is not None else None,
            },
        )
        append_history(
            root,
            "adaptive_mesh_accepted",
            version=version,
            recognizable=recognizable,
            better_than_previous=better if comparison is not None else None,
        )

    build["comparison"] = comparison
    build["quality_gate"] = quality
    build["status"] = status
    return build


async def _ask_modeling_director(
    job_id: str,
    *,
    stage: str,
    current_strategy: str,
    include_renders: bool = True,
) -> dict:
    root = _require_job(job_id)
    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    inventory = _load_subject_inventory(root)
    images, labels = _collect_images(
        root,
        VisionAnalyzeRequest(
            stage=stage,
            include_references=True,
            include_renders=include_renders,
            max_images=10,
        ),
    )

    system = (
        "You are the lead 3D modeling director. Make the next modeling decision from the actual user request, "
        "reference images and current renders. You are responsible for visual judgment; the Python application "
        "only orchestrates your decision. Choose exactly one action: accept, build_procedural, revise_procedural, "
        "build_mesh, or rebuild_mesh. Before renders exist, choose build_procedural or build_mesh. After renders "
        "exist, choose accept only when the object is clearly recognizable as the requested subject and its main "
        "silhouette/proportions/identity-critical parts are credible. Choose revise_procedural when the existing "
        "declarative primitive strategy can plausibly be corrected. Choose rebuild_mesh when the primary silhouette "
        "needs a continuous surface or the current strategy is fundamentally wrong. Ignore filenames and object "
        "names as proof of correctness: judge the visible geometry. Return JSON only matching the schema."
    )
    prompt = (
        f"Exact user request: {job_request.get('prompt', '')}\n"
        f"Intended use: {job_request.get('intended_use', '')}\n"
        f"Current strategy: {current_strategy}\n"
        f"Current stage: {stage}\n"
        f"AI-generated subject inventory: "
        f"{json.dumps(inventory.model_dump() if inventory else {}, ensure_ascii=False)}\n"
        f"Images in order: {labels}\n"
        "Give concrete instructions for the next modeling pass. Spend the available reasoning budget on visual "
        "comparison and geometry decisions rather than generic commentary."
    )

    client = OllamaProxyClient()
    errors: list[str] = []
    for candidate_model in (*VISION_MODELS, REASONING_MODEL):
        try:
            result = await client.chat_json(
                model=candidate_model,
                system=system,
                prompt=prompt,
                images=images or None,
                schema=ModelingDirectorDecision.model_json_schema(),
                temperature=0.0,
                num_predict=4096,
            )
            decision = ModelingDirectorDecision.model_validate(result.data)
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            errors.append(f"{candidate_model}: {exc}")
            continue

        action = decision.action
        if not include_renders and action == "accept":
            action = "build_procedural"
        payload = {
            **decision.model_dump(),
            "action": action,
            "model": candidate_model,
            "endpoint": result.endpoint,
            "usage": result.usage,
            "stage": stage,
            "current_strategy": current_strategy,
            "images": labels,
            "created_at": datetime.now(UTC).isoformat(),
        }
        _write_llm_log(root, "modeling-director", payload)
        append_history(
            root,
            "modeling_director",
            stage=stage,
            model=candidate_model,
            action=action,
            subject_match_score=decision.subject_match_score,
            summary=decision.summary,
        )
        return payload

    raise HTTPException(
        status_code=502,
        detail="Modeling director failed across configured models: " + " | ".join(errors[-4:]),
    )


async def _generic_recognizability_check(job_id: str, *, stage: str) -> dict:
    root = _require_job(job_id)
    current_strategy = "procedural"
    status_path = root / "status.json"
    if status_path.exists():
        try:
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            current_strategy = str(status_payload.get("modeling_strategy") or "procedural")
        except (OSError, json.JSONDecodeError):
            pass

    decision = await _ask_modeling_director(
        job_id,
        stage=stage,
        current_strategy=current_strategy,
        include_renders=True,
    )
    action = decision["action"]
    recognizable = action == "accept"
    recommended_strategy = (
        "base_mesh"
        if action in {"build_mesh", "rebuild_mesh"}
        else "procedural"
    )
    return {
        "recognizable": recognizable,
        "subject_match_score": decision.get("subject_match_score"),
        "recommended_strategy": recommended_strategy,
        "summary": decision.get("summary"),
        "major_missing_parts": decision.get("major_problems") or [],
        "director_action": action,
        "instructions": decision.get("instructions") or [],
        "director": decision,
        "stage": stage,
    }


async def generate_generic_scene(job_id: str, request: GenericGenerateRequest) -> dict:
    root = _require_job(job_id)

    if request.auto_research and not _load_reference_index(root):
        try:
            await research_job(job_id, ResearchRequest(max_images=5))
        except HTTPException:
            append_history(root, "research_skipped", reason="automatic research did not return usable references")

    initial_decision = await _ask_modeling_director(
        job_id,
        stage="initial_modeling_strategy",
        current_strategy="none",
        include_renders=False,
    )
    initial_action = initial_decision["action"]
    if initial_action in {"build_mesh", "rebuild_mesh"}:
        _write_status(root, state="running", stage="agent_selected_mesh", modeling_strategy="adaptive_loft")
        return await _generate_adaptive_mesh_fallback(
            job_id,
            reason=(
                initial_decision.get("summary")
                or "The AI modeling director selected a continuous mesh strategy."
            )
            + "\n"
            + "\n".join(initial_decision.get("instructions") or []),
        )

    _write_status(root, state="running", stage="agent_selected_procedural", modeling_strategy="procedural")
    spec = await _build_generic_scene_spec(job_id, auto_research=False)
    version = 1 + len(list((root / "scene").glob("model-v*.blend")))
    build = await _execute_generic_spec(job_id, spec, version=version)

    try:
        quality_gate = await _generic_recognizability_check(
            job_id,
            stage="generic_initial_quality",
        )
    except HTTPException as exc:
        append_history(root, "generic_initial_quality_unavailable", error=str(exc.detail))
        status = _write_status(
            root,
            state="ready",
            stage="generic_quality_unverified",
            modeling_strategy="procedural",
            quality_gate={"recognizable": None, "error": str(exc.detail)},
        )
        build["status"] = status
        build["quality_gate"] = None
        return build

    if quality_gate.get("recognizable") is True:
        status = _write_status(
            root,
            state="ready",
            stage="generic_initial_recognizable",
            modeling_strategy="procedural",
            quality_gate=quality_gate,
        )
        build["status"] = status
        build["quality_gate"] = quality_gate
        return build

    if quality_gate.get("director_action") in {"build_mesh", "rebuild_mesh"}:
        append_history(
            root,
            "automatic_strategy_switch",
            from_strategy="procedural",
            to_strategy="adaptive_loft",
            reason=quality_gate.get("summary"),
        )
        return await _generate_adaptive_mesh_fallback(
            job_id,
            reason=(quality_gate.get("summary") or "")
            + "\n"
            + "\n".join(quality_gate.get("instructions") or []),
        )

    _write_status(
        root,
        state="ready",
        stage="generic_needs_refinement",
        modeling_strategy="procedural",
        quality_gate=quality_gate,
    )
    return await refine_generic_scene(job_id, GenericRefineRequest(iterations=2))


@app.post("/v1/jobs/{job_id}/generate", dependencies=[Depends(require_api_token)])
async def generate_generic_scene_api(job_id: str, request: GenericGenerateRequest) -> dict:
    return await generate_generic_scene(job_id, request)


async def refine_generic_scene(job_id: str, request: GenericRefineRequest) -> dict:
    root = _require_job(job_id)
    status_path = root / "status.json"
    status_payload: dict = {}
    if status_path.exists():
        try:
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status_payload = {}

    current_strategy = str(status_payload.get("modeling_strategy") or "procedural")
    if current_strategy == "adaptive_loft" or status_payload.get("stage") in {
        "generic_needs_strategy_switch",
        "adaptive_mesh_needs_refinement",
    }:
        decision = await _ask_modeling_director(
            job_id,
            stage="adaptive_mesh_continuation",
            current_strategy="adaptive_loft",
            include_renders=True,
        )
        if decision["action"] == "accept":
            quality_gate = {
                "recognizable": True,
                "subject_match_score": decision.get("subject_match_score"),
                "recommended_strategy": "base_mesh",
                "summary": decision.get("summary"),
                "director_action": "accept",
                "instructions": decision.get("instructions") or [],
            }
            status = _write_status(
                root,
                state="ready",
                stage="adaptive_mesh_recognizable",
                modeling_strategy="adaptive_loft",
                quality_gate=quality_gate,
            )
            return {"job_id": job_id, "iterations": [], "rejected": None, "quality_gate": quality_gate, "status": status}
        return await _generate_adaptive_mesh_fallback(
            job_id,
            reason=(decision.get("summary") or "")
            + "\n"
            + "\n".join(decision.get("instructions") or []),
        )

    spec_files = sorted(root.glob("scene-spec-v*.json"), key=lambda path: path.stat().st_mtime)
    if not spec_files:
        return await generate_generic_scene(job_id, GenericGenerateRequest(auto_research=True))

    active_version = None
    generic_model = status_payload.get("generic_model")
    if isinstance(generic_model, dict) and isinstance(generic_model.get("version"), int):
        active_version = generic_model["version"]
    current_spec_path = (
        root / f"scene-spec-v{active_version}.json"
        if active_version is not None and (root / f"scene-spec-v{active_version}.json").is_file()
        else spec_files[-1]
    )
    current_payload = json.loads(current_spec_path.read_text(encoding="utf-8"))
    current_spec = GenericSceneSpec.model_validate(current_payload["spec"])
    current_version = int(current_payload.get("version") or active_version or 1)
    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    inventory = _load_subject_inventory(root)
    completed: list[dict] = []

    for offset in range(request.iterations):
        decision = await _ask_modeling_director(
            job_id,
            stage=f"agent_refinement_{offset + 1}",
            current_strategy="procedural",
            include_renders=True,
        )
        if decision["action"] == "accept":
            break
        if decision["action"] in {"build_mesh", "rebuild_mesh"}:
            append_history(
                root,
                "agent_strategy_switch",
                from_strategy="procedural",
                to_strategy="adaptive_loft",
                reason=decision.get("summary"),
            )
            return await _generate_adaptive_mesh_fallback(
                job_id,
                reason=(decision.get("summary") or "")
                + "\n"
                + "\n".join(decision.get("instructions") or []),
            )

        images, labels = _collect_images(
            root,
            VisionAnalyzeRequest(
                stage=f"agent_scene_revision_{offset + 1}",
                include_references=True,
                include_renders=True,
                max_images=10,
            ),
        )
        system = (
            "You are the autonomous 3D modeling agent revising a declarative SceneSpec. The AI modeling director "
            "has already inspected the references and current renders. Follow its instructions and return the full "
            "replacement SceneSpec. You may preserve, move, resize, rotate, replace, add or delete geometry as needed; "
            "the current spec is not sacred. Do not use object-family templates. Base decisions on the supplied images "
            "and director critique. Use rod/beam only with explicit start and end. Return JSON only matching the schema."
        )
        prompt = (
            f"Exact user request: {job_request.get('prompt', '')}\n"
            f"Current SceneSpec: {json.dumps(current_spec.model_dump(), ensure_ascii=False)}\n"
            f"AI director decision: {json.dumps(decision, ensure_ascii=False)}\n"
            f"AI-generated subject inventory: "
            f"{json.dumps(inventory.model_dump() if inventory else {}, ensure_ascii=False)}\n"
            f"Universal coordinate guidance:\n{_generic_spatial_guidance('')}\n"
            f"Images in order: {labels}\n"
            "Return the complete next SceneSpec. Spend the available token budget on concrete geometry."
        )

        revised = None
        selected_model = None
        errors: list[str] = []
        client = OllamaProxyClient()
        for candidate_model in (*VISION_MODELS, REASONING_MODEL):
            try:
                result = await client.chat_json(
                    model=candidate_model,
                    system=system,
                    prompt=prompt,
                    images=images or None,
                    schema=GenericSceneSpec.model_json_schema(),
                    temperature=0.0,
                    num_predict=8192,
                )
                normalized = _normalize_scene_spec_payload(result.data, current_spec.title)
                revised = GenericSceneSpec.model_validate(normalized)
            except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
                errors.append(f"{candidate_model}: {exc}")
                continue
            selected_model = candidate_model
            break

        if revised is None or selected_model is None:
            raise HTTPException(
                status_code=502,
                detail="AI SceneSpec revision failed across configured models: " + " | ".join(errors[-4:]),
            )
        if revised.model_dump() == current_spec.model_dump():
            append_history(root, "agent_refinement_stop", reason="AI returned an unchanged SceneSpec")
            break

        version = 1 + len(list((root / "scene").glob("model-v*.blend")))
        build = await _execute_generic_spec(job_id, revised, version=version)
        append_history(
            root,
            "agent_revision",
            version=version,
            model=selected_model,
            object_count=len(revised.objects),
            director_summary=decision.get("summary"),
        )
        completed.append(
            {
                "director": decision,
                "spec": revised.model_dump(),
                "build": build,
                "accepted": True,
            }
        )
        current_spec = revised
        current_version = version

    try:
        final_quality = await _generic_recognizability_check(
            job_id,
            stage="generic_final_quality",
        )
    except HTTPException as exc:
        append_history(root, "generic_final_quality_unavailable", error=str(exc.detail))
        status = _write_status(
            root,
            state="ready",
            stage="generic_quality_unverified",
            modeling_strategy="procedural",
            quality_gate={"recognizable": None, "error": str(exc.detail)},
        )
        return {
            "job_id": job_id,
            "iterations": completed,
            "rejected": None,
            "quality_gate": None,
            "status": status,
        }

    if final_quality.get("recognizable") is True:
        final_stage = "generic_refinement_complete"
    elif final_quality.get("director_action") in {"build_mesh", "rebuild_mesh"}:
        return await _generate_adaptive_mesh_fallback(
            job_id,
            reason=(final_quality.get("summary") or "")
            + "\n"
            + "\n".join(final_quality.get("instructions") or []),
        )
    else:
        final_stage = "generic_needs_refinement"

    status = _write_status(
        root,
        state="ready",
        stage=final_stage,
        modeling_strategy="procedural",
        generic_model={
            "version": current_version,
            "title": current_spec.title,
            "blend": f"model-v{current_version}.blend",
            "renders": [
                f"model-v{current_version}-{view}.png"
                for view in (
                    "front", "front-left", "left", "back-left", "back",
                    "back-right", "right", "front-right", "top",
                )
            ],
            "qa": f"model-v{current_version}-qa.json",
        },
        quality_gate=final_quality,
    )
    return {
        "job_id": job_id,
        "iterations": completed,
        "rejected": None,
        "quality_gate": final_quality,
        "status": status,
    }


@app.post("/v1/jobs/{job_id}/improve", dependencies=[Depends(require_api_token)])
async def refine_generic_scene_api(job_id: str, request: GenericRefineRequest) -> dict:
    return await refine_generic_scene(job_id, request)


@app.post("/v1/jobs/{job_id}/quality-benchmark", dependencies=[Depends(require_api_token)])
async def quality_benchmark_api(job_id: str, request: QualityBenchmarkRequest) -> dict:
    return await evaluate_quality_benchmark(job_id, request)


async def _execute_pikachu(
    job_id: str,
    *,
    tuning: dict[str, float] | None = None,
    prefix: str = "pikachu",
    version: int = 1,
) -> dict:
    root = _require_job(job_id)
    blend_name = f"{prefix}.blend"
    blend_path = str(root / "scene" / blend_name)
    render_dir = str(root / "renders")
    exports_dir = str(root / "exports")
    qa_name = f"{prefix}-qa.json"
    qa_path = str(root / "exports" / qa_name)

    _write_status(root, state="running", stage=f"pikachu_build_v{version}")
    request_payload = {
        "tool": "blender_python_exec",
        "arguments": {
            "code": pikachu_script(),
            "args": {
                "blend_path": blend_path,
                "output_dir": render_dir,
                "exports_dir": exports_dir,
                "qa_path": qa_path,
                "prefix": prefix,
                "params": tuning or {},
            },
            "transport": "headless",
            "factory_startup": True,
            "timeout_seconds": 300,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=360) as client:
            response = await client.post(f"{WORKER_URL}/v1/mcp/call", json=request_payload)
            response.raise_for_status()
            result = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        _write_status(root, state="failed", stage=f"pikachu_build_v{version}", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Pikachu Blender build failed: {exc}") from exc

    blender_error = _worker_blender_error(result)
    if blender_error:
        _write_status(root, state="failed", stage=f"pikachu_build_v{version}", error=blender_error)
        raise HTTPException(status_code=502, detail=f"Blender script failed: {blender_error}")

    views = [
        "front",
        "front-left",
        "left",
        "back-left",
        "back",
        "back-right",
        "right",
        "front-right",
        "top",
    ]
    expected = [f"{prefix}-{view}.png" for view in views]
    missing = [name for name in expected if not (root / "renders" / name).is_file()]
    if missing or not (root / "scene" / blend_name).is_file():
        _write_status(root, state="failed", stage=f"pikachu_build_v{version}", error=f"Missing artifacts: {missing}")
        raise HTTPException(status_code=502, detail=f"Blender completed but expected artifacts are missing: {missing}")

    append_history(
        root,
        "pikachu_iteration",
        version=version,
        prefix=prefix,
        tuning=tuning or {},
        renders=expected,
        blend=blend_name,
        qa=qa_name,
    )
    status = _write_status(
        root,
        state="ready",
        stage=f"pikachu_rendered_v{version}",
        pikachu={"version": version, "blend": blend_name, "renders": expected, "qa": qa_name},
    )
    return {
        "job_id": job_id,
        "version": version,
        "status": status,
        "worker_result": result,
        "renders": expected,
    }


@app.post("/v1/jobs/{job_id}/tests/pikachu", dependencies=[Depends(require_api_token)])
async def generate_pikachu_test(job_id: str) -> dict:
    """Generate the deterministic first-pass benchmark."""
    return await _execute_pikachu(job_id, prefix="pikachu", version=1)


@app.post("/v1/jobs/{job_id}/refine/pikachu", dependencies=[Depends(require_api_token)])
async def refine_pikachu(job_id: str, request: PikachuRefineRequest) -> dict:
    root = _require_job(job_id)
    if not any((root / "renders").glob("pikachu-*.png")):
        await _execute_pikachu(job_id, prefix="pikachu", version=1)

    if request.auto_research and not _load_reference_index(root):
        try:
            await research_job(job_id, ResearchRequest(max_images=5))
        except HTTPException:
            append_history(root, "research_skipped", reason="automatic research did not return usable references")

    completed = []
    current_params: dict[str, float] = {}
    for offset in range(request.iterations):
        vision = await analyze_vision(
            job_id,
            VisionAnalyzeRequest(
                stage=f"pikachu_refinement_{offset + 1}",
                include_references=True,
                include_renders=True,
                max_images=16,
                instruction=(
                    "Judge the newest Pikachu renders against the references. Prioritize silhouette, "
                    "head/body ratio, ear length, eye spacing, cheek size, feet, and lightning-tail scale."
                ),
            ),
        )
        report = vision["report"]
        issues = report.get("issues", [])
        high = sum(1 for issue in issues if issue.get("severity") == "high")
        medium = sum(1 for issue in issues if issue.get("severity") == "medium")
        if offset > 0 and high == 0 and medium <= 1:
            append_history(root, "refinement_stop", reason="visual severity threshold met", high=high, medium=medium)
            break

        system = (
            "You translate visual QA into bounded numeric parameters for a deterministic Pikachu Blender "
            "builder. Values multiply the baseline geometry. 1.0 means no change. Prefer small changes "
            "(usually 0.90-1.10) and never compensate for uncertainty with extreme values. Return JSON only."
        )
        prompt = (
            f"Current parameters: {json.dumps(current_params or PikachuTuning().model_dump())}\n"
            f"Vision report: {json.dumps(report, ensure_ascii=False)}\n"
            "Choose the next parameter set. Only use evidence from the report."
        )
        try:
            tuned_result = await OllamaProxyClient().chat_json(
                model=REASONING_MODEL,
                system=system,
                prompt=prompt,
                schema=PikachuTuning.model_json_schema(),
                temperature=0.1,
            )
            tuning = PikachuTuning.model_validate(tuned_result.data)
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError) as exc:
            append_history(root, "refinement_tuning_failed", error=str(exc))
            raise HTTPException(status_code=502, detail=f"Refinement tuning failed: {exc}") from exc

        current_params = tuning.model_dump()
        version = len(list((root / "logs").glob("tuning-v*.json"))) + 2
        tuning_payload = {
            "version": version,
            "model": REASONING_MODEL,
            "vision_log": vision.get("created_at"),
            "params": current_params,
            "created_at": datetime.now(UTC).isoformat(),
        }
        (root / "logs" / f"tuning-v{version}.json").write_text(
            json.dumps(tuning_payload, indent=2),
            encoding="utf-8",
        )
        build = await _execute_pikachu(
            job_id,
            tuning=current_params,
            prefix=f"pikachu-v{version}",
            version=version,
        )
        completed.append({"vision": vision, "tuning": current_params, "build": build})

    status = _write_status(root, state="ready", stage="pikachu_refinement_complete")
    return {"job_id": job_id, "iterations": completed, "status": status}


async def repair_print_model(job_id: str, request: PrintRepairRequest) -> dict:
    root = _require_job(job_id)
    candidates = [
        path
        for path in (root / "scene").glob("*.blend")
        if "print-repaired" not in path.stem
    ]
    if not candidates:
        raise HTTPException(status_code=400, detail="This job has no source .blend file to repair.")
    source = max(candidates, key=lambda path: path.stat().st_mtime)
    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    target_width_mm = request.target_width_mm or job_request.get("target_width_mm")

    stem = source.stem
    output_blend = root / "scene" / f"{stem}-print-repaired.blend"
    output_stl = root / "exports" / f"{stem}-print-repaired.stl"
    qa_path = root / "exports" / f"{stem}-print-repaired-qa.json"

    _write_status(root, state="running", stage="print_repair")
    payload = {
        "tool": "blender_python_exec",
        "arguments": {
            "code": print_repair_script(),
            "args": {
                "source_blend": str(source),
                "output_blend": str(output_blend),
                "output_stl": str(output_stl),
                "qa_path": str(qa_path),
                "voxel_size": request.voxel_size,
                "target_width_mm": target_width_mm,
            },
            "transport": "headless",
            "factory_startup": True,
            "timeout_seconds": 300,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=360) as client:
            response = await client.post(f"{WORKER_URL}/v1/mcp/call", json=payload)
            response.raise_for_status()
            result = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        _write_status(root, state="failed", stage="print_repair", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Print repair failed: {exc}") from exc

    blender_error = _worker_blender_error(result)
    if blender_error:
        _write_status(root, state="failed", stage="print_repair", error=blender_error)
        raise HTTPException(status_code=502, detail=f"Blender print repair failed: {blender_error}")

    missing = [
        path.name
        for path in (output_blend, output_stl, qa_path)
        if not path.is_file()
    ]
    if missing:
        _write_status(root, state="failed", stage="print_repair", error=f"Missing repair artifacts: {missing}")
        raise HTTPException(status_code=502, detail=f"Repair completed but files are missing: {missing}")

    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    append_history(
        root,
        "print_repair",
        source=source.name,
        blend=output_blend.name,
        stl=output_stl.name,
        qa=qa_path.name,
        print_ready=qa.get("print_ready", False),
        non_manifold_edges=qa.get("non_manifold_edges"),
        connected_components=qa.get("connected_components"),
    )
    status = _write_status(
        root,
        state="ready",
        stage="print_repair_complete",
        print_repair={
            "blend": output_blend.name,
            "stl": output_stl.name,
            "qa": qa_path.name,
            "print_ready": qa.get("print_ready", False),
        },
    )
    return {"job_id": job_id, "status": status, "qa": qa, "worker_result": result}


@app.post("/v1/jobs/{job_id}/repair-print", dependencies=[Depends(require_api_token)])
async def repair_print_model_api(job_id: str, request: PrintRepairRequest) -> dict:
    return await repair_print_model(job_id, request)


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
