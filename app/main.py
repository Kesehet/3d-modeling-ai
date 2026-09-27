from __future__ import annotations

import base64
import hashlib
import json
import math
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


def _semantic_name_tokens(value: str) -> set[str]:
    ignored = {
        "left", "right", "front", "rear", "back", "top", "bottom",
        "upper", "lower", "inner", "outer", "main", "small", "large",
        "primary", "secondary", "part", "object", "segment", "side",
    }
    normalized = "".join(character if character.isalnum() else " " for character in value.lower())
    tokens: set[str] = set()
    for raw in normalized.split():
        if len(raw) < 3 or raw.isdigit() or raw in ignored:
            continue
        token = raw
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("es"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        if token and token not in ignored:
            tokens.add(token)
    return tokens


def _scene_inventory_coverage(spec: GenericSceneSpec, inventory: SubjectInventory) -> dict:
    object_tokens = [
        (obj.name, _semantic_name_tokens(obj.name))
        for obj in spec.objects
    ]
    part_rows = []
    required_total = 0
    required_covered = 0
    for part in inventory.major_parts:
        tokens = _semantic_name_tokens(part.name)
        matches = [
            name for name, candidate_tokens in object_tokens
            if tokens and (tokens & candidate_tokens)
        ]
        needed = max(1, part.count)
        covered_count = min(len(matches), needed)
        covered = covered_count >= needed
        if part.importance == "required":
            required_total += needed
            required_covered += covered_count
        part_rows.append(
            {
                "name": part.name,
                "importance": part.importance,
                "required_count": needed,
                "matched_count": len(matches),
                "matched_objects": matches[:12],
                "covered": covered,
            }
        )

    ratio = 1.0 if required_total == 0 else required_covered / required_total
    minimum_parts_ok = len(spec.objects) >= inventory.minimum_distinct_parts
    missing_required = [
        f"{row['name']} ({row['matched_count']}/{row['required_count']})"
        for row in part_rows
        if row["importance"] == "required" and not row["covered"]
    ]
    return {
        "required_coverage": round(ratio, 4),
        "minimum_distinct_parts": inventory.minimum_distinct_parts,
        "object_count": len(spec.objects),
        "minimum_parts_ok": minimum_parts_ok,
        "missing_required": missing_required,
        "parts": part_rows,
        "passes": bool(ratio >= 0.80 and minimum_parts_ok),
    }


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
    try:
        result = await OllamaProxyClient().chat_json(
            model=REASONING_MODEL,
            system=system,
            prompt=prompt,
            schema=SubjectInventory.model_json_schema(),
            temperature=0.0,
        )
        inventory = SubjectInventory.model_validate(result.data)
    except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
        append_history(root, "subject_inventory_failed", error=str(exc))
        return None

    payload = {
        "model": REASONING_MODEL,
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
        "beam": "rod",
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

        objects.append(
            {
                "name": str(item.get("name") or f"{shape}-{index + 1}")[:80],
                "shape": shape,
                "location": vec3(location, [0.0, 0.0, 0.0]),
                "scale": vec3(scale, [1.0, 1.0, 1.0]),
                "rotation_deg": vec3(rotation, [0.0, 0.0, 0.0]),
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


def _enforce_character_visibility(data: dict, prompt: str) -> dict:
    """Apply conservative character-layout rules after LLM normalization.

    The generic renderer has a fixed front camera on negative Y. LLMs frequently place
    face details on +Y or inside the head, which creates a semantically complete but
    visually blank/blob-like character. This pass only activates for character-like
    prompts and only adjusts strongly semantic parts.
    """
    text = prompt.lower()
    if not any(term in text for term in ("pikachu", "character", "creature", "animal", "figurine")):
        return data

    objects = data.get("objects")
    if not isinstance(objects, list):
        return data

    def clean_name(item: dict) -> str:
        return str(item.get("name") or "").lower().replace("_", " ")

    def find_part(words: tuple[str, ...]) -> dict | None:
        for item in objects:
            if not isinstance(item, dict):
                continue
            name = clean_name(item)
            if any(word in name for word in words):
                return item
        return None

    def vec(item: dict, key: str, default: list[float]) -> list[float]:
        value = item.get(key)
        if isinstance(value, list) and len(value) >= 3:
            return [float(value[0]), float(value[1]), float(value[2])]
        return list(default)

    head = find_part(("head", "face"))
    body = find_part(("body", "torso"))
    if head is None or body is None:
        return data

    head_loc = vec(head, "location", [0.0, 0.0, 2.5])
    head_scale = [max(0.15, abs(v)) for v in vec(head, "scale", [1.2, 1.0, 1.1])]
    body_loc = vec(body, "location", [0.0, 0.0, 1.0])
    body_scale = [max(0.15, abs(v)) for v in vec(body, "scale", [1.0, 0.8, 1.2])]
    front_face_y = head_loc[1] - head_scale[1] * 0.94

    for item in objects:
        if not isinstance(item, dict):
            continue
        name = clean_name(item)

        if "eye" in name or "cheek" in name:
            item["shape"] = "sphere"
            scale = vec(item, "scale", [0.16, 0.10, 0.18])
            scale[0] = min(max(scale[0], head_scale[0] * 0.10), head_scale[0] * 0.24)
            scale[1] = min(max(scale[1], 0.04), head_scale[1] * 0.12)
            scale[2] = min(max(scale[2], head_scale[2] * 0.10), head_scale[2] * 0.24)
            item["scale"] = scale
            location = vec(item, "location", head_loc)
            location[1] = min(location[1], front_face_y - scale[1] * 0.15)
            item["location"] = location
            item["start"] = None
            item["end"] = None
            item["radius"] = None

        is_main_ear = "ear" in name and "tip" not in name
        if is_main_ear:
            item["shape"] = "cone"
            scale = vec(item, "scale", [head_scale[0] * 0.24, head_scale[1] * 0.22, head_scale[2] * 0.8])
            scale[0] = min(max(scale[0], head_scale[0] * 0.18), head_scale[0] * 0.34)
            scale[1] = min(max(scale[1], head_scale[1] * 0.16), head_scale[1] * 0.30)
            scale[2] = max(scale[2], head_scale[2] * 0.72)
            item["scale"] = scale
            location = vec(item, "location", head_loc)
            if "left" in name:
                location[0] = min(location[0], head_loc[0] - head_scale[0] * 0.48)
            elif "right" in name:
                location[0] = max(location[0], head_loc[0] + head_scale[0] * 0.48)
            location[1] = head_loc[1]
            location[2] = max(location[2], head_loc[2] + head_scale[2] * 1.12)
            item["location"] = location
            item["start"] = None
            item["end"] = None
            item["radius"] = None

        if "ear" in name and "tip" in name:
            item["shape"] = "cone"
            item["color"] = "#111111"
            location = vec(item, "location", head_loc)
            side = -1.0 if "left" in name else 1.0
            location[0] = head_loc[0] + side * head_scale[0] * 0.55
            location[1] = head_loc[1]
            location[2] = max(location[2], head_loc[2] + head_scale[2] * 1.70)
            item["location"] = location
            item["scale"] = [
                head_scale[0] * 0.16,
                head_scale[1] * 0.14,
                head_scale[2] * 0.28,
            ]
            item["start"] = None
            item["end"] = None
            item["radius"] = None

        if "arm" in name and "tool" not in name:
            item["shape"] = "sphere"
            side = -1.0 if "left" in name else 1.0
            item["location"] = [
                body_loc[0] + side * body_scale[0] * 0.92,
                body_loc[1] - body_scale[1] * 0.58,
                body_loc[2] + body_scale[2] * 0.05,
            ]
            item["scale"] = [
                body_scale[0] * 0.24,
                body_scale[1] * 0.24,
                body_scale[2] * 0.42,
            ]
            item["start"] = None
            item["end"] = None
            item["radius"] = None

        if "foot" in name or "feet" in name:
            item["shape"] = "sphere"
            side = -1.0 if "left" in name else 1.0
            item["location"] = [
                body_loc[0] + side * body_scale[0] * 0.48,
                body_loc[1] - body_scale[1] * 0.38,
                body_loc[2] - body_scale[2] * 0.86,
            ]
            item["scale"] = [
                body_scale[0] * 0.34,
                body_scale[1] * 0.44,
                body_scale[2] * 0.18,
            ]
            item["start"] = None
            item["end"] = None
            item["radius"] = None

    tail_parts = [
        item for item in objects
        if isinstance(item, dict) and "tail" in clean_name(item)
    ]
    if len(tail_parts) >= 2:
        y = body_loc[1] + body_scale[1] * 0.10
        x0 = body_loc[0] + body_scale[0] * 0.70
        z0 = body_loc[2] + body_scale[2] * 0.02
        points = [
            [x0, y, z0],
            [body_loc[0] + body_scale[0] * 1.38, y, body_loc[2] + body_scale[2] * 0.30],
            [body_loc[0] + body_scale[0] * 1.02, y, body_loc[2] + body_scale[2] * 0.62],
            [body_loc[0] + body_scale[0] * 1.62, y, body_loc[2] + body_scale[2] * 0.92],
            [body_loc[0] + body_scale[0] * 1.28, y, body_loc[2] + body_scale[2] * 1.30],
        ]
        usable_segments = min(len(tail_parts), len(points) - 1)
        for index, item in enumerate(tail_parts[:usable_segments]):
            item["shape"] = "beam"
            item["start"] = points[index]
            item["end"] = points[index + 1]
            item["radius"] = max(0.11, body_scale[0] * 0.18)
            item["location"] = points[index]
            item["scale"] = [1.0, 1.0, 1.0]
            item["rotation_deg"] = [0.0, 0.0, 0.0]
            item["color"] = "#FACC15"

    return data


def _enforce_subject_geometry(data: dict, prompt: str) -> dict:
    """Arrange semantic parts into coherent assemblies for common object families."""
    objects = data.get("objects")
    if not isinstance(objects, list):
        return data
    text = prompt.lower()

    def clean(item: dict) -> str:
        return str(item.get("name") or "").lower().replace("_", " ").replace("-", " ")

    def matching(*terms: str) -> list[dict]:
        return [
            item for item in objects
            if isinstance(item, dict) and any(term in clean(item) for term in terms)
        ]

    def first(*terms: str) -> dict | None:
        found = matching(*terms)
        return found[0] if found else None

    def reset(item: dict, shape: str, location: list[float], scale: list[float], color: str | None = None) -> None:
        item["shape"] = shape
        item["location"] = location
        item["scale"] = scale
        item["rotation_deg"] = [0.0, 0.0, 0.0]
        item["start"] = None
        item["end"] = None
        item["radius"] = None
        if color:
            item["color"] = color

    def place(
        item: dict | None,
        shape: str,
        location: list[float],
        scale: list[float],
        color: str | None = None,
    ) -> None:
        if item is not None:
            reset(item, shape, location, scale, color)

    def link(
        item: dict | None,
        start: list[float],
        end: list[float],
        radius: float,
        color: str | None = None,
        *,
        beam: bool = False,
    ) -> None:
        if item is None:
            return
        item["shape"] = "beam" if beam else "rod"
        item["start"] = start
        item["end"] = end
        item["radius"] = radius
        item["location"] = start
        item["scale"] = [1.0, 1.0, 1.0]
        item["rotation_deg"] = [0.0, 0.0, 0.0]
        if color:
            item["color"] = color

    if "lamp" in text:
        base = first("base")
        stem = first("stem", "upright", "post")
        joint = first("joint", "pivot", "hinge")
        neck = first("neck", "boom", "angled arm")
        shade = first("shade", "dome", "hood", "lamp head", "lamphead")

        place(base, "cylinder", [0.0, 0.0, 0.28], [1.25, 1.05, 0.28], "#3B3F46")
        stem_top = [0.0, 0.0, 2.65]
        shade_top = [1.35, 0.0, 3.85]
        link(stem, [0.0, 0.0, 0.46], stem_top, 0.14, "#737A84")
        place(joint, "sphere", stem_top, [0.28, 0.28, 0.28], "#3B3F46")
        link(neck, stem_top, shade_top, 0.13, "#737A84")
        place(shade, "frustum", [1.35, 0.0, 3.33], [0.82, 0.82, 0.52], "#F97316")

    if "sneaker" in text or "shoe" in text:
        soles = matching("outsole", "midsole", "sole")
        while len(soles) < 2 and len(objects) < 40:
            item = {
                "name": "midsole layer" if soles else "outsole layer",
                "shape": "cube", "location": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0], "rotation_deg": [0.0, 0.0, 0.0],
                "start": None, "end": None, "radius": None,
                "color": "#F4F4F2", "bevel": True, "smooth": True,
            }
            objects.append(item)
            soles.append(item)
        sole_levels = (
            ([0.0, 0.0, 0.22], [2.42, 0.80, 0.18], "#3B3F46"),
            ([0.0, 0.0, 0.50], [2.32, 0.76, 0.13], "#F4F4F2"),
            ([0.0, 0.0, 0.68], [2.20, 0.72, 0.10], "#C9CDD3"),
        )
        for index, item in enumerate(soles[:3]):
            location, scale, color = sole_levels[index]
            place(item, "cube", location, scale, color)

        place(first("upper", "shoe body"), "wedge", [-0.12, 0.0, 1.08], [1.86, 0.70, 0.62], "#2563EB")
        place(first("toe"), "sphere", [1.62, -0.02, 0.94], [0.80, 0.70, 0.44], "#2563EB")
        place(first("heel"), "cube", [-1.62, 0.0, 1.20], [0.44, 0.68, 0.74], "#2563EB")

        tongue = first("tongue")
        place(tongue, "cube", [-0.35, -0.72, 1.42], [0.58, 0.10, 0.55], "#3B3F46")
        if tongue is not None:
            tongue["rotation_deg"] = [0.0, -12.0, 0.0]

        opening = first("opening", "collar")
        if opening is None and len(objects) < 40:
            opening = {
                "name": "foot opening collar", "shape": "torus",
                "location": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0],
                "rotation_deg": [0.0, 0.0, 0.0], "start": None, "end": None,
                "radius": None, "color": "#111111", "bevel": True, "smooth": True,
            }
            objects.append(opening)
        place(opening, "torus", [-1.02, -0.66, 1.48], [0.62, 0.18, 0.46], "#111111")
        if opening is not None:
            opening["rotation_deg"] = [90.0, 0.0, 0.0]

        lace_parts = matching("lace", "cord", "string")
        for index, item in enumerate(lace_parts[:6]):
            z = 1.15 + index * 0.14
            x_span = max(0.28, 0.72 - index * 0.06)
            link(item, [-x_span, -0.80, z], [x_span, -0.80, z], 0.045, "#F4F4F2")

    if "chair" in text:
        place(first("seat"), "cube", [0.0, 0.0, 2.65], [1.45, 1.30, 0.25], "#3B3F46")
        place(first("backrest", "back rest", "chair back"), "cube", [0.0, 1.08, 4.15], [1.35, 0.22, 1.45], "#3B3F46")
        for index, item in enumerate(matching("armrest", "arm rest", "arm support")[:2]):
            side = -1.0 if index == 0 else 1.0
            place(item, "cube", [side * 1.55, -0.05, 3.45], [0.16, 1.02, 0.16], "#737A84")
        place(first("column", "gas lift", "lift", "central post"), "cylinder", [0.0, 0.0, 1.60], [0.22, 0.22, 0.90], "#737A84")

        spokes = matching("spoke")
        while len(spokes) < 5 and len(objects) < 40:
            item = {
                "name": f"base spoke {len(spokes) + 1}", "shape": "rod",
                "location": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0],
                "rotation_deg": [0.0, 0.0, 0.0], "start": None, "end": None,
                "radius": 0.10, "color": "#3B3F46", "bevel": True, "smooth": True,
            }
            objects.append(item)
            spokes.append(item)

        wheels = matching("wheel", "caster")
        while len(wheels) < 5 and len(objects) < 40:
            item = {
                "name": f"caster wheel {len(wheels) + 1}", "shape": "torus",
                "location": [0.0, 0.0, 0.0], "scale": [0.30, 0.14, 0.30],
                "rotation_deg": [90.0, 0.0, 0.0], "start": None, "end": None,
                "radius": None, "color": "#111111", "bevel": True, "smooth": True,
            }
            objects.append(item)
            wheels.append(item)

        for index in range(5):
            angle = math.radians(-90.0 + index * 72.0)
            endpoint = [1.72 * math.cos(angle), 1.72 * math.sin(angle), 0.62]
            link(spokes[index], [0.0, 0.0, 0.88], endpoint, 0.11, "#3B3F46")
            place(wheels[index], "torus", endpoint, [0.30, 0.14, 0.30], "#111111")
            wheels[index]["rotation_deg"] = [90.0, 0.0, math.degrees(angle)]

    if "quadruped" in text or ("robot" in text and "leg" in text):
        place(first("torso", "chassis", "body"), "cube", [0.0, 0.0, 3.10], [2.15, 1.25, 0.72], "#F4F4F2")
        layouts = {
            ("front", "left"): (-1.75, -0.90),
            ("front", "right"): (1.75, -0.90),
            ("rear", "left"): (-1.75, 0.90),
            ("rear", "right"): (1.75, 0.90),
        }
        for (front_rear, side), (x, y) in layouts.items():
            relevant = [
                item for item in objects
                if isinstance(item, dict)
                and front_rear in clean(item)
                and side in clean(item)
                and any(token in clean(item) for token in ("leg", "joint", "knee", "foot"))
            ]
            upper = next((item for item in relevant if "upper" in clean(item)), None)
            joint = next((item for item in relevant if "joint" in clean(item) or "knee" in clean(item)), None)
            lower = next((item for item in relevant if "lower" in clean(item)), None)
            foot = next((item for item in relevant if "foot" in clean(item)), None)
            hip = [x, y, 3.05]
            knee = [x * 1.16, y * 1.36, 1.78]
            ankle = [x * 1.24, y * 1.48, 0.62]
            link(upper, hip, knee, 0.20, "#737A84")
            place(joint, "sphere", knee, [0.30, 0.30, 0.30], "#F97316")
            link(lower, knee, ankle, 0.18, "#737A84")
            place(foot, "cube", [ankle[0], ankle[1] - 0.10, 0.36], [0.42, 0.58, 0.20], "#3B3F46")

        camera_head = first("camera head", "head")
        place(camera_head, "cube", [0.0, -1.68, 3.92], [0.82, 0.42, 0.52], "#3B3F46")
        camera_lens = first("camera lens", "lens", "optic")
        place(camera_lens, "cylinder", [-0.20, -2.13, 3.95], [0.28, 0.28, 0.16], "#111111")
        if camera_lens is not None:
            camera_lens["rotation_deg"] = [90.0, 0.0, 0.0]
        for index, item in enumerate(matching("sensor pod", "sensor")[:2]):
            side = -1.0 if index == 0 else 1.0
            place(item, "cube", [side * 2.40, -0.10, 3.28], [0.34, 0.54, 0.42], "#F97316")
        link(first("antenna mast", "antenna"), [0.65, 0.0, 3.82], [0.65, 0.0, 5.15], 0.08, "#3B3F46")
        place(first("antenna tip"), "sphere", [0.65, 0.0, 5.18], [0.16, 0.16, 0.16], "#F97316")
        place(first("battery", "rear pack"), "cube", [0.0, 1.56, 3.15], [1.02, 0.34, 0.52], "#3B3F46")

        tool_start = [2.05, -0.65, 3.35]
        tool_mid = [2.90, -1.00, 2.72]
        tool_tip = [3.58, -1.20, 2.12]
        link(first("tool arm upper"), tool_start, tool_mid, 0.16, "#F97316")
        place(first("tool arm joint"), "sphere", tool_mid, [0.24, 0.24, 0.24], "#3B3F46")
        link(first("tool arm lower"), tool_mid, tool_tip, 0.14, "#F97316")
        place(first("tool end", "effector", "gripper"), "cube", [3.72, -1.22, 2.05], [0.28, 0.42, 0.18], "#111111")

    return data


def _scene_spec_semantic_tokens(spec: GenericSceneSpec) -> set[str]:
    ignored = {
        "left", "right", "front", "rear", "back", "top", "bottom",
        "upper", "lower", "inner", "outer", "main", "small", "large",
        "primary", "secondary", "part", "object", "segment", "side",
    }
    tokens: set[str] = set()
    for obj in spec.objects:
        normalized = "".join(character if character.isalnum() else " " for character in obj.name.lower())
        for token in normalized.split():
            if len(token) >= 3 and not token.isdigit() and token not in ignored:
                tokens.add(token)
    return tokens


def _scene_spec_regression_reasons(
    current: GenericSceneSpec,
    revised: GenericSceneSpec,
) -> list[str]:
    """Conservative pre-render guard against destructive LLM SceneSpec rewrites."""
    reasons: list[str] = []
    current_count = len(current.objects)
    revised_count = len(revised.objects)

    if current_count >= 5:
        minimum_count = max(3, (current_count * 4 + 4) // 5)  # ceil(80%)
        if revised_count < minimum_count:
            reasons.append(
                f"object count collapsed from {current_count} to {revised_count}; "
                f"minimum safe count is {minimum_count}"
            )

    current_tokens = _scene_spec_semantic_tokens(current)
    revised_tokens = _scene_spec_semantic_tokens(revised)
    if len(current_tokens) >= 4:
        minimum_tokens = max(3, (len(current_tokens) * 2 + 2) // 3)  # ceil(2/3)
        retained = len(current_tokens & revised_tokens)
        if retained < minimum_tokens:
            lost = sorted(current_tokens - revised_tokens)
            reasons.append(
                "semantic part coverage regressed: "
                f"retained {retained}/{len(current_tokens)} tokens; "
                f"lost {', '.join(lost[:12])}"
            )

    current_rods = sum(1 for obj in current.objects if obj.shape == "rod")
    revised_rods = sum(1 for obj in revised.objects if obj.shape == "rod")
    if current_rods >= 2 and revised_rods < max(1, current_rods // 2):
        reasons.append(
            f"connector/limb rods collapsed from {current_rods} to {revised_rods}"
        )

    return reasons


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
                temperature=0.1,
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
    )[-2:]
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


def _generic_spatial_guidance(prompt: str) -> str:
    text = prompt.lower()
    hints = [
        (
            "Coordinate convention is mandatory: X is left/right, Y is depth, Z is up. "
            "The FRONT camera sits on negative Y and looks toward positive Y, so front-facing details "
            "(eyes, cheeks, buttons, screens, grille details) must protrude on the negative-Y surface. "
            "The back of the subject is positive Y."
        ),
        (
            "Do not bury small details inside a larger primitive. Visible secondary parts must sit just outside "
            "the parent surface with a small overlap so they read clearly while remaining connected."
        ),
        (
            "Use rods only for genuinely thin rigid connectors, limbs, stems, handles, spokes or struts. "
            "Do not represent broad ears, heads, shoes, shades or other silhouette masses as antenna-like rods."
        ),
        "For paired parts, place both explicitly and symmetrically unless the request asks for asymmetry.",
    ]
    if any(term in text for term in ("pikachu", "character", "creature", "animal", "figurine")):
        hints.append(
            "Character rule: keep head and torso as distinct masses; pointed ears/horns should normally be "
            "elongated cones or tapered masses above the head; eyes/cheeks belong on negative Y and must "
            "protrude beyond the face; arms and feet must extend beyond the torso silhouette; a tail must "
            "emerge from the rear/side of the torso and remain visible in side or rear views."
        )
    if "lamp" in text:
        hints.append(
            "Lamp rule: build an unbroken contact chain base -> vertical stem -> pivot/joint -> angled neck "
            "-> shade. Use endpoint-aligned rods for stem/neck and overlap each endpoint with the adjoining "
            "part. The shade must be centered on and attached to the end of the neck, never floating."
        )
    if "sneaker" in text or "shoe" in text:
        hints.append(
            "Shoe rule: establish a long low sole first, then a distinct upper/toe/heel volume above it. "
            "The tongue must rise from the opening, and several thin lace rods should cross over the tongue. "
            "Keep the shoe low and elongated rather than stacking round primitives vertically."
        )
    if "chair" in text:
        hints.append(
            "Chair rule: seat and backrest are separate broad surfaces; the backrest sits behind the seat "
            "toward positive Y; armrests sit on both X sides; the central column extends downward from the "
            "seat; five radial spokes and caster wheels must be visibly separated around the base."
        )
    if "quadruped" in text or ("robot" in text and "leg" in text):
        hints.append(
            "Quadruped rule: each of four legs needs an upper segment, visible joint, lower segment and foot. "
            "Attach legs to four distinct torso corners and keep the camera/sensors/antenna/battery/tool arm "
            "outside the torso surface so they remain visible across multiple views."
        )
    return "\n".join(f"- {hint}" for hint in hints)


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
        "You are a 3D blockout planner. Return a safe declarative scene made only from the allowed "
        "primitive types in the supplied JSON schema. Prefer beam for rectangular articulated segments, "
        "frustum for lampshades/truncated cones, and wedge for sloped shoe/body masses. Use the exact keys "
        "title, objects, name, shape, "
        "location, scale, rotation_deg, start, end, radius, color, bevel, and smooth. Use shape='rod' "
        "with start/end/radius for limbs, handles, stems, necks, struts, antennas, and connectors because "
        "it aligns itself between two points. Do not output Python. Build a recognizable model with enough "
        "separate primitives to represent EVERY requested major part and silhouette-defining feature. Never "
        "collapse distinct requested parts into a generic blob merely to reduce primitive count. Major parts "
        "that are physically connected must touch or overlap their parent geometry; do not leave floating "
        "stems, necks, limbs, handles, shades, ears, tails, or connectors. Coordinates should normally stay "
        "within -8..8. Place the subject around the origin and keep its lowest major geometry near Z=0. "
        "Use meaningful semantic object names and realistic relative proportions. Treat the provided subject "
        "inventory as an acceptance contract: every required major part and required repeated count must be "
        "represented by clearly named objects unless the safe primitive vocabulary truly cannot represent it. "
        "Treat the provided spatial guidance as hard geometry constraints, especially the negative-Y front-face "
        "convention. Before returning JSON, verify that every required inventory part exists and will be visibly "
        "exposed from at least one standard QA camera view."
    )
    prompt = (
        f"User request: {job_request.get('prompt', '')}\n"
        f"Intended use: {job_request.get('intended_use', '')}\n"
        f"Target width mm: {job_request.get('target_width_mm')}\n"
        f"Visual reference analysis: {json.dumps(visual_context, ensure_ascii=False)}\n"
        f"Web research context: {json.dumps(research_context, ensure_ascii=False)}\n"
        f"Required subject inventory: {json.dumps(inventory_context, ensure_ascii=False)}\n"
        f"Spatial/modeling guidance:\n{_generic_spatial_guidance(str(job_request.get('prompt') or ''))}\n"
        "Create a primitive-based blockout scene specification. If the subject is organic, approximate "
        "it with overlapping ellipsoids/cones while keeping distinct head/body/limb/appendage/face forms "
        "when the prompt calls for them. If hard-surface, use cubes/cylinders/torus/rods as needed and keep "
        "the mechanical connection chain explicit (for example base -> stem/neck -> joint -> shade). "
        "Favor recognizability, requested-part coverage and physical connectivity over primitive count."
    )
    try:
        result = await OllamaProxyClient().chat_json(
            model=REASONING_MODEL,
            system=system,
            prompt=prompt,
            schema=GenericSceneSpec.model_json_schema(),
            temperature=0.1,
        )
        normalized = _normalize_scene_spec_payload(
            result.data,
            str(job_request.get("prompt") or "Generated model"),
        )
        normalized = _enforce_character_visibility(
            normalized,
            str(job_request.get("prompt") or ""),
        )
        normalized = _enforce_subject_geometry(
            normalized,
            str(job_request.get("prompt") or ""),
        )
        spec = GenericSceneSpec.model_validate(normalized)
    except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=502, detail=f"Generic scene planning failed: {exc}") from exc

    coverage: dict | None = None
    if inventory is not None:
        coverage = _scene_inventory_coverage(spec, inventory)
        if not coverage["passes"]:
            repair_system = (
                "Repair a safe declarative SceneSpec so it covers the supplied general subject inventory. "
                "Return the full SceneSpec JSON only. Preserve existing useful geometry, but add or split "
                "objects needed for missing REQUIRED parts and repeated counts. Use semantic object names "
                "that identify each part. Do not invent subject-specific shortcuts outside the allowed schema. "
                "Favor silhouette and recognition over cosmetic detail."
            )
            repair_prompt = (
                f"User request: {job_request.get('prompt', '')}\n"
                f"Required inventory: {json.dumps(inventory.model_dump(), ensure_ascii=False)}\n"
                f"Current SceneSpec: {json.dumps(spec.model_dump(), ensure_ascii=False)}\n"
                f"Coverage failure: {json.dumps(coverage, ensure_ascii=False)}\n"
                f"Spatial/modeling guidance:\n{_generic_spatial_guidance(str(job_request.get('prompt') or ''))}\n"
                "Return a revised full SceneSpec that covers the missing required parts and counts."
            )
            try:
                repair_result = await OllamaProxyClient().chat_json(
                    model=REASONING_MODEL,
                    system=repair_system,
                    prompt=repair_prompt,
                    schema=GenericSceneSpec.model_json_schema(),
                    temperature=0.0,
                )
                repaired_payload = _normalize_scene_spec_payload(
                    repair_result.data,
                    str(job_request.get("prompt") or "Generated model"),
                )
                repaired_payload = _enforce_character_visibility(
                    repaired_payload,
                    str(job_request.get("prompt") or ""),
                )
                repaired_payload = _enforce_subject_geometry(
                    repaired_payload,
                    str(job_request.get("prompt") or ""),
                )
                repaired = GenericSceneSpec.model_validate(repaired_payload)
                repaired_coverage = _scene_inventory_coverage(repaired, inventory)
                if (
                    repaired_coverage["passes"]
                    or repaired_coverage["required_coverage"] > coverage["required_coverage"]
                ):
                    spec = repaired
                    coverage = repaired_coverage
                    append_history(
                        root,
                        "scene_inventory_repair",
                        required_coverage=coverage["required_coverage"],
                        object_count=len(spec.objects),
                        missing_required=coverage["missing_required"],
                    )
            except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
                append_history(root, "scene_inventory_repair_failed", error=str(exc))

        (root / "scene-inventory-coverage.json").write_text(
            json.dumps(coverage, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if not coverage["passes"]:
            append_history(
                root,
                "scene_inventory_incomplete",
                strategy=inventory.recommended_strategy,
                required_coverage=coverage["required_coverage"],
                missing_required=coverage["missing_required"],
            )

    payload = {
        "job_id": job_id,
        "model": REASONING_MODEL,
        "endpoint": result.endpoint,
        "usage": result.usage,
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


async def _generic_recognizability_check(job_id: str, *, stage: str) -> dict:
    root = _require_job(job_id)
    vision = await analyze_vision(
        job_id,
        VisionAnalyzeRequest(
            stage=stage,
            include_references=True,
            include_renders=True,
            max_images=12,
            instruction=(
                "This is a strict generic-model quality gate. Compare the ACTIVE model renders with the exact "
                "user request and any reference images. Set recognizable=true only if an unfamiliar viewer "
                "would identify the requested subject from the geometry alone. Missing the subject's main "
                "silhouette, primary body masses, required repeated structural parts, or identity-defining "
                "features means recognizable=false even if a few colors or primitive parts are plausible. "
                "If the primitive SceneSpec approach is fundamentally inadequate, recommend base_mesh or hybrid."
            ),
        ),
    )
    report = vision.get("report") or {}
    recognizable = report.get("recognizable")
    append_history(
        root,
        "generic_recognizability_gate",
        stage=stage,
        recognizable=recognizable,
        subject_match_score=report.get("subject_match_score"),
        recommended_strategy=report.get("recommended_modeling_strategy"),
        summary=report.get("summary"),
    )
    return {
        "recognizable": recognizable,
        "subject_match_score": report.get("subject_match_score"),
        "recommended_strategy": report.get("recommended_modeling_strategy"),
        "summary": report.get("summary"),
        "vision": vision,
    }


async def generate_generic_scene(job_id: str, request: GenericGenerateRequest) -> dict:
    root = _require_job(job_id)
    _write_status(root, state="running", stage="planning_generic_scene")
    spec = await _build_generic_scene_spec(job_id, request.auto_research)
    version = 1 + len(list((root / "scene").glob("model-v*.blend")))
    build = await _execute_generic_spec(job_id, spec, version=version)

    quality_gate: dict | None = None
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
            quality_gate={"recognizable": None, "error": str(exc.detail)},
        )
    else:
        recognizable = quality_gate.get("recognizable")
        if recognizable is True:
            next_stage = "generic_initial_recognizable"
        elif quality_gate.get("recommended_strategy") in {"base_mesh", "hybrid"}:
            next_stage = "generic_needs_strategy_switch"
        else:
            next_stage = "generic_needs_refinement"
        status = _write_status(
            root,
            state="ready",
            stage=next_stage,
            quality_gate={
                "recognizable": recognizable,
                "subject_match_score": quality_gate.get("subject_match_score"),
                "recommended_strategy": quality_gate.get("recommended_strategy"),
                "summary": quality_gate.get("summary"),
            },
        )

    build["status"] = status
    build["quality_gate"] = quality_gate
    return build


@app.post("/v1/jobs/{job_id}/generate", dependencies=[Depends(require_api_token)])
async def generate_generic_scene_api(job_id: str, request: GenericGenerateRequest) -> dict:
    return await generate_generic_scene(job_id, request)


async def refine_generic_scene(job_id: str, request: GenericRefineRequest) -> dict:
    root = _require_job(job_id)
    status_path = root / "status.json"
    if status_path.exists():
        try:
            existing_status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_status = {}
        if existing_status.get("stage") == "generic_needs_strategy_switch":
            raise HTTPException(
                status_code=409,
                detail=(
                    "The active primitive SceneSpec failed the recognizability gate. "
                    "Further primitive refinement is disabled; this job needs a mesh/hybrid strategy."
                ),
            )
        if existing_status.get("stage") == "generic_quality_unverified":
            raise HTTPException(
                status_code=409,
                detail=(
                    "The active model could not be quality-verified. "
                    "Refinement is disabled until visual QA is available again."
                ),
            )

    spec_files = sorted(
        root.glob("scene-spec-v*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not spec_files:
        await generate_generic_scene(job_id, GenericGenerateRequest(auto_research=True))
        spec_files = sorted(
            root.glob("scene-spec-v*.json"),
            key=lambda path: path.stat().st_mtime,
        )

    current_spec_path = spec_files[-1]
    status_path = root / "status.json"
    if status_path.exists():
        try:
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            active_generic = status_payload.get("generic_model")
            active_version = active_generic.get("version") if isinstance(active_generic, dict) else None
            if isinstance(active_version, int):
                accepted_spec_path = root / f"scene-spec-v{active_version}.json"
                if accepted_spec_path.is_file():
                    current_spec_path = accepted_spec_path
        except (OSError, json.JSONDecodeError):
            pass

    current_payload = json.loads(current_spec_path.read_text(encoding="utf-8"))
    current_spec = GenericSceneSpec.model_validate(current_payload["spec"])
    current_version = int(current_payload.get("version") or 1)
    completed = []
    rejected: dict | None = None

    for _ in range(request.iterations):
        vision = await analyze_vision(
            job_id,
            VisionAnalyzeRequest(
                stage="generic_visual_refinement",
                include_references=True,
                include_renders=True,
                max_images=10,
                instruction=(
                    "Critique the newest generic model renders against the references and prompt. "
                    "Prioritize silhouette, proportions, missing major parts, relative placement, and colors. "
                    "Ignore tiny surface detail that cannot be represented by primitive geometry."
                ),
            ),
        )
        report = vision["report"]
        issues = report.get("issues", [])
        high = sum(1 for issue in issues if issue.get("severity") == "high")
        medium = sum(1 for issue in issues if issue.get("severity") == "medium")
        if report.get("recognizable") is True and high == 0 and medium <= 1:
            append_history(
                root,
                "generic_refinement_stop",
                reason="recognizable subject with visual severity threshold met",
            )
            break

        system = (
            "Revise a safe declarative 3D SceneSpec using the visual critique. Return JSON only matching "
            "the supplied schema. This is a SURGICAL REPAIR, not a redesign. Preserve every unaffected "
            "current object and its semantic role. Fix only the 1-3 highest-priority visible defects per pass. "
            "Do not simplify the model, remove defining parts, or replace a detailed assembly with generic "
            "blobs. You may add, resize, rotate, recolor, or reposition objects; remove an object only when "
            "the critique explicitly identifies it as wrong or redundant. You may only use the schema's "
            "allowed primitive shapes. Prefer rod objects for articulated limbs, stems, necks, struts and "
            "connectors when two endpoints are known. Connected parts must touch or overlap their parent "
            "geometry rather than float. Keep the coordinate convention fixed: negative Y is the visible "
            "front surface, positive Y is the back, and Z is up. Do not move face/front details behind the "
            "parent surface or bury them inside it. Preserve good geometry and make the smallest changes that "
            "address the critique. Do not output Python."
        )
        job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
        prompt = (
            f"Current SceneSpec: {json.dumps(current_spec.model_dump(), ensure_ascii=False)}\n"
            f"Visual critique: {json.dumps(report, ensure_ascii=False)}\n"
            f"Spatial/modeling guidance:\n{_generic_spatial_guidance(str(job_request.get('prompt') or ''))}\n"
            "Return the improved full SceneSpec. Keep unaffected object names and parts. Do not reduce the "
            "overall part inventory unless the critique explicitly requires removal. The candidate will be "
            "rejected automatically if it loses too many existing semantic parts."
        )
        try:
            result = await OllamaProxyClient().chat_json(
                model=REASONING_MODEL,
                system=system,
                prompt=prompt,
                schema=GenericSceneSpec.model_json_schema(),
                temperature=0.1,
            )
            normalized = _normalize_scene_spec_payload(result.data, current_spec.title)
            normalized = _enforce_character_visibility(
                normalized,
                str(job_request.get("prompt") or ""),
            )
            normalized = _enforce_subject_geometry(
                normalized,
                str(job_request.get("prompt") or ""),
            )
            revised = GenericSceneSpec.model_validate(normalized)
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            append_history(root, "generic_refinement_failed", error=str(exc))
            raise HTTPException(status_code=502, detail=f"Generic refinement failed: {exc}") from exc

        if revised.model_dump() == current_spec.model_dump():
            append_history(root, "generic_refinement_stop", reason="revised SceneSpec was unchanged")
            break

        regression_reasons = _scene_spec_regression_reasons(current_spec, revised)
        if regression_reasons:
            rejected = {
                "reasons": regression_reasons,
                "current_object_count": len(current_spec.objects),
                "candidate_object_count": len(revised.objects),
            }
            append_history(
                root,
                "generic_refinement_rejected",
                reasons=regression_reasons,
                current_object_count=len(current_spec.objects),
                candidate_object_count=len(revised.objects),
            )
            break

        version = 1 + len(list((root / "scene").glob("model-v*.blend")))
        build = await _execute_generic_spec(job_id, revised, version=version)
        comparison = await _compare_generic_versions(
            root,
            baseline_version=current_version,
            candidate_version=version,
        )
        accepted = bool(comparison.get("candidate_is_better"))
        completed.append(
            {
                "vision": vision,
                "spec": revised.model_dump(),
                "build": build,
                "comparison": comparison,
                "accepted": accepted,
            }
        )
        if not accepted:
            rejected = {
                "reasons": comparison.get("regressions") or [comparison.get("summary") or "candidate did not improve"],
                "summary": comparison.get("summary"),
                "baseline_version": current_version,
                "candidate_version": version,
                "current_object_count": len(current_spec.objects),
                "candidate_object_count": len(revised.objects),
            }
            append_history(
                root,
                "generic_refinement_visual_rejected",
                baseline_version=current_version,
                candidate_version=version,
                summary=comparison.get("summary"),
                regressions=comparison.get("regressions") or [],
            )
            break

        append_history(
            root,
            "generic_revision",
            version=version,
            high_issues=high,
            medium_issues=medium,
            object_count=len(revised.objects),
            comparison_summary=comparison.get("summary"),
        )
        current_spec = revised
        current_version = version

    if rejected:
        active_renders = [
            f"model-v{current_version}-{view}.png"
            for view in (
                "front", "front-left", "left", "back-left", "back",
                "back-right", "right", "front-right", "top",
            )
        ]
        _write_status(
            root,
            state="ready",
            stage="generic_refinement_preserved_previous",
            generic_model={
                "version": current_version,
                "title": current_spec.title,
                "blend": f"model-v{current_version}.blend",
                "renders": active_renders,
                "qa": f"model-v{current_version}-qa.json",
            },
        )

    final_quality: dict | None = None
    try:
        final_quality = await _generic_recognizability_check(
            job_id,
            stage="generic_final_quality",
        )
    except HTTPException as exc:
        append_history(root, "generic_final_quality_unavailable", error=str(exc.detail))
        final_stage = "generic_quality_unverified"
        quality_gate = {"recognizable": None, "error": str(exc.detail)}
    else:
        final_recognizable = final_quality.get("recognizable")
        quality_gate = {
            "recognizable": final_recognizable,
            "subject_match_score": final_quality.get("subject_match_score"),
            "recommended_strategy": final_quality.get("recommended_strategy"),
            "summary": final_quality.get("summary"),
        }
        if final_recognizable is True:
            final_stage = (
                "generic_refinement_preserved_previous"
                if rejected
                else "generic_refinement_complete"
            )
        else:
            final_stage = "generic_needs_strategy_switch"
            append_history(
                root,
                "generic_strategy_switch_needed",
                current_version=current_version,
                recommended_strategy=final_quality.get("recommended_strategy"),
                summary=final_quality.get("summary"),
            )

    status = _write_status(
        root,
        state="ready",
        stage=final_stage,
        quality_gate=quality_gate,
    )
    return {
        "job_id": job_id,
        "iterations": completed,
        "rejected": rejected,
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
