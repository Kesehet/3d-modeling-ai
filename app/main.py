from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import re
import shutil
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError, model_validator

from .artifacts import require_unused_version, reserve_model_version
from .builders import pikachu_script
from .cage_edits import CageEditAction, apply_cage_edit_action
from .component_assembly import component_assembly_script
from .config import (
    JOBS_ROOT,
    OLLAMA_PROXY_BASE_URL,
    REASONING_MODEL,
    VISION_MODEL,
    VISION_MODELS,
    WORKER_URL,
)
from .dashboard import (
    dashboard_page,
    jobs_snapshot,
    public_artifact,
    public_render,
    reconcile_all_running_jobs,
)
from .feature_tasks import (
    FeatureEvaluation,
    FeaturePlan,
    FeatureTask,
    active_or_next_feature,
    begin_feature,
    feature_plan_summary,
    finish_feature,
    invalidate_unverified_acceptances,
    link_component_job,
    load_feature_plan,
    mark_component_ready,
    normalize_feature_plan_payload,
    record_feature_progress,
    save_feature_plan,
)
from .generic_builder import generic_scene_script
from .hard_surface_builder import hard_surface_cage_script
from .history import append_history, load_history
from .mesh_builder import adaptive_loft_script
from .mesh_parts import ParametricGeometry, geometry_context, validate_mesh_part
from .ollama import OllamaProxyClient, OllamaProxyError
from .quality import evaluate_scene_spec_structural, get_benchmark
from .repair import print_repair_script
from .research import research_web_references, write_research_manifest
from .security import require_api_token

app = FastAPI(title="3D Modeling AI", version="0.2.0")
AUTO_IMPROVE_TASKS: dict[str, asyncio.Task[None]] = {}
REFERENCE_RECOVERY_TASKS: dict[str, asyncio.Task[None]] = {}
FEATURE_MAX_ATTEMPTS = 3
AUTO_IMPROVE_HARD_ROUND_CAP = 60
COMPONENT_MAX_DEPTH = 2
COMPONENT_AUTO_IMPROVE_ROUNDS = 6
COMPONENT_MAX_INSTANCES = 16


@app.on_event("startup")
async def reconcile_interrupted_jobs_after_restart() -> None:
    # Any persisted running state predates this process and therefore cannot
    # represent an operation still executing in this API process.
    reconcile_all_running_jobs(force=True)
    if JOBS_ROOT.exists():
        for root in JOBS_ROOT.iterdir():
            if not root.is_dir():
                continue

            plan = load_feature_plan(root)
            if plan is not None and plan.plan_version < 3:
                source = root / "feature-plan.json"
                backup = root / "feature-plan-v1-legacy.json"
                try:
                    if backup.exists():
                        backup.unlink()
                    source.replace(backup)
                except OSError:
                    source.unlink(missing_ok=True)

                append_history(
                    root,
                    "legacy_feature_plan_archived",
                    previous_version=plan.plan_version,
                    reason="recursive component-job feature plan v3 required",
                )

                status = _read_status(root)
                auto = status.get("auto_improve")
                if isinstance(auto, dict) and auto.get("enabled") is True:
                    try:
                        requested_rounds = int(auto.get("max_rounds") or 30)
                    except (TypeError, ValueError):
                        requested_rounds = 30
                    requested_rounds = max(1, min(30, requested_rounds))
                    _write_status(
                        root,
                        auto_improve=_auto_improve_payload(
                            state="scheduled",
                            current_round=0,
                            max_rounds=requested_rounds,
                            reason=(
                                "Replanning the legacy feature backlog with strict "
                                "reference-based acceptance."
                            ),
                        ),
                    )
                    append_history(
                        root,
                        "auto_improve_rearmed_for_feature_plan_v2",
                        requested_rounds=requested_rounds,
                    )
                continue

            reset = invalidate_unverified_acceptances(root)
            if reset:
                append_history(
                    root,
                    "legacy_feature_acceptances_invalidated",
                    count=reset,
                    reason="recursive component-job feature plan v3 enabled",
                )
    _resume_auto_improve_jobs()
    _resume_interrupted_reference_jobs()


Image.MAX_IMAGE_PIXELS = 40_000_000

MAX_REFERENCE_FILES = 8
MAX_REFERENCE_BYTES = 12 * 1024 * 1024
MAX_REFERENCE_TOTAL_BYTES = 48 * 1024 * 1024
IMAGE_FORMATS = {
    "JPEG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
    "WEBP": ("image/webp", ".webp"),
}
ARTIFACT_CATEGORIES = {"references", "reference-candidates", "scene", "renders", "exports", "logs"}

# Keep multimodal requests comfortably below the MediaPitch Ollama proxy body limit.
# The budget counts base64 image characters only; prompt/schema/JSON overhead still has headroom.
VISION_IMAGE_BATCH_MAX_B64_CHARS = 650_000
VISION_IMAGE_ENCODING_PROFILES: tuple[tuple[int, int], ...] = (
    (640, 70),
    (560, 65),
    (480, 60),
    (384, 55),
    (320, 50),
    (256, 45),
    (192, 45),
    (160, 45),
)


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
    render_version: int | None = Field(default=None, ge=1)


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
        "refine_mesh",
        "rebuild_mesh",
    ]
    mesh_representation: Literal["hard_surface_cage", "adaptive_loft"] = Field(
        default="hard_surface_cage",
        description="For mesh actions, choose a horizontal mirrored half-cage or a closed contour loft on X/Y/Z.",
    )
    subject_match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = Field(default="", max_length=2400)
    instructions: list[str] = Field(default_factory=list, max_length=16)
    major_problems: list[str] = Field(default_factory=list, max_length=20)


def _normalize_modeling_director_payload(data: object) -> dict:
    """Tolerate common structured-output variations from multimodal models."""
    if not isinstance(data, dict):
        raise TypeError("Modeling director response is not a JSON object.")

    normalized = dict(data)

    action_aliases = {
        "accept_model": "accept",
        "approve": "accept",
        "approved": "accept",
        "build": "build_procedural",
        "procedural": "build_procedural",
        "revise": "revise_procedural",
        "improve": "revise_procedural",
        "mesh": "build_mesh",
        "base_mesh": "build_mesh",
        "refine_mesh": "refine_mesh",
        "refine mesh": "refine_mesh",
        "mesh_refine": "refine_mesh",
        "improve_mesh": "refine_mesh",
        "revise_mesh": "refine_mesh",
        "rebuild": "rebuild_mesh",
    }
    raw_action = str(
        normalized.get("action")
        or normalized.get("next_action")
        or normalized.get("decision")
        or ""
    ).strip().lower().replace("-", "_").replace(" ", "_")
    normalized["action"] = action_aliases.get(raw_action, raw_action)

    raw_score = normalized.get("subject_match_score", normalized.get("match_score", 0.0))
    try:
        score = float(raw_score)
        if 1.0 < score <= 100.0:
            score /= 100.0
    except (TypeError, ValueError):
        score = 0.0
    normalized["subject_match_score"] = max(0.0, min(1.0, score))

    normalized["summary"] = str(
        normalized.get("summary")
        or normalized.get("reasoning")
        or normalized.get("assessment")
        or ""
    ).strip()[:2400]

    for key, aliases, limit in (
        ("instructions", ("instructions", "instruction", "actions", "next_steps", "recommendations"), 16),
        ("major_problems", ("major_problems", "problems", "issues", "missing_parts", "major_issues"), 20),
    ):
        value = None
        for alias in aliases:
            if alias in normalized:
                value = normalized.get(alias)
                break

        items: list[str] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    text = (
                        item.get("instruction")
                        or item.get("action")
                        or item.get("issue")
                        or item.get("problem")
                        or item.get("description")
                        or item.get("text")
                    )
                    if text:
                        items.append(str(text).strip())
                elif item is not None and str(item).strip():
                    items.append(str(item).strip())
        elif isinstance(value, str):
            text = value.strip()
            if text:
                items = [text]
        elif value is not None and str(value).strip():
            items = [str(value).strip()]

        normalized[key] = items[:limit]

    return normalized


class RefinementComparison(BaseModel):
    candidate_is_better: bool
    summary: str = Field(min_length=1, max_length=2400)
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
    for wrapper in ("comparison", "evaluation", "result"):
        if isinstance(normalized.get(wrapper), dict):
            normalized = normalized[wrapper]
            break
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
    if "candidate_is_better" not in normalized:
        raise ValueError("Visual comparison omitted its explicit keep/revert judgment.")
    normalized.setdefault("summary", normalized.get("reason") or normalized.get("explanation") or "")
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
    render_version: int | None = Field(default=None, ge=1)


class ResearchRequest(BaseModel):
    query: str | None = Field(default=None, max_length=500)
    max_images: int = Field(default=6, ge=1, le=8)


class ReferenceSearchPlan(BaseModel):
    # Exact requested identity used by strict visual acceptance.
    primary_query: str = Field(min_length=1, max_length=180)
    # Broadest still-correct search category. Named identities must remain intact;
    # generic subjects may drop attributes that belong in identity_constraints.
    discovery_query: str = Field(default="", max_length=180)
    alternate_queries: list[str] = Field(default_factory=list, max_length=3)
    subject_description: str = Field(default="", max_length=1200)
    identity_constraints: list[str] = Field(default_factory=list, max_length=12)


class ReferenceCandidateDecision(BaseModel):
    stored_name: str = Field(min_length=1, max_length=220)
    accept: bool
    match_score: float = Field(ge=0.0, le=1.0)
    score_inferred: bool = False
    exact_identity_match: bool
    useful_for_geometry: bool
    reason: str = Field(default="", max_length=1200)


class ReferencePackDecision(BaseModel):
    decisions: list[ReferenceCandidateDecision] = Field(min_length=1, max_length=8)


class ReferenceCoherenceDecision(BaseModel):
    anchor_stored_name: str = Field(default="", max_length=220)
    keep_stored_names: list[str] = Field(default_factory=list, max_length=8)
    search_hint: str = Field(default="", max_length=180)
    summary: str = Field(default="", max_length=1200)


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
    auto_improve_rounds: int = Field(default=0, ge=0, le=30)


class AutoImproveRequest(BaseModel):
    rounds: int = Field(default=30, ge=1, le=30)


class GenericRefineRequest(BaseModel):
    iterations: int = Field(default=1, ge=1, le=3)


class ComponentInstanceSpec(BaseModel):
    location: list[float] = Field(min_length=3, max_length=3)
    rotation_deg: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)
    scale: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0], min_length=3, max_length=3)


class ComponentAssemblySpec(BaseModel):
    rationale: str = Field(default="", max_length=1600)
    instances: list[ComponentInstanceSpec] = Field(min_length=1, max_length=COMPONENT_MAX_INSTANCES)


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


class SceneObjectSpec(ParametricGeometry):
    name: str = Field(min_length=1, max_length=80)
    shape: Literal["sphere", "cube", "cylinder", "cone", "torus", "rod", "beam", "frustum", "wedge", "mesh", "lathe", "sweep"]
    location: list[float] = Field(min_length=3, max_length=3)
    scale: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0], min_length=3, max_length=3, description="Legacy local scale: standard primitives are size 2 (radius 1). Prefer full dimensions. Lathe/sweep coordinates are already in local units; normally use scale=[1,1,1].")
    rotation_deg: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)
    start: list[float] | None = Field(default=None, min_length=3, max_length=3)
    end: list[float] | None = Field(default=None, min_length=3, max_length=3)
    radius: float | None = Field(default=None, gt=0.01, le=5.0)
    color: str = Field(default="#808080", pattern=r"^#[0-9A-Fa-f]{6}$")
    bevel: bool = True
    smooth: bool = True
    vertices: list[tuple[float, float, float]] = Field(default_factory=list, max_length=8192)
    faces: list[Annotated[list[int], Field(min_length=3, max_length=32)]] = Field(default_factory=list, max_length=8192)

    @model_validator(mode="after")
    def validate_mesh_geometry(self):
        validate_mesh_part(self.shape, self.vertices, self.faces)
        return self


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


class CageStation(BaseModel):
    position: float = Field(ge=-10.0, le=10.0)
    # Each pair is [half_width_from_mirror_plane, height]. The first and last
    # points lie on the mirror plane so the mirrored cage closes cleanly.
    profile: list[tuple[float, float]] = Field(min_length=4, max_length=8)


class BooleanCutterSpec(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    shape: Literal["cube", "cylinder", "sphere"] = "cube"
    location: list[float] = Field(min_length=3, max_length=3)
    scale: list[float] = Field(min_length=3, max_length=3)
    rotation_deg: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        min_length=3,
        max_length=3,
    )


class HardSurfaceCageSpec(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    rationale: str = Field(default="", max_length=2400)
    axis: Literal["x", "y"] = "y"
    # Optional for saved specs; new primary plans declare their coordinate intent.
    intended_dimensions_xyz: list[float] | None = Field(default=None, min_length=3, max_length=3)
    color: str = Field(default="#B8BDC6", pattern=r"^#[0-9A-Fa-f]{6}$")
    subdivision_levels: int = Field(default=1, ge=0, le=2)
    bevel_width: float = Field(default=0.04, ge=0.0, le=0.3)
    bevel_segments: int = Field(default=2, ge=1, le=4)
    smooth: bool = True
    presentation_base: bool = False
    stations: list[CageStation] = Field(min_length=4, max_length=16)
    cutters: list[BooleanCutterSpec] = Field(default_factory=list, max_length=16)
    attachments: list[SceneObjectSpec] = Field(default_factory=list, max_length=24)


class GeometryBrief(BaseModel):
    dimensions_xyz: list[float] = Field(min_length=3, max_length=3)
    silhouette_notes: list[str] = Field(min_length=2, max_length=12)
    construction_notes: list[str] = Field(min_length=1, max_length=12)


class GeometryEditBrief(BaseModel):
    diagnosis: str = Field(min_length=1, max_length=1600)
    correction: str = Field(min_length=1, max_length=1600)
    preserve: list[str] = Field(default_factory=list, max_length=12)


async def _reference_geometry_brief(root: Path, *, request: dict, feature: dict,
                                    images: list[str], labels: list[str]) -> dict:
    """Separate visual observation from the reasoning model's coordinate construction."""
    references = [(image, label) for image, label in zip(images, labels, strict=True)
                  if label.startswith("references/")]
    if references:
        images = [image for image, _ in references]
        labels = [label for _, label in references]
    client = OllamaProxyClient()
    errors = []
    for model in VISION_MODELS:
        try:
            result = await client.chat_json(
                model=model,
                system=(
                    "Inspect reference pixels as a 3D artist and write a construction brief for a separate geometry "
                    "engineer. Do not emit mesh coordinates. World X is left/right, Y is front/back, Z is vertical; "
                    "front is -Y. Estimate the ACTIVE FEATURE's bounding dimensions in world XYZ, in consistent "
                    "units with its longest dimension between 1 and 8. Dimensions must describe the reference "
                    "target, not a failed current render. These are approximate 3D dimensions: account for camera "
                    "perspective and foreshortening, never equate projected image width with world X. The geometry "
                    "engineer may correct uncertain estimates. Give concrete silhouette landmarks: where height/width "
                    "changes, transitions, flat versus curved regions, and the dominant axis. Describe only the "
                    "active feature if supplied, reserving other components for later passes. Return JSON."
                ),
                prompt=f"Request: {request.get('prompt')}\nActive feature: {json.dumps(feature)}\nImages: {labels}",
                images=images or None, schema=GeometryBrief.model_json_schema(), temperature=0.0, num_predict=3072,
            )
            brief = GeometryBrief.model_validate(result.data)
            if any(not math.isfinite(v) or v <= 0 or v > 20 for v in brief.dimensions_xyz):
                raise ValueError("Geometry brief requires three positive finite dimensions up to 20 units.")
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            errors.append(f"{model}: {exc}")
            continue
        payload = {**brief.model_dump(), "model": model, "images": labels}
        _write_llm_log(root, "reference-geometry-brief", payload)
        return payload
    raise HTTPException(status_code=502, detail="Reference geometry brief failed: " + " | ".join(errors[-2:]))


def _validate_cage_dimensions(spec: HardSurfaceCageSpec, expected: list[float]) -> None:
    if len(expected) != 3 or any(not math.isfinite(v) or v <= 0 for v in expected):
        raise ValueError("Intended XYZ dimensions must be three positive finite values.")
    positions = [s.position for s in spec.stations]
    half_widths = [p[0] for s in spec.stations for p in s.profile]
    heights = [p[1] for s in spec.stations for p in s.profile]
    length = max(positions) - min(positions)
    width = 2 * max(half_widths)
    height = max(heights) - min(heights)
    actual = [width, length, height] if spec.axis == "y" else [length, width, height]
    # This catches swapped axes / flattened geometry before spending a Blender render.
    if any(abs(a / e - 1) > 0.30 for a, e in zip(actual, expected, strict=True)):
        raise ValueError(f"Cage XYZ dimensions {actual} contradict the planner's intended dimensions {expected}. "
                         "Station positions are horizontal; every profile second coordinate is absolute world Z.")


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
    candidate_models = VISION_MODELS if reference_images else (REASONING_MODEL, *VISION_MODELS)
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


async def _build_feature_plan(
    job_id: str,
    inventory: SubjectInventory | None,
) -> FeaturePlan | None:
    root = _require_job(job_id)
    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    reference_images, reference_labels = _collect_images(
        root,
        VisionAnalyzeRequest(
            stage="feature_inventory",
            include_references=True,
            include_renders=True,
            max_images=10,
        ),
    )
    try:
        component_depth = int(job_request.get("component_depth") or 0)
    except (TypeError, ValueError):
        component_depth = 0
    component_jobs_allowed = component_depth < COMPONENT_MAX_DEPTH
    component_policy = (
        "Recursive component jobs ARE allowed at this depth. Mark build_mode=component_job for an independently "
        "modelable visible assembly that has meaningful internal visible structure and benefits from isolated QA "
        "(for example a wheel assembly that itself contains a tire, rim and visible fasteners). Keep the primary "
        "supporting body/silhouette and ordinary surface details in_place. A component job will be frozen after its "
        "own children and QA pass, then installed into the parent; do not duplicate its internal details as sibling "
        "parent features. Repeated identical components should be one component job with count/symmetry metadata, "
        "not separate rebuilds for each instance. Treat component_job features as terminal assembly leaves at the parent level: "
        "finish all supporting in_place shell/surface work first, and do not make a later in_place feature depend on an "
        "installed component. If a detail only makes sense inside that component, put it in the child component plan. "
        if component_jobs_allowed
        else (
            "This job is already at the maximum recursive component depth. Every feature MUST use build_mode=in_place; "
            "group smaller internal details into the current component instead of creating more child jobs. "
        )
    )
    system = (
        "You are the feature coordinator for an autonomous 3D modeling system. Build an exhaustive but useful "
        "inventory of ALL externally visible features that should be modeled for the requested object. Return JSON "
        "matching the FeaturePlan schema. Include the primary body/silhouette plus visible secondary features such "
        "as wheels, windows, lights, handles, mirrors, openings, trim, screens, feet, buttons, appendages, seams or "
        "other identity-bearing geometry when they are actually visible/relevant. Do not include hidden internals. "
        "Each feature is a separate sub-job. Return features in the intended BUILD ORDER. Give every sub-job a stable "
        "lowercase id, priority, modeling strategy, build_mode, target regions, ownership scope, acceptance criteria, "
        "assembly anchor/notes where relevant, and dependency ids. Priority uses 10=highest/most important and 1=lowest. "
        "Dependencies must form a DAG. " + component_policy +
        "Each criterion must be verifiable using ONLY the geometry owned by that feature. Never require a "
        "dependent component to exist before its supporting feature can pass. Cross-component fit and dimensions "
        "belong to the later installation/assembly check. "
        "The primary silhouette/body should normally be first; dependent details should wait for the supporting "
        "surface. In-place workers share one best-so-far model, so ownership scopes must be narrow enough to prevent "
        "one feature worker from unnecessarily rewriting unrelated geometry. Think like a production 3D modeler and "
        "separate PRIMARY FORM from SECONDARY/TERTIARY FORM. A base_mesh_region feature owns only broad mass, global "
        "proportions, silhouette, large continuous planes/curves, and major transitions. Its acceptance criteria MUST "
        "NOT require panel seams, trim, small openings, glazing boundaries, fasteners, badges, handles, lighting internals, "
        "or other detail that belongs to later surface_cutout/surface_detail/attachment/component features. Create those "
        "as separate later sub-jobs instead. If a component_job installs into a visible recess/opening/socket/mounting "
        "region, create that supporting in_place surface_cutout or surface-detail feature BEFORE the component and make "
        "the component depend on it. This avoids installing finished components into an unprepared primary shell. "
        "Acceptance criteria MUST be visually verifiable from the supplied references/renders. Do not invent exact "
        "millimetres, percentages, tolerances, materials, badge dimensions, or other measurements unless the "
        "user/reference evidence explicitly provides them. Phrase criteria as visible shape, proportion, count, "
        "placement, continuity, and identity checks appropriate to that feature's modeling pass."
    )
    prompt = (
        f"Exact user request: {job_request.get('prompt', '')}\n"
        f"Reference scope: {'parent object; focus only on component '+str(job_request.get('component_name')) if job_request.get('component_job') else 'requested whole object'}\n"
        f"Intended use: {job_request.get('intended_use', '')}\n"
        f"Subject inventory: {json.dumps(inventory.model_dump() if inventory else {}, ensure_ascii=False)}\n"
        f"Reference images in order: {reference_labels}\n"
        "List everything visibly important to the requested object's identity, not just a minimal recognition "
        "checklist. Group only truly inseparable micro-details. Keep left/right or repeated instances in one feature "
        "task when a single coordinated worker should create them together."
    )
    candidate_models = VISION_MODELS if reference_images else (REASONING_MODEL, *VISION_MODELS)
    client = OllamaProxyClient()
    errors: list[str] = []
    for candidate_model in candidate_models:
        try:
            result = await client.chat_json(
                model=candidate_model,
                system=system,
                prompt=prompt,
                images=reference_images or None,
                schema=FeaturePlan.model_json_schema(),
                temperature=0.0,
                num_predict=8192,
            )
            _write_llm_log(
                root,
                "feature-plan-raw",
                {
                    "job_id": job_id,
                    "model": candidate_model,
                    "endpoint": result.endpoint,
                    "usage": result.usage,
                    "images": reference_labels,
                    "raw_response": result.data,
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
            normalized = normalize_feature_plan_payload(
                result.data,
                subject=str(job_request.get("prompt") or ""),
            )
            plan = FeaturePlan.model_validate(normalized)
            if not component_jobs_allowed:
                for feature in plan.features:
                    feature.build_mode = "in_place"
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            errors.append(f"{candidate_model}: {exc}")
            continue

        save_feature_plan(root, plan)
        _write_llm_log(
            root,
            "feature-plan",
            {
                "job_id": job_id,
                "model": candidate_model,
                "endpoint": result.endpoint,
                "usage": result.usage,
                "images": reference_labels,
                "plan": plan.model_dump(),
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        append_history(
            root,
            "feature_plan_created",
            model=candidate_model,
            feature_count=len(plan.features),
            required_count=sum(1 for feature in plan.features if feature.required),
        )
        return plan

    append_history(root, "feature_plan_failed", error=" | ".join(errors[-4:]))
    return None


async def _ensure_feature_plan(
    job_id: str,
    inventory: SubjectInventory | None = None,
) -> FeaturePlan | None:
    root = _require_job(job_id)
    existing = load_feature_plan(root)
    if existing is not None and existing.plan_version >= 3:
        return existing
    if existing is not None:
        append_history(
            root,
            "feature_plan_replanned",
            previous_version=existing.plan_version,
            reason="strict visual acceptance criteria upgrade",
        )
    return await _build_feature_plan(job_id, inventory)


def _component_child_prompt(parent_request: dict, feature_task: FeatureTask) -> str:
    criteria = "; ".join(feature_task.acceptance_criteria) or "Match the visible reference geometry closely."
    assembly = "; ".join(feature_task.assembly_notes)
    parent_prompt = str(parent_request.get("prompt") or "parent object")
    count_note = (
        f"The parent needs {feature_task.count} instance(s), but build ONE canonical reusable component; "
        "the parent assembler will instance it."
        if feature_task.count > 1
        else "Build one canonical component."
    )
    return (
        f"Model ONLY the isolated component '{feature_task.name}' for this parent subject: {parent_prompt}. "
        f"{count_note} Do NOT model the complete parent object. The component must be complete enough to be judged "
        "on its own from multiple angles and later installed as frozen geometry into the parent. "
        f"Visible acceptance criteria: {criteria}. "
        f"Target/ownership context: {', '.join(feature_task.target_regions + feature_task.owner_scope)}. "
        + (f"Assembly context: {assembly}. " if assembly else "")
        + "Keep the component centered around a sensible mounting/origin point. Decompose it recursively only when "
        "a visible child assembly has meaningful independent geometry; do not recurse into microscopic or trivial details."
    )


def _create_component_child_job(parent_job_id: str, feature_task: FeatureTask) -> str:
    parent_root = _require_job(parent_job_id)
    parent_request = json.loads((parent_root / "request.json").read_text(encoding="utf-8"))
    try:
        parent_depth = int(parent_request.get("component_depth") or 0)
    except (TypeError, ValueError):
        parent_depth = 0
    if parent_depth >= COMPONENT_MAX_DEPTH:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Component recursion depth {parent_depth} reached the configured maximum "
                f"of {COMPONENT_MAX_DEPTH}; this feature must be modeled in-place."
            ),
        )

    existing_id = feature_task.component_job_id
    if existing_id:
        existing_root = _job_dir(existing_id)
        if existing_root.is_dir():
            return existing_id

    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    child_job_id = str(uuid.uuid4())
    child_root = _job_dir(child_job_id)
    for category in ARTIFACT_CATEGORIES:
        (child_root / category).mkdir(parents=True, exist_ok=True)

    child_depth = parent_depth + 1
    request = {
        "prompt": _component_child_prompt(parent_request, feature_task),
        "intended_use": parent_request.get("intended_use") or "rendering",
        "target_width_mm": None,
        "job_id": child_job_id,
        "created_at": datetime.now(UTC).isoformat(),
        "component_job": True,
        "component_depth": child_depth,
        "component_name": feature_task.name,
        "component_count_in_parent": feature_task.count,
        "parent_job_id": parent_job_id,
        "parent_feature_id": feature_task.id,
        "parent_prompt": parent_request.get("prompt"),
    }
    (child_root / "request.json").write_text(
        json.dumps(request, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_status(
        child_root,
        job_id=child_job_id,
        state="created",
        stage="component_waiting_for_research",
        component_job={
            "parent_job_id": parent_job_id,
            "parent_feature_id": feature_task.id,
            "depth": child_depth,
            "name": feature_task.name,
        },
    )
    link_component_job(
        parent_root,
        feature_task.id,
        component_job_id=child_job_id,
        component_depth=child_depth,
    )
    append_history(
        parent_root,
        "component_child_created",
        feature_id=feature_task.id,
        feature_name=feature_task.name,
        child_job_id=child_job_id,
        depth=child_depth,
    )
    append_history(
        child_root,
        "component_job_created",
        parent_job_id=parent_job_id,
        parent_feature_id=feature_task.id,
        depth=child_depth,
    )
    return child_job_id


def _active_model_artifact(root: Path, status: dict) -> tuple[int, Path, dict] | None:
    model = status.get("generic_model")
    if not isinstance(model, dict):
        return None
    version = model.get("version")
    blend_name = model.get("blend")
    if not isinstance(version, int) or not isinstance(blend_name, str):
        return None
    blend_path = root / "scene" / Path(blend_name).name
    if not blend_path.is_file():
        return None
    return version, blend_path, model


async def _ensure_component_child_ready(
    parent_job_id: str,
    feature_task: FeatureTask,
) -> dict:
    parent_root = _require_job(parent_job_id)
    child_job_id = _create_component_child_job(parent_job_id, feature_task)
    child_root = _require_job(child_job_id)
    child_status = _read_status(child_root)
    if not _usable_reference_index(child_root):
        inherited = []
        for record in _usable_reference_index(parent_root):
            name = Path(str(record.get("stored_name") or "")).name
            source = parent_root / "references" / name
            if not source.is_file():
                continue
            shutil.copy2(source, child_root / "references" / name)
            inherited.append({**record, "reference_scope": "parent_context",
                              "parent_job_id": parent_job_id, "focus_component": feature_task.name})
        if inherited:
            (child_root / "references.json").write_text(json.dumps(inherited, indent=2), encoding="utf-8")
            append_history(child_root, "parent_references_inherited", count=len(inherited),
                           focus_component=feature_task.name)

    if _active_model_artifact(child_root, child_status) is None:
        append_history(
            parent_root,
            "component_child_build_started",
            feature_id=feature_task.id,
            child_job_id=child_job_id,
        )
        try:
            await generate_generic_scene(
                child_job_id,
                GenericGenerateRequest(auto_research=True, auto_improve_rounds=0),
            )
        except Exception as exc:
            detail = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
            append_history(
                parent_root,
                "component_child_build_failed",
                feature_id=feature_task.id,
                child_job_id=child_job_id,
                error=detail,
            )
            raise

    child_status = _read_status(child_root)
    if not _auto_improve_goal_reached(child_root, child_status):
        await _run_auto_improve(child_job_id, COMPONENT_AUTO_IMPROVE_ROUNDS)
        child_status = _read_status(child_root)

    active = _active_model_artifact(child_root, child_status)
    if active is None or not _auto_improve_goal_reached(child_root, child_status):
        auto_state = child_status.get("auto_improve")
        reason = (
            auto_state.get("reason")
            if isinstance(auto_state, dict)
            else child_status.get("error")
        ) or "Component child did not satisfy its own strict quality gate."
        return {
            "ready": False,
            "child_job_id": child_job_id,
            "reason": str(reason),
            "status": child_status,
        }

    component_version, component_blend, component_model = active
    mark_component_ready(
        parent_root,
        feature_task.id,
        component_version=component_version,
        component_artifact=f"{child_job_id}/scene/{component_blend.name}",
        summary="Child component passed its own quality gate and is frozen for parent assembly.",
    )
    _write_status(
        child_root,
        state="ready",
        stage="component_frozen",
        frozen_component={
            "version": component_version,
            "blend": component_blend.name,
            "parent_job_id": parent_job_id,
            "parent_feature_id": feature_task.id,
        },
    )
    append_history(
        parent_root,
        "component_child_frozen",
        feature_id=feature_task.id,
        feature_name=feature_task.name,
        child_job_id=child_job_id,
        version=component_version,
        blend=component_blend.name,
    )
    return {
        "ready": True,
        "child_job_id": child_job_id,
        "version": component_version,
        "blend_path": component_blend,
        "model": component_model,
        "status": child_status,
    }


def _read_json_if_present(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


async def _plan_component_assembly(
    parent_job_id: str,
    feature_task: FeatureTask,
    child_job_id: str,
) -> ComponentAssemblySpec:
    parent_root = _require_job(parent_job_id)
    child_root = _require_job(child_job_id)
    parent_status = _read_status(parent_root)
    child_status = _read_status(child_root)
    parent_active = _active_model_artifact(parent_root, parent_status)
    child_active = _active_model_artifact(child_root, child_status)
    if parent_active is None or child_active is None:
        raise HTTPException(status_code=409, detail="Parent and frozen child models are required for assembly.")

    parent_version, _, parent_model = parent_active
    child_version, _, child_model = child_active
    parent_request = _read_json_if_present(parent_root / "request.json")

    active_parent_spec = _read_json_if_present(parent_root / f"scene-spec-v{parent_version}.json")
    if not active_parent_spec:
        active_parent_spec = _read_json_if_present(parent_root / f"mesh-spec-v{parent_version}.json")
    parent_qa = _read_json_if_present(parent_root / "exports" / str(parent_model.get("qa") or ""))
    child_qa = _read_json_if_present(child_root / "exports" / str(child_model.get("qa") or ""))

    image_paths: list[Path] = []
    for record in _usable_reference_index(parent_root)[-2:]:
        stored_name = str(record.get("stored_name") or "")
        path = parent_root / "references" / Path(stored_name).name
        if path.is_file():
            image_paths.append(path)

    for view in ("front", "left", "back", "top"):
        path = parent_root / "renders" / f"model-v{parent_version}-{view}.png"
        if path.is_file():
            image_paths.append(path)
    for view in ("front", "left", "front-right", "top"):
        path = child_root / "renders" / f"model-v{child_version}-{view}.png"
        if path.is_file():
            image_paths.append(path)

    images = _encode_vision_images(image_paths) if image_paths else []
    labels = [
        (
            f"parent/{path.name}"
            if path.parent == parent_root / "renders"
            else f"child/{path.name}"
            if path.parent == child_root / "renders"
            else f"reference/{path.name}"
        )
        for path in image_paths
    ]

    expected_instances = max(1, min(COMPONENT_MAX_INSTANCES, int(feature_task.count)))
    system = (
        "You are the assembly coordinator for an autonomous Blender system. The child component is FROZEN accepted "
        "geometry; do not redesign it. Decide only where/how to instance it in the parent. Return JSON matching the "
        "ComponentAssemblySpec schema. Coordinates are Blender world coordinates: X left/right, Y depth, Z up; the "
        "front camera is on negative Y. Use parent dimensions/spec and pixels to place the component on the correct "
        "visible mounting regions. Preserve realistic contact with the parent and avoid floating/intersection errors. "
        f"Return exactly {expected_instances} instance transform(s). For repeated identical parts, reuse this one "
        "frozen component with separate transforms. If unsure, prefer conservative scale and physically plausible contact."
    )
    prompt = (
        f"Exact parent request: {parent_request.get('prompt', '')}\n"
        f"Feature to install: {json.dumps(feature_task.model_dump(), ensure_ascii=False)}\n"
        f"Parent active spec/context: {json.dumps(active_parent_spec, ensure_ascii=False)[:12000]}\n"
        f"Parent QA/bounds: {json.dumps(parent_qa, ensure_ascii=False)}\n"
        f"Frozen child QA/bounds: {json.dumps(child_qa, ensure_ascii=False)}\n"
        f"Image labels: {labels}\n"
        f"Return exactly {expected_instances} transforms. The child geometry is normalized around its own bounds center "
        "before each transform is applied."
    )

    client = OllamaProxyClient()
    errors: list[str] = []
    for candidate_model in (VISION_MODELS if images else (REASONING_MODEL, *VISION_MODELS)):
        try:
            result = await client.chat_json(
                model=candidate_model,
                system=system,
                prompt=prompt,
                images=images or None,
                schema=ComponentAssemblySpec.model_json_schema(),
                temperature=0.0,
                num_predict=4096,
            )
            assembly = ComponentAssemblySpec.model_validate(result.data)
            if len(assembly.instances) != expected_instances:
                raise ValueError(
                    f"Expected {expected_instances} assembly instances, got {len(assembly.instances)}."
                )
            for instance in assembly.instances:
                instance.location = [max(-50.0, min(50.0, float(v))) for v in instance.location]
                instance.rotation_deg = [max(-360.0, min(360.0, float(v))) for v in instance.rotation_deg]
                instance.scale = [max(0.02, min(20.0, float(v))) for v in instance.scale]
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            errors.append(f"{candidate_model}: {exc}")
            continue

        payload = {
            "job_id": parent_job_id,
            "child_job_id": child_job_id,
            "feature_id": feature_task.id,
            "model": candidate_model,
            "images": labels,
            "assembly": assembly.model_dump(),
            "created_at": datetime.now(UTC).isoformat(),
        }
        _write_llm_log(parent_root, "component-assembly-plan", payload)
        append_history(
            parent_root,
            "component_assembly_planned",
            feature_id=feature_task.id,
            child_job_id=child_job_id,
            model=candidate_model,
            instances=len(assembly.instances),
            rationale=assembly.rationale,
        )
        return assembly

    raise HTTPException(
        status_code=502,
        detail="Component assembly planning failed across configured models: " + " | ".join(errors[-4:]),
    )


async def _execute_component_assembly_candidate(
    parent_job_id: str,
    feature_task: FeatureTask,
    child_job_id: str,
    assembly: ComponentAssemblySpec,
) -> dict:
    parent_root = _require_job(parent_job_id)
    child_root = _require_job(child_job_id)
    previous_status = _read_status(parent_root)
    parent_active = _active_model_artifact(parent_root, previous_status)
    child_active = _active_model_artifact(child_root, _read_status(child_root))
    if parent_active is None or child_active is None:
        raise HTTPException(status_code=409, detail="Parent and child model artifacts are required for assembly.")

    baseline_version, parent_blend, _ = parent_active
    child_version, child_blend, _ = child_active
    version = reserve_model_version(parent_root)
    require_unused_version(parent_root, version)
    prefix = f"model-v{version}"
    blend_path = parent_root / "scene" / f"{prefix}.blend"
    qa_path = parent_root / "exports" / f"{prefix}-qa.json"
    _write_status(
        parent_root,
        state="running",
        stage=f"component_assembly_build_v{version}",
        modeling_strategy=previous_status.get("modeling_strategy") or "procedural",
    )
    payload = {
        "tool": "blender_python_exec",
        "arguments": {
            "code": component_assembly_script(),
            "args": {
                "parent_blend_path": str(parent_blend),
                "component_blend_path": str(child_blend),
                "blend_path": str(blend_path),
                "output_dir": str(parent_root / "renders"),
                "exports_dir": str(parent_root / "exports"),
                "qa_path": str(qa_path),
                "prefix": prefix,
                "component_name": feature_task.name,
                "instances": [item.model_dump() for item in assembly.instances],
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
        _write_status(
            parent_root,
            state="ready",
            stage=previous_status.get("stage") or "generic_needs_refinement",
            generic_model=previous_status.get("generic_model"),
            quality_gate=previous_status.get("quality_gate"),
            error=str(exc),
        )
        raise HTTPException(status_code=502, detail=f"Component assembly Blender build failed: {exc}") from exc

    blender_error = _worker_blender_error(result)
    if blender_error:
        _write_status(
            parent_root,
            state="ready",
            stage=previous_status.get("stage") or "generic_needs_refinement",
            generic_model=previous_status.get("generic_model"),
            quality_gate=previous_status.get("quality_gate"),
            error=blender_error,
        )
        raise HTTPException(status_code=502, detail=f"Component assembly Blender script failed: {blender_error}")

    views = (
        "front", "front-left", "left", "back-left", "back",
        "back-right", "right", "front-right", "top",
    )
    expected = [f"{prefix}-{view}.png" for view in views]
    missing = [name for name in expected if not (parent_root / "renders" / name).is_file()]
    if missing or not blend_path.is_file():
        _write_status(
            parent_root,
            state="ready",
            stage=previous_status.get("stage") or "generic_needs_refinement",
            generic_model=previous_status.get("generic_model"),
            quality_gate=previous_status.get("quality_gate"),
        )
        raise HTTPException(
            status_code=502,
            detail=f"Component assembly completed but expected artifacts are missing: {missing}",
        )

    assembly_payload = {
        "parent_job_id": parent_job_id,
        "child_job_id": child_job_id,
        "feature_id": feature_task.id,
        "baseline_version": baseline_version,
        "candidate_version": version,
        "child_version": child_version,
        "assembly": assembly.model_dump(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    (parent_root / f"component-assembly-v{version}.json").write_text(
        json.dumps(assembly_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    append_history(
        parent_root,
        "component_assembly_candidate",
        feature_id=feature_task.id,
        child_job_id=child_job_id,
        baseline_version=baseline_version,
        candidate_version=version,
        child_version=child_version,
        instances=len(assembly.instances),
    )
    _write_status(
        parent_root,
        state="ready",
        stage=f"component_candidate_rendered_v{version}",
        generic_model=previous_status.get("generic_model"),
        quality_gate=previous_status.get("quality_gate"),
    )
    return {
        "baseline_version": baseline_version,
        "candidate_version": version,
        "blend": blend_path.name,
        "renders": expected,
        "qa": qa_path.name,
        "previous_status": previous_status,
        "worker_result": result,
    }


async def _build_and_install_component_feature(
    parent_job_id: str,
    feature_task: FeatureTask,
) -> dict:
    parent_root = _require_job(parent_job_id)
    previous_status = _read_status(parent_root)

    try:
        child = await _ensure_component_child_ready(parent_job_id, feature_task)
    except Exception as exc:
        detail = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
        finish_feature(
            parent_root,
            feature_task.id,
            accepted=False,
            version=None,
            error=detail,
        )
        _write_status(
            parent_root,
            state="ready",
            stage=previous_status.get("stage") or "generic_needs_refinement",
            generic_model=previous_status.get("generic_model"),
            quality_gate=previous_status.get("quality_gate"),
        )
        raise

    if not child.get("ready"):
        reason = str(child.get("reason") or "Component child is not ready.")
        finish_feature(
            parent_root,
            feature_task.id,
            accepted=False,
            version=None,
            error=reason,
        )
        status = _write_status(
            parent_root,
            state="ready",
            stage=previous_status.get("stage") or "generic_needs_refinement",
            generic_model=previous_status.get("generic_model"),
            quality_gate=previous_status.get("quality_gate"),
        )
        append_history(
            parent_root,
            "component_child_not_ready",
            feature_id=feature_task.id,
            child_job_id=child.get("child_job_id"),
            reason=reason,
        )
        return {
            "job_id": parent_job_id,
            "component_feature": feature_task.id,
            "child_job_id": child.get("child_job_id"),
            "accepted": False,
            "reason": reason,
            "status": status,
        }

    child_job_id = str(child["child_job_id"])
    # Refresh the feature record because mark_component_ready persisted the frozen child metadata.
    plan = load_feature_plan(parent_root)
    refreshed_task = (
        next((item for item in plan.features if item.id == feature_task.id), feature_task)
        if plan is not None
        else feature_task
    )
    assembly = await _plan_component_assembly(parent_job_id, refreshed_task, child_job_id)
    candidate = await _execute_component_assembly_candidate(
        parent_job_id,
        refreshed_task,
        child_job_id,
        assembly,
    )
    baseline_version = int(candidate["baseline_version"])
    candidate_version = int(candidate["candidate_version"])

    comparison = await _compare_generic_versions(
        parent_root,
        baseline_version=baseline_version,
        candidate_version=candidate_version,
    )
    evaluation = await _evaluate_feature_candidate(
        parent_job_id,
        refreshed_task,
        baseline_version=baseline_version,
        candidate_version=candidate_version,
    )
    feature_passed = _feature_evaluation_accepts(refreshed_task, evaluation)
    better = bool(comparison.get("candidate_is_better"))
    accept_candidate = bool(feature_passed and better)

    if accept_candidate:
        finish_feature(
            parent_root,
            refreshed_task.id,
            accepted=True,
            version=candidate_version,
            summary=str(evaluation.get("summary") or comparison.get("summary") or ""),
            verified=True,
            acceptance_score=float(evaluation.get("reference_match_score") or 0.0),
            acceptance_model=(
                str(evaluation.get("model"))
                if evaluation.get("model")
                else str(comparison.get("model") or "") or None
            ),
        )
        previous_model_meta = (
            previous_status.get("generic_model")
            if isinstance(previous_status.get("generic_model"), dict)
            else {}
        )
        assembled_components = list(previous_model_meta.get("assembled_components") or [])
        assembled_components.append(
            {
                "feature_id": refreshed_task.id,
                "name": refreshed_task.name,
                "child_job_id": child_job_id,
                "child_version": child.get("version"),
                "parent_version": candidate_version,
                "instances": len(assembly.instances),
            }
        )
        candidate_model = {
            "version": candidate_version,
            "title": previous_model_meta.get("title") or refreshed_task.name,
            "blend": str(candidate["blend"]),
            "renders": list(candidate["renders"]),
            "qa": str(candidate["qa"]),
            "assembled_component": refreshed_task.name,
            "component_child_job_id": child_job_id,
            "assembled_components": assembled_components,
        }
        previous_quality = (
            previous_status.get("quality_gate")
            if isinstance(previous_status.get("quality_gate"), dict)
            else {}
        )
        # Promote only after feature + regression QA have both passed, then immediately
        # re-score the assembled whole object so the autonomous stop condition reflects
        # the actual parent with the frozen component installed.
        _write_status(
            parent_root,
            state="ready",
            stage="component_assembly_validating",
            modeling_strategy=previous_status.get("modeling_strategy") or "procedural",
            generic_model=candidate_model,
            quality_gate=previous_quality,
        )
        try:
            assembled_quality = await _generic_recognizability_check(
                parent_job_id,
                stage="component_assembly_quality",
            )
        except HTTPException as exc:
            assembled_quality = {
                **previous_quality,
                "recognizable": previous_quality.get("recognizable"),
                "summary": str(exc.detail),
            }
            append_history(
                parent_root,
                "component_assembly_quality_unavailable",
                feature_id=refreshed_task.id,
                error=str(exc.detail),
            )
        quality = {
            **assembled_quality,
            "better_than_previous": True,
            "baseline_version": baseline_version,
            "candidate_version": candidate_version,
            "component_feature_id": refreshed_task.id,
            "component_child_job_id": child_job_id,
            "component_feature_passed": True,
            "component_summary": evaluation.get("summary") or comparison.get("summary"),
        }
        status = _write_status(
            parent_root,
            state="ready",
            stage=(
                "component_assembly_recognizable"
                if quality.get("recognizable") is True
                else "component_assembly_accepted"
            ),
            modeling_strategy=previous_status.get("modeling_strategy") or "procedural",
            generic_model=candidate_model,
            quality_gate=quality,
        )
        append_history(
            parent_root,
            "component_assembly_accepted",
            feature_id=refreshed_task.id,
            child_job_id=child_job_id,
            baseline_version=baseline_version,
            candidate_version=candidate_version,
            reference_match_score=evaluation.get("reference_match_score"),
            comparison_summary=comparison.get("summary"),
        )
    else:
        reason = str(
            evaluation.get("summary")
            or comparison.get("summary")
            or "Parent-level QA rejected the installed component."
        )
        finish_feature(
            parent_root,
            refreshed_task.id,
            accepted=False,
            version=None,
            summary=reason,
            error=(
                "Frozen component was preserved, but this installation candidate did not pass "
                "both feature QA and whole-parent regression QA."
            ),
        )
        status = _write_status(
            parent_root,
            state="ready",
            stage=previous_status.get("stage") or "generic_needs_refinement",
            modeling_strategy=previous_status.get("modeling_strategy") or "procedural",
            generic_model=previous_status.get("generic_model"),
            quality_gate=previous_status.get("quality_gate"),
        )
        append_history(
            parent_root,
            "component_assembly_rejected",
            feature_id=refreshed_task.id,
            child_job_id=child_job_id,
            baseline_version=baseline_version,
            candidate_version=candidate_version,
            feature_passed=feature_passed,
            better_than_previous=better,
            reason=reason,
        )

    return {
        "job_id": parent_job_id,
        "component_feature": refreshed_task.id,
        "child_job_id": child_job_id,
        "candidate_version": candidate_version,
        "accepted": accept_candidate,
        "comparison": comparison,
        "feature_evaluation": evaluation,
        "status": status,
    }


def _feature_task_context(root: Path) -> tuple[FeaturePlan | None, FeatureTask | None]:
    plan = load_feature_plan(root)
    if plan is None:
        return None, None
    return plan, active_or_next_feature(plan)


def _normalize_scene_spec_payload(data: object, fallback_title: str) -> dict:
    if not isinstance(data, dict):
        raise TypeError("SceneSpec response is not a JSON object.")

    normalized = dict(data)
    normalized.setdefault("title", (fallback_title.strip() or "Generated model")[:120])
    normalized.setdefault("rationale", "")
    normalized.setdefault("presentation_base", True)

    allowed_shapes = {"sphere", "cube", "cylinder", "cone", "torus", "rod", "beam", "frustum", "wedge", "mesh", "lathe", "sweep"}
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
                "dimensions": item.get("dimensions"),
                "profile": item.get("profile") or [],
                "path": item.get("path") or [],
                "segments": item.get("segments", 64),
                "vertices": item.get("vertices") or [],
                "faces": item.get("faces") or [],
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
    temporary = path / "status.json.tmp"
    temporary.write_text(json.dumps(current, indent=2), encoding="utf-8")
    temporary.replace(status_file)
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


async def _guard_job_action(
    job_id: str,
    action: Callable[[], Awaitable[dict]],
) -> dict:
    """Ensure a failed dashboard/API action cannot strand a job in running."""
    try:
        return await action()
    except Exception as exc:
        root = _job_dir(job_id)
        status_path = root / "status.json"
        if root.is_dir() and status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                status = {}
            if isinstance(status, dict) and status.get("state") == "running":
                detail = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
                stage = str(status.get("stage") or "failed")
                _write_status(
                    root,
                    state="failed",
                    stage=stage,
                    error=detail or exc.__class__.__name__,
                    interrupted=True,
                    interrupted_stage=stage,
                )
                append_history(
                    root,
                    "job_action_failed",
                    stage=stage,
                    error=detail or exc.__class__.__name__,
                    exception_type=exc.__class__.__name__,
                )
        raise



def _quality_snapshot(value: object) -> dict:
    """Keep only the latest quality facts; candidate history belongs in history.json."""

    if not isinstance(value, dict):
        return {}
    snapshot = dict(value)
    snapshot.pop("last_candidate_evaluation", None)
    return snapshot


def _compact_quality_gate(value: object) -> dict:
    """Bound legacy recursive candidate-evaluation state to a single level."""

    if not isinstance(value, dict):
        return {}
    compact = _quality_snapshot(value)
    latest = value.get("last_candidate_evaluation")
    if isinstance(latest, dict):
        compact["last_candidate_evaluation"] = _quality_snapshot(latest)
    return compact


def _read_status(root: Path) -> dict:
    status_path = root / "status.json"
    if not status_path.is_file():
        return {}
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get("quality_gate"), dict):
        payload["quality_gate"] = _compact_quality_gate(payload["quality_gate"])
    return payload


def _auto_improve_goal_reached(root: Path, status: dict) -> bool:
    quality = status.get("quality_gate")
    if isinstance(quality, dict) and quality.get("scope") == "feature":
        return False
    recognizable = isinstance(quality, dict) and quality.get("recognizable") is True
    if not recognizable:
        return False
    if (quality.get("scope") == "whole_object"
            and quality.get("evaluated_version") != (status.get("generic_model") or {}).get("version")):
        return False

    plan = feature_plan_summary(root)
    if plan is None:
        return True
    # Do not call an object "completed" merely because required feature workers
    # exhausted their retries. The autonomous loop may eventually stall safely,
    # but completion means the required visible features were actually accepted.
    return bool(plan.get("required_complete"))


def _auto_improve_progress_signature(root: Path, status: dict) -> tuple[object, ...]:
    """Track accepted visual/feature progress, not bookkeeping or strategy churn."""

    model = status.get("generic_model")
    version = model.get("version") if isinstance(model, dict) else None
    plan = feature_plan_summary(root) or {}
    counts = plan.get("counts") or {}
    return (
        version,
        status.get("working_cage_version"),
        counts.get("accepted", 0),
    )


def _persisted_auto_improve_no_progress_rounds(root: Path) -> int:
    """Carry a no-progress streak across repeated auto-improve invocations.

    Without this, callers can accidentally bypass the token/geometry safety gate
    by scheduling many short runs. Accepted geometry or an accepted feature resets
    the streak; administrative state changes do not.
    """

    streak = 0
    accepted_events = {
        "cage_edit_kept",
        "hard_surface_cage_progress",
        "hard_surface_cage_accepted",
        "adaptive_mesh_accepted",
        "feature_subjob_accepted",
        "component_assembly_accepted",
    }
    for item in reversed(load_history(root)):
        event = item.get("event")
        if event == "auto_improve_round_completed":
            if item.get("progress_changed") is True:
                return 0
            if item.get("progress_changed") is False:
                streak += 1
            continue
        if event in accepted_events:
            return 0
    return streak


def _assembled_parent_requires_safe_stop(root: Path, status: dict) -> bool:
    """Do not let generic refinement discard already-frozen component geometry."""

    model = status.get("generic_model")
    if not isinstance(model, dict):
        return False
    assembled = model.get("assembled_components")
    if not isinstance(assembled, list) or not assembled:
        return False
    quality = status.get("quality_gate")
    if isinstance(quality, dict) and quality.get("recognizable") is True:
        return False
    summary = feature_plan_summary(root) or {}
    return bool(summary.get("required_complete") and not summary.get("next_feature_id"))


def _remaining_feature_attempt_budget(root: Path) -> int:
    plan = load_feature_plan(root)
    if plan is None or plan.plan_version < 2:
        return 0

    remaining = 0
    for feature in plan.features:
        if feature.status in {"accepted", "blocked", "failed"}:
            continue
        remaining += max(0, FEATURE_MAX_ATTEMPTS - int(feature.attempts))
    return remaining


def _feature_queue_is_blocked(root: Path) -> tuple[bool, list[str]]:
    summary = feature_plan_summary(root)
    if not summary:
        return False, []
    unresolved = list(summary.get("required_unresolved") or [])
    no_worker_available = not summary.get("active_feature_id") and not summary.get("next_feature_id")
    return bool(unresolved and no_worker_available), unresolved


def _auto_improve_payload(
    *,
    state: str,
    current_round: int,
    max_rounds: int,
    reason: str = "",
) -> dict:
    return {
        "enabled": True,
        "state": state,
        "current_round": current_round,
        "max_rounds": max_rounds,
        "reason": reason[:1200],
        "updated_at": datetime.now(UTC).isoformat(),
    }


async def _ensure_final_model_quality(job_id: str, status: dict) -> dict:
    """A local feature verdict cannot certify a completed assembly."""
    root = _require_job(job_id)
    plan = feature_plan_summary(root) or {}
    version = (status.get("generic_model") or {}).get("version")
    quality = status.get("quality_gate") or {}
    if not plan.get("required_complete") or version is None:
        return status
    if quality.get("scope") == "whole_object" and quality.get("evaluated_version") == version:
        return status
    try:
        final_quality = await _generic_recognizability_check(
            job_id, stage="final_whole_object_quality", render_version=version,
        )
    except Exception as exc:  # noqa: BLE001 - persist background QA failures for retry
        detail = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
        append_history(root, "final_quality_unavailable", version=version, error=detail)
        final_quality = {"scope": "whole_object", "recognizable": False,
                         "summary": "Final visual evaluation unavailable: " + detail}
    return _write_status(root, quality_gate=final_quality)


def _expanded_auto_improve_round_limit(
    *,
    requested_rounds: int,
    round_number: int,
    remaining_feature_attempts: int,
) -> int:
    """Extend only the default full autonomous budget, never an explicit short run."""

    requested_rounds = max(1, min(30, int(requested_rounds)))
    if requested_rounds < 30 or remaining_feature_attempts <= 0:
        return requested_rounds
    computed_limit = min(
        AUTO_IMPROVE_HARD_ROUND_CAP,
        round_number + remaining_feature_attempts + 2,
    )
    return max(requested_rounds, computed_limit)


async def _run_auto_improve(job_id: str, max_rounds: int) -> None:
    root = _job_dir(job_id)
    if not root.is_dir():
        return

    if not _usable_reference_index(root):
        append_history(root, "reference_research_retry", reason="auto-improve had no usable references")
        try:
            await _ensure_reference_pack(job_id, max_images=5, attempts=2)
        except HTTPException as exc:
            _write_status(
                root,
                state="ready",
                stage="waiting_for_references",
                auto_improve=_auto_improve_payload(
                    state="waiting_for_references",
                    current_round=0,
                    max_rounds=max_rounds,
                    reason=str(exc.detail),
                ),
            )
            append_history(
                root,
                "auto_improve_waiting_for_references",
                error=str(exc.detail),
            )
            return

    requested_rounds = max(1, min(30, int(max_rounds)))
    round_limit = requested_rounds
    round_number = 0
    no_progress_rounds = _persisted_auto_improve_no_progress_rounds(root)
    consecutive_errors = 0
    append_history(root, "auto_improve_started", requested_rounds=requested_rounds)

    while round_number < round_limit:
        if not root.is_dir():
            return

        blocked, unresolved = _feature_queue_is_blocked(root)
        if blocked:
            status = _read_status(root)
            _write_status(
                root,
                state="ready" if isinstance(status.get("generic_model"), dict) else status.get("state", "ready"),
                auto_improve=_auto_improve_payload(
                    state="feature_blocked",
                    current_round=round_number,
                    max_rounds=round_limit,
                    reason=(
                        "Required feature work cannot continue because a dependency "
                        "failed strict QA: " + ", ".join(unresolved[:8])
                    ),
                ),
            )
            append_history(
                root,
                "auto_improve_stopped",
                round=round_number,
                reason="required feature dependency failed",
                unresolved_features=unresolved[:16],
            )
            return

        before = await _ensure_final_model_quality(job_id, _read_status(root))
        if _assembled_parent_requires_safe_stop(root, before):
            _write_status(
                root,
                state="ready",
                stage="assembled_parent_needs_coordinated_replan",
                auto_improve=_auto_improve_payload(
                    state="assembled_needs_review",
                    current_round=round_number,
                    max_rounds=round_limit,
                    reason=(
                        "All frozen required components are installed, but whole-object QA still needs work. "
                        "Generic SceneSpec rebuilding is paused because it could erase accepted component geometry."
                    ),
                ),
            )
            append_history(
                root,
                "auto_improve_stopped",
                round=round_number,
                reason="preserved frozen component assembly before destructive global rebuild",
            )
            return

        if _auto_improve_goal_reached(root, before):
            _write_status(
                root,
                state="ready",
                auto_improve=_auto_improve_payload(
                    state="completed",
                    current_round=round_number,
                    max_rounds=round_limit,
                    reason="Quality gate and visible-feature backlog are satisfied.",
                ),
            )
            append_history(
                root,
                "auto_improve_completed",
                rounds=round_number,
                reason="quality gate satisfied",
            )
            return

        no_progress_limit = (
            6 if before.get("modeling_strategy") == "hard_surface_cage"
            else FEATURE_MAX_ATTEMPTS
        )
        if no_progress_rounds >= no_progress_limit:
            _write_status(
                root,
                state="ready",
                auto_improve=_auto_improve_payload(
                    state="stalled",
                    current_round=round_number,
                    max_rounds=round_limit,
                    reason=(
                        f"Persistent no-progress budget already reached "
                        f"({no_progress_rounds}/{no_progress_limit}); no new model/vision "
                        "calls were made."
                    ),
                ),
            )
            append_history(
                root,
                "auto_improve_stopped",
                round=round_number,
                reason="persistent no-progress budget reached",
                no_progress_rounds=no_progress_rounds,
            )
            return

        round_number += 1
        before_signature = _auto_improve_progress_signature(root, before)
        _write_status(
            root,
            state="running",
            auto_improve=_auto_improve_payload(
                state="running",
                current_round=round_number,
                max_rounds=round_limit,
                reason="Working the next strict feature/refinement task.",
            ),
        )
        append_history(
            root,
            "auto_improve_round_started",
            round=round_number,
            max_rounds=round_limit,
            quality_summary=(
                before.get("quality_gate", {}).get("summary")
                if isinstance(before.get("quality_gate"), dict)
                else None
            ),
        )

        try:
            await refine_generic_scene(job_id, GenericRefineRequest(iterations=1))
        except Exception as exc:  # noqa: BLE001 - background loop must persist/report arbitrary worker failures
            consecutive_errors += 1
            detail = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
            fallback_state = "ready" if isinstance(before.get("generic_model"), dict) else "failed"
            _write_status(
                root,
                state=fallback_state,
                stage=before.get("stage") or "auto_improve_failed",
                generic_model=before.get("generic_model"),
                quality_gate=before.get("quality_gate"),
                auto_improve=_auto_improve_payload(
                    state="retrying" if consecutive_errors < 2 else "stopped",
                    current_round=round_number,
                    max_rounds=round_limit,
                    reason=detail or exc.__class__.__name__,
                ),
            )
            append_history(
                root,
                "auto_improve_round_failed",
                round=round_number,
                error=detail or exc.__class__.__name__,
                consecutive_errors=consecutive_errors,
            )
            if consecutive_errors >= 2:
                append_history(
                    root,
                    "auto_improve_stopped",
                    round=round_number,
                    reason="two consecutive refinement errors",
                )
                return
            await asyncio.sleep(1)
            continue

        consecutive_errors = 0
        after = await _ensure_final_model_quality(job_id, _read_status(root))
        after_signature = _auto_improve_progress_signature(root, after)
        if _auto_improve_goal_reached(root, after):
            _write_status(
                root,
                state="ready",
                auto_improve=_auto_improve_payload(
                    state="completed",
                    current_round=round_number,
                    max_rounds=round_limit,
                    reason="AI satisfied the strict feature backlog and quality gate.",
                ),
            )
            append_history(
                root,
                "auto_improve_completed",
                rounds=round_number,
                reason="strict feature backlog and quality gate satisfied",
            )
            return

        if after_signature == before_signature:
            no_progress_rounds += 1
        else:
            no_progress_rounds = 0

        # The default 30-round autonomous run may extend to finish a healthy
        # finite feature queue. Explicit short budgets (for example a one-round
        # live diagnostic) are hard caps and must never silently expand.
        remaining_feature_attempts = _remaining_feature_attempt_budget(root)
        round_limit = _expanded_auto_improve_round_limit(
            requested_rounds=requested_rounds,
            round_number=round_number,
            remaining_feature_attempts=remaining_feature_attempts,
        )

        _write_status(
            root,
            state="ready",
            auto_improve=_auto_improve_payload(
                state="running",
                current_round=round_number,
                max_rounds=round_limit,
                reason=(
                    f"No measurable progress for {no_progress_rounds} consecutive round(s)."
                    if no_progress_rounds
                    else (
                        f"Continuing strict feature queue; "
                        f"{remaining_feature_attempts} feature attempt(s) remain."
                    )
                ),
            ),
        )
        append_history(
            root,
            "auto_improve_round_completed",
            round=round_number,
            max_rounds=round_limit,
            remaining_feature_attempts=remaining_feature_attempts,
            no_progress_rounds=no_progress_rounds,
            progress_changed=after_signature != before_signature,
        )

        blocked, unresolved = _feature_queue_is_blocked(root)
        if blocked:
            _write_status(
                root,
                state="ready",
                auto_improve=_auto_improve_payload(
                    state="feature_blocked",
                    current_round=round_number,
                    max_rounds=round_limit,
                    reason=(
                        "Required feature work stopped because a prerequisite "
                        "failed strict QA: " + ", ".join(unresolved[:8])
                    ),
                ),
            )
            append_history(
                root,
                "auto_improve_stopped",
                round=round_number,
                reason="required feature dependency failed",
                unresolved_features=unresolved[:16],
            )
            return

        no_progress_limit = 6 if after.get("modeling_strategy") == "hard_surface_cage" else FEATURE_MAX_ATTEMPTS
        if no_progress_rounds >= no_progress_limit:
            _write_status(
                root,
                state="ready",
                auto_improve=_auto_improve_payload(
                    state="stalled",
                    current_round=round_number,
                    max_rounds=round_limit,
                    reason=(
                        f"Stopped after {no_progress_limit} rounds with no accepted best-so-far "
                        "progress to avoid wasting vision/model tokens."
                    ),
                ),
            )
            append_history(
                root,
                "auto_improve_stopped",
                round=round_number,
                reason=f"{no_progress_limit} no-progress rounds",
            )
            return

        await asyncio.sleep(0)

    final_status = _read_status(root)
    _write_status(
        root,
        state="ready" if isinstance(final_status.get("generic_model"), dict) else final_status.get("state", "ready"),
        auto_improve=_auto_improve_payload(
            state="safety_cap_reached",
            current_round=round_number,
            max_rounds=round_limit,
            reason=(
                "Reached the hard autonomous safety cap before strict completion. "
                "The system will not spend more tokens automatically."
            ),
        ),
    )
    append_history(
        root,
        "auto_improve_stopped",
        round=round_number,
        reason="hard safety cap reached",
    )


async def _recover_interrupted_reference_job(job_id: str) -> None:
    root = _job_dir(job_id)
    if not root.is_dir():
        return
    append_history(root, "reference_gated_job_recovery_started")
    try:
        await _guard_job_action(
            job_id,
            lambda: generate_generic_scene(
                job_id, GenericGenerateRequest(auto_research=True, auto_improve_rounds=0)
            ),
        )
    except Exception as exc:  # noqa: BLE001 - recovery must persist failure state
        detail = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
        append_history(root, "reference_gated_job_recovery_failed", error=detail)
        return

    append_history(root, "reference_gated_job_recovery_completed")
    _schedule_auto_improve(job_id, 30)


def _resume_interrupted_reference_jobs() -> None:
    if not JOBS_ROOT.exists():
        return

    recoverable_stages = {
        "researching_references",
        "waiting_for_references",
        "agent_selected_procedural",
        "agent_selected_mesh",
    }
    for root in JOBS_ROOT.iterdir():
        if not root.is_dir() or not (root / "request.json").is_file():
            continue
        status = _read_status(root)
        if status.get("interrupted") is not True:
            continue
        if str(status.get("stage") or "") not in recoverable_stages:
            continue
        if _usable_reference_index(root):
            continue
        if list((root / "renders").glob("*.png")) if (root / "renders").is_dir() else []:
            continue
        if list((root / "scene").glob("model-v*.blend")) if (root / "scene").is_dir() else []:
            continue

        job_id = root.name
        existing = REFERENCE_RECOVERY_TASKS.get(job_id)
        if existing is not None and not existing.done():
            continue

        task = asyncio.create_task(_recover_interrupted_reference_job(job_id))
        REFERENCE_RECOVERY_TASKS[job_id] = task

        def _cleanup(completed: asyncio.Task[None], *, recovery_job_id: str = job_id) -> None:
            if REFERENCE_RECOVERY_TASKS.get(recovery_job_id) is completed:
                REFERENCE_RECOVERY_TASKS.pop(recovery_job_id, None)

        task.add_done_callback(_cleanup)


def _schedule_auto_improve(job_id: str, max_rounds: int = 30) -> bool:
    current = AUTO_IMPROVE_TASKS.get(job_id)
    if current is not None and not current.done():
        return False

    root = _job_dir(job_id)
    if not root.is_dir():
        return False
    max_rounds = max(1, min(30, int(max_rounds)))
    _write_status(
        root,
        auto_improve=_auto_improve_payload(
            state="scheduled",
            current_round=0,
            max_rounds=max_rounds,
            reason="Waiting for autonomous refinement worker.",
        ),
    )

    task = asyncio.create_task(_run_auto_improve(job_id, max_rounds))
    AUTO_IMPROVE_TASKS[job_id] = task

    def _cleanup(completed: asyncio.Task[None]) -> None:
        if AUTO_IMPROVE_TASKS.get(job_id) is completed:
            AUTO_IMPROVE_TASKS.pop(job_id, None)

    task.add_done_callback(_cleanup)
    return True


def _resume_auto_improve_jobs() -> None:
    if not JOBS_ROOT.exists():
        return
    for root in JOBS_ROOT.iterdir():
        if not root.is_dir():
            continue
        status = _read_status(root)
        auto = status.get("auto_improve")
        if not isinstance(auto, dict) or auto.get("enabled") is not True:
            continue
        if auto.get("state") not in {"scheduled", "running", "retrying"}:
            continue
        try:
            max_rounds = int(auto.get("max_rounds") or 30)
            current_round = int(auto.get("current_round") or 0)
        except (TypeError, ValueError):
            max_rounds, current_round = 30, 0
        remaining = max(1, min(30, max_rounds - current_round))
        append_history(
            root,
            "auto_improve_resumed",
            previous_round=current_round,
            remaining_rounds=remaining,
        )
        _schedule_auto_improve(root.name, remaining)


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


def _is_auto_reference_record(record: dict) -> bool:
    provider = str(record.get("provider") or "").lower()
    stored_name = str(record.get("stored_name") or "")
    return (
        provider in {"wikipedia", "wikimedia_commons", "wikimedia"}
        or stored_name.startswith(("web-", "candidate-"))
    )


def _is_usable_reference_record(record: dict) -> bool:
    # User uploads are authoritative. Automatically researched images are trusted
    # only after a multimodal verifier has inspected their actual pixels.
    if record.get("uploaded_at"):
        return True
    if not _is_auto_reference_record(record):
        return bool(record.get("stored_name"))
    if record.get("verified") is not True:
        return False
    try:
        score = float(record.get("match_score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return score >= 0.70 and bool(record.get("exact_identity_match", True))


def _usable_reference_index(root: Path) -> list[dict]:
    return [
        record
        for record in _load_reference_index(root)
        if isinstance(record, dict) and _is_usable_reference_record(record)
    ]


def _prune_unverified_auto_references(root: Path, index: list[dict]) -> list[dict]:
    kept: list[dict] = []
    removed = 0
    for record in index:
        if not isinstance(record, dict):
            continue
        if _is_auto_reference_record(record) and not _is_usable_reference_record(record):
            stored_name = str(record.get("stored_name") or "")
            if stored_name:
                path = root / "references" / Path(stored_name).name
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            removed += 1
            continue
        kept.append(record)

    # Also remove abandoned candidate files from an interrupted research run.
    for path in (root / "references").glob("candidate-*"):
        if path.is_file() and not any(record.get("stored_name") == path.name for record in kept):
            try:
                path.unlink()
            except OSError:
                pass

    if removed:
        append_history(root, "reference_cleanup", removed=removed, reason="unverified automatic references")
    return kept


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


def _encode_vision_images(
    image_paths: list[Path],
    *,
    max_total_chars: int = VISION_IMAGE_BATCH_MAX_B64_CHARS,
) -> list[str]:
    """Encode a whole multimodal batch under a hard proxy-safe request budget."""
    if not image_paths:
        return []

    latest: list[str] = []
    for max_side, quality in VISION_IMAGE_ENCODING_PROFILES:
        latest = [
            _encode_vision_image(path, max_side=max_side, quality=quality)
            for path in image_paths
        ]
        if sum(len(item) for item in latest) <= max_total_chars:
            return latest

    # At 160px/JPEG45 even a 16-image batch should be far below the proxy limit.
    # Fail locally with a useful error rather than repeatedly sending a known-oversize body.
    total = sum(len(item) for item in latest)
    raise ValueError(
        f"Vision image batch is still too large after compression: {total} base64 chars "
        f"(limit {max_total_chars})."
    )


def _collect_images(root: Path, request: VisionAnalyzeRequest) -> tuple[list[str], list[str]]:
    def images_in(folder: str) -> list[Path]:
        paths = []
        for path in (root / folder).glob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                paths.append(path)
        return sorted(paths, key=lambda item: item.stat().st_mtime)

    references = images_in("references") if request.include_references else []
    if references:
        reference_index = {
            str(record.get("stored_name")): record
            for record in _load_reference_index(root)
            if isinstance(record, dict) and record.get("stored_name")
        }
        references = [
            path
            for path in references
            if (
                path.name.startswith("ref-")
                or (
                    path.name in reference_index
                    and _is_usable_reference_record(reference_index[path.name])
                )
            )
        ]
    renders = images_in("renders") if request.include_renders else []

    # Generic refinement must critique one coherent model version. Mixing older model-vN
    # renders into the current set can make the vision model "fix" geometry that no longer exists.
    if renders:
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
            target_version = request.render_version or accepted_version
            if target_version is None:
                target_version = _read_status(root).get("working_cage_version")
            if target_version is None:
                target_version = max(version for version, _ in versioned)
            renders = [path for version, path in versioned if version == target_version]

    if references and renders:
        reference_budget = min(len(references), max(2, request.max_images // 3))
        render_budget = max(1, request.max_images - reference_budget)
        image_paths = references[-reference_budget:] + renders[-render_budget:]
    else:
        image_paths = (references or renders)[-request.max_images :]

    encoded = _encode_vision_images(image_paths)
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
    return await _guard_job_action(job_id, lambda: generate_pikachu_test(job_id))


@app.post("/dashboard/jobs/{job_id}/research", include_in_schema=False)
async def dashboard_research(job_id: str, request: ResearchRequest) -> dict:
    return await _guard_job_action(job_id, lambda: research_job(job_id, request))


@app.post("/dashboard/jobs/{job_id}/generate", include_in_schema=False)
async def dashboard_generate_generic(job_id: str, request: GenericGenerateRequest) -> dict:
    result = await _guard_job_action(job_id, lambda: generate_generic_scene(job_id, request))
    if request.auto_improve_rounds:
        result["auto_improve_started"] = _schedule_auto_improve(
            job_id,
            request.auto_improve_rounds,
        )
    return result


@app.post("/dashboard/jobs/{job_id}/auto-improve", include_in_schema=False)
async def dashboard_auto_improve(job_id: str, request: AutoImproveRequest) -> dict:
    root = _require_job(job_id)
    started = _schedule_auto_improve(job_id, request.rounds)
    return {
        "job_id": job_id,
        "started": started,
        "auto_improve": _read_status(root).get("auto_improve"),
    }


@app.post("/dashboard/jobs/{job_id}/improve", include_in_schema=False)
async def dashboard_improve_generic(job_id: str, request: GenericRefineRequest) -> dict:
    return await _guard_job_action(job_id, lambda: refine_generic_scene(job_id, request))


@app.post("/dashboard/jobs/{job_id}/design", include_in_schema=False)
async def dashboard_submit_design(job_id: str, request: HardSurfaceCageSpec) -> dict:
    return await _guard_job_action(job_id, lambda: submit_model_design(job_id, request))


@app.post("/dashboard/jobs/{job_id}/quality-benchmark", include_in_schema=False)
async def dashboard_quality_benchmark(job_id: str, request: QualityBenchmarkRequest) -> dict:
    return await _guard_job_action(job_id, lambda: evaluate_quality_benchmark(job_id, request))


@app.delete("/dashboard/jobs/{job_id}", include_in_schema=False)
async def dashboard_delete_job(job_id: str) -> dict:
    return await delete_job(job_id)


@app.post("/dashboard/jobs/{job_id}/refine-pikachu", include_in_schema=False)
async def dashboard_refine_pikachu(job_id: str, request: PikachuRefineRequest) -> dict:
    return await _guard_job_action(job_id, lambda: refine_pikachu(job_id, request))


@app.get("/dashboard/jobs/{job_id}/history", include_in_schema=False)
async def dashboard_history(job_id: str) -> dict:
    root = _require_job(job_id)
    return {"job_id": job_id, "history": load_history(root)}


@app.post("/dashboard/jobs/{job_id}/repair-print", include_in_schema=False)
async def dashboard_repair_print(job_id: str, request: PrintRepairRequest) -> dict:
    return await _guard_job_action(job_id, lambda: repair_print_model(job_id, request))


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


async def _plan_reference_search(
    job_request: dict,
    requested_query: str,
) -> ReferenceSearchPlan:
    """Turn a free-form modeling prompt into a precise image-search identity."""
    system = (
        "You create precise web image-search plans for 3D reconstruction. Extract the exact visible subject identity "
        "from the user's modeling request. For a named make/model, character, product, animal species, landmark, etc., "
        "keep that identity intact and remove instructions such as 'make', '3D model', print dimensions, materials, "
        "or workflow chatter. Return TWO query levels: primary_query is the exact visible identity the verifier must "
        "enforce; discovery_query is the broadest search phrase that is still the SAME requested subject. Alternate "
        "queries may add useful attributes/exterior/view words but must not broaden to sibling products or similar-looking "
        "subjects. identity_constraints must state what an image has to visibly depict to count as the requested subject. "
        "NEVER invent a model year, "
        "generation, trim, body style, color, or edition that the user did not request. If the user asks only for a "
        "Volkswagen Polo, for example, a genuine Volkswagen Polo is an identity match regardless of generation; prefer "
        "a coherent set, but do not reject all references for lack of an unspecified year. Return JSON only."
        " For an unnamed generic object, primary_query may preserve identity-defining requested attributes, but "
        "discovery_query MUST be the ordinary category noun phrase, usually 1-3 words. Strip material, color, size, "
        "shape adjectives, part counts, part profiles, dimensions, style, proportions and camera/view words from "
        "discovery_query and preserve those requirements in identity_constraints instead. This broad discovery query "
        "exists only to FIND candidates; the strict pixel verifier still enforces primary_query plus constraints. "
        "For a named make/model, character, product, species or landmark, discovery_query MUST keep the complete named "
        "identity and must never degrade to a parent category such as 'car' or 'character'. Include useful alternate "
        "queries only after discovery_query and primary_query. Return JSON only."
    )
    prompt = (
        f"Full modeling request: {job_request.get('prompt', '')}\n"
        f"Requested research query: {requested_query}\n"
        f"Intended use: {job_request.get('intended_use', '')}\n"
        "Produce a strict image-search plan for geometry references."
    )
    client = OllamaProxyClient()
    # Keep the Pydantic model backwards-compatible for stored/tests plans, but make
    # discovery_query mandatory in the actual planner contract so reasoning models
    # cannot silently omit the broad-recall decision.
    plan_schema = ReferenceSearchPlan.model_json_schema()
    required = list(plan_schema.get("required") or [])
    if "discovery_query" not in required:
        required.append("discovery_query")
    plan_schema["required"] = required

    errors: list[str] = []
    for candidate_model in (REASONING_MODEL, *VISION_MODELS):
        try:
            result = await client.chat_json(
                model=candidate_model,
                system=system,
                prompt=prompt,
                schema=plan_schema,
                temperature=0.0,
                num_predict=2048,
            )
            plan = ReferenceSearchPlan.model_validate(result.data)
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            errors.append(f"{candidate_model}: {exc}")
            continue
        return plan

    return ReferenceSearchPlan(
        primary_query=requested_query[:180],
        discovery_query=requested_query[:180],
        alternate_queries=[],
        subject_description=str(job_request.get("prompt") or requested_query)[:1200],
        identity_constraints=[],
    )



def _reference_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "y", "1", "accept", "accepted", "match", "matched"}:
        return True
    if text in {"false", "no", "n", "0", "reject", "rejected", "mismatch"}:
        return False
    return default


def _normalize_reference_pack_payload(
    data: object,
    records: list[dict],
) -> dict:
    """Tolerate common structured-output variations from multimodal models."""
    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict):
        raw_items = (
            data.get("decisions")
            or data.get("results")
            or data.get("references")
            or data.get("images")
            or data.get("candidates")
            or []
        )
        if isinstance(raw_items, dict):
            raw_items = [
                {"stored_name": key, **(value if isinstance(value, dict) else {"accept": value})}
                for key, value in raw_items.items()
            ]
    else:
        raw_items = []

    if not isinstance(raw_items, list):
        raw_items = []

    expected_names = [
        str(record.get("stored_name") or "")
        for record in records
        if record.get("stored_name")
    ]
    normalized: list[dict] = []
    used_names: set[str] = set()

    for position, raw in enumerate(raw_items[: len(expected_names)]):
        if not isinstance(raw, dict):
            raw = {"accept": raw}

        stored_name = str(
            raw.get("stored_name")
            or raw.get("filename")
            or raw.get("file")
            or raw.get("image")
            or raw.get("name")
            or ""
        ).strip()
        if stored_name not in expected_names:
            stored_name = expected_names[position] if position < len(expected_names) else ""
        if not stored_name or stored_name in used_names:
            continue
        used_names.add(stored_name)

        score_keys = ("match_score", "score", "confidence", "similarity")
        score_present = any(key in raw and raw.get(key) is not None for key in score_keys)
        raw_score = next(
            (raw.get(key) for key in score_keys if key in raw and raw.get(key) is not None),
            0.0,
        )
        try:
            score = float(raw_score or 0.0)
            if 1.0 < score <= 100.0:
                score /= 100.0
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))

        exact_raw = raw.get(
            "exact_identity_match",
            raw.get("identity_match", raw.get("exact_match")),
        )
        useful_raw = raw.get(
            "useful_for_geometry",
            raw.get("geometry_useful", raw.get("useful")),
        )
        exact = _reference_bool(exact_raw, default=False)
        useful = _reference_bool(useful_raw, default=False)
        accept_raw = raw.get("accept", raw.get("accepted", raw.get("relevant")))
        accepted = _reference_bool(
            accept_raw,
            default=bool(exact and useful),
        )
        score_inferred = False
        if not score_present and accepted and exact and useful:
            # Gemma often returns the strict booleans correctly but omits the
            # redundant numeric score. Preserve that decision, while marking
            # the score as inferred so acceptance can require stronger metadata.
            score = 0.85
            score_inferred = True
        if exact_raw is None and accepted and score >= 0.85:
            exact = True
        if useful_raw is None and accepted:
            useful = True
        reason = str(
            raw.get("reason")
            or raw.get("explanation")
            or raw.get("summary")
            or ""
        ).strip()[:1200]

        normalized.append(
            {
                "stored_name": stored_name,
                "accept": accepted,
                "match_score": score,
                "score_inferred": score_inferred,
                "exact_identity_match": exact,
                "useful_for_geometry": useful,
                "reason": reason,
            }
        )

    return {"decisions": normalized}


def _metadata_supports_reference_identity(
    plan: ReferenceSearchPlan,
    record: dict,
) -> bool:
    """Strong metadata support can rescue a cautious visual identity flag, not a bad image."""
    def tokens(value: object) -> list[str]:
        ignored = {
            "a", "an", "the", "car", "vehicle", "model", "image", "photo",
            "front", "rear", "side", "exterior", "view", "of",
        }
        return [
            token
            for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
            if len(token) > 1 and token not in ignored
        ]

    identity_terms = tokens(plan.primary_query)
    if not identity_terms:
        return False
    metadata_text = " ".join(
        str(record.get(key) or "")
        for key in ("title", "source_title", "description")
    ).lower()
    metadata_tokens = set(tokens(metadata_text))
    return all(term in metadata_tokens for term in identity_terms)


def _metadata_title_supports_reference_identity(
    plan: ReferenceSearchPlan,
    record: dict,
) -> bool:
    ignored = {
        "a", "an", "the", "car", "vehicle", "model", "image", "photo",
        "front", "rear", "side", "exterior", "view", "of",
    }
    identity_terms = [
        token
        for token in re.findall(r"[a-z0-9]+", str(plan.primary_query or "").lower())
        if len(token) > 1 and token not in ignored
    ]
    if not identity_terms:
        return False
    title_text = " ".join(
        str(record.get(key) or "")
        for key in ("title", "source_title")
    ).lower()
    title_tokens = set(re.findall(r"[a-z0-9]+", title_text))
    return all(term in title_tokens for term in identity_terms)


def _normalize_reference_coherence_payload(
    data: object,
    records: list[dict],
) -> dict:
    expected = [
        str(record.get("stored_name") or "")
        for record in records
        if record.get("stored_name")
    ]
    anchor = expected[0] if expected else ""
    keep: list[str] = []
    search_hint = ""
    summary = ""

    if isinstance(data, list):
        keep = [str(item).strip() for item in data if str(item).strip() in expected]
    elif isinstance(data, dict):
        anchor_candidate = str(
            data.get("anchor_stored_name")
            or data.get("anchor")
            or data.get("canonical_reference")
            or anchor
        ).strip()
        if anchor_candidate in expected:
            anchor = anchor_candidate

        raw_keep = (
            data.get("keep_stored_names")
            or data.get("keep")
            or data.get("selected")
            or data.get("references")
            or []
        )
        if isinstance(raw_keep, str):
            raw_keep = [raw_keep]
        if isinstance(raw_keep, list):
            for item in raw_keep:
                if isinstance(item, dict):
                    value = str(
                        item.get("stored_name")
                        or item.get("filename")
                        or item.get("name")
                        or ""
                    ).strip()
                else:
                    value = str(item).strip()
                if value in expected and value not in keep:
                    keep.append(value)

        search_hint = str(
            data.get("search_hint")
            or data.get("canonical_identity")
            or data.get("design_identity")
            or ""
        ).strip()[:180]
        summary = str(
            data.get("summary")
            or data.get("reason")
            or data.get("explanation")
            or ""
        ).strip()[:1200]

    if anchor and anchor not in keep:
        keep.insert(0, anchor)
    return {
        "anchor_stored_name": anchor,
        "keep_stored_names": keep[:8],
        "search_hint": search_hint,
        "summary": summary,
    }


async def _cohere_reference_pack(
    root: Path,
    *,
    plan: ReferenceSearchPlan,
    records: list[dict],
) -> tuple[list[dict], list[dict], ReferenceCoherenceDecision | None]:
    """Keep one visually coherent design/generation in an automatic reference pack."""
    if len(records) <= 1:
        return records, [], None

    usable_records: list[dict] = []
    image_paths: list[Path] = []
    for record in records[:8]:
        stored_name = str(record.get("stored_name") or "")
        path = root / "references" / Path(stored_name).name
        if stored_name and path.is_file():
            usable_records.append(record)
            image_paths.append(path)
    if len(usable_records) <= 1:
        return usable_records, [], None

    metadata = [
        {
            "stored_name": record.get("stored_name"),
            "title": record.get("title"),
            "source_title": record.get("source_title"),
            "description": str(record.get("description") or "")[:500],
            "image_url": record.get("image_url"),
        }
        for record in usable_records
    ]
    labels = [path.name for path in image_paths]
    images = _encode_vision_images(image_paths)
    anchor_name = labels[0]
    system = (
        "You curate a coherent multi-view reference pack for 3D reconstruction. The FIRST image is the preferred "
        "canonical anchor because it ranked highest after identity verification. For a named vehicle, product, "
        "character, landmark, or other specific design, keep only images that depict the SAME visible design "
        "generation/body/form as that anchor. Reject older/newer generations and family variants when their geometry "
        "differs, even if they share the requested name. For a generic category such as 'office chair', keep images "
        "that are visually compatible enough to describe one plausible object rather than contradictory designs. "
        "Do not reject the anchor unless it plainly does not depict the requested subject. Return JSON with "
        "anchor_stored_name, keep_stored_names, search_hint, and summary. search_hint should name the canonical "
        "design/generation more specifically when the pixels/metadata support it, otherwise repeat the requested "
        "identity. Never invent a generation/year that is not supported by the supplied images or metadata."
    )
    prompt = (
        f"Requested identity: {plan.primary_query}\n"
        f"Subject description: {plan.subject_description}\n"
        f"Candidate metadata in image order: {json.dumps(metadata, ensure_ascii=False)}\n"
        f"Image labels in order: {labels}\n"
        f"Preferred anchor: {anchor_name}\n"
        "Choose the largest useful subset that can all describe ONE coherent 3D design."
    )

    client = OllamaProxyClient()
    errors: list[str] = []
    decision: ReferenceCoherenceDecision | None = None
    selected_model: str | None = None
    for candidate_model in VISION_MODELS:
        try:
            result = await client.chat_json(
                model=candidate_model,
                system=system,
                prompt=prompt,
                images=images,
                schema=None,
                temperature=0.0,
                num_predict=2048,
            )
            decision = ReferenceCoherenceDecision.model_validate(
                _normalize_reference_coherence_payload(result.data, usable_records)
            )
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            errors.append(f"{candidate_model}: {exc}")
            continue
        selected_model = candidate_model
        break

    if decision is None:
        # Safe fallback: one verified anchor is preferable to a contradictory pack.
        decision = ReferenceCoherenceDecision(
            anchor_stored_name=anchor_name,
            keep_stored_names=[anchor_name],
            search_hint=plan.primary_query,
            summary="Coherence verification failed; retained only the highest-ranked verified anchor.",
        )

    keep_names = set(decision.keep_stored_names)
    if decision.anchor_stored_name:
        keep_names.add(decision.anchor_stored_name)
    if not keep_names:
        keep_names.add(anchor_name)

    kept: list[dict] = []
    rejected: list[dict] = []
    for record in usable_records:
        stored_name = str(record.get("stored_name") or "")
        if stored_name in keep_names:
            kept.append(
                {
                    **record,
                    "coherence_verified": True,
                    "coherence_anchor": decision.anchor_stored_name or anchor_name,
                    "coherence_model": selected_model,
                    "coherence_summary": decision.summary,
                }
            )
        else:
            archived_name = _archive_reference_candidate(root, stored_name)
            rejected.append(
                {
                    **record,
                    "coherence_verified": False,
                    "coherence_anchor": decision.anchor_stored_name or anchor_name,
                    "coherence_model": selected_model,
                    "coherence_summary": decision.summary,
                    "candidate_artifact": archived_name,
                    "verification_reason": (
                        str(record.get("verification_reason") or "")
                        + " Excluded because it conflicts with the canonical reference design."
                    ).strip(),
                }
            )

    append_history(
        root,
        "reference_pack_coherence",
        model=selected_model,
        anchor=decision.anchor_stored_name or anchor_name,
        kept=[record.get("stored_name") for record in kept],
        excluded=[record.get("stored_name") for record in rejected],
        search_hint=decision.search_hint,
        summary=decision.summary,
        errors=errors[-2:],
    )
    return kept, rejected, decision


def _archive_reference_candidate(root: Path, stored_name: str) -> str | None:
    """Keep rejected web candidates available for production diagnostics."""
    safe_name = Path(stored_name).name
    if not safe_name:
        return None
    source = root / "references" / safe_name
    if not source.is_file():
        return None
    destination_dir = root / "reference-candidates"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / safe_name
    try:
        source.replace(destination)
    except OSError:
        return None
    return destination.name


async def _verify_reference_batch(
    root: Path,
    *,
    plan: ReferenceSearchPlan,
    query: str,
    records: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Visually verify automatic search results before they enter the reference pack."""
    usable_records: list[dict] = []
    image_paths: list[Path] = []
    for record in records[:8]:
        stored_name = str(record.get("stored_name") or "")
        path = root / "references" / Path(stored_name).name
        if not stored_name or not path.is_file():
            continue
        usable_records.append(record)
        image_paths.append(path)
    if not image_paths:
        return [], []

    images = _encode_vision_images(image_paths)
    labels = [path.name for path in image_paths]
    metadata = [
        {
            "stored_name": record.get("stored_name"),
            "title": record.get("title"),
            "description": record.get("description"),
            "source_title": record.get("source_title"),
            "provider": record.get("provider"),
        }
        for record in usable_records
    ]
    system = (
        "You are a strict visual reference curator for 3D reconstruction. Inspect the ACTUAL PIXELS of every supplied "
        "candidate image. Accept an image only when it clearly depicts the exact requested subject and is useful for "
        "modeling its visible geometry. For a named make/model/product/character, reject sibling models, unrelated "
        "objects, logos, maps, diagrams, screenshots, isolated parts, "
        "or images where identity is uncertain. Only enforce a particular year/generation/trim when it is explicitly "
        "present in the user's requested identity/constraints; otherwise do not fail a genuine subject merely because "
        "its generation was not specified. Use filenames/titles as supporting evidence, never as a substitute for pixels. "
        "A useful geometry reference "
        "should show a substantial portion of the requested subject with readable silhouette/proportions. Return one "
        "decision for every supplied stored_name. exact_identity_match must be true for accepted images. "
        "useful_for_geometry must also be true for accepted images. Return JSON only."
    )
    prompt = (
        f"Primary requested identity: {plan.primary_query}\n"
        f"Subject description: {plan.subject_description}\n"
        f"Identity constraints: {json.dumps(plan.identity_constraints, ensure_ascii=False)}\n"
        f"Search query used: {query}\n"
        f"Candidate metadata in image order: {json.dumps(metadata, ensure_ascii=False)}\n"
        f"Image labels in order: {labels}\n"
        "Be conservative. A false positive reference can corrupt the entire 3D model."
    )

    client = OllamaProxyClient()
    errors: list[str] = []
    decision_map: dict[str, ReferenceCandidateDecision] = {}
    selected_model: str | None = None
    decision_schema = ReferencePackDecision.model_json_schema()
    candidate_schema = decision_schema["$defs"]["ReferenceCandidateDecision"]
    candidate_schema["required"].append("reason")
    candidate_schema["properties"]["reason"]["minLength"] = 1
    for candidate_model in VISION_MODELS:
        try:
            result = await client.chat_json(
                model=candidate_model,
                system=system,
                prompt=prompt,
                images=images,
                schema=decision_schema,
                temperature=0.0,
                num_predict=4096,
            )
            _write_llm_log(root, "reference-verification-raw", {
                "model": candidate_model, "query": query, "images": labels, "raw": result.data,
            })
            pack = ReferencePackDecision.model_validate(
                _normalize_reference_pack_payload(result.data, usable_records)
            )
            if {decision.stored_name for decision in pack.decisions} != set(labels):
                raise ValueError("Reference verifier omitted one or more candidate judgments.")
            if any(not decision.reason.strip() for decision in pack.decisions):
                raise ValueError("Reference verifier omitted its judgment explanations.")
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            errors.append(f"{candidate_model}: {exc}")
            continue
        selected_model = candidate_model
        decision_map = {decision.stored_name: decision for decision in pack.decisions}
        break

    accepted: list[dict] = []
    rejected: list[dict] = []
    for record in usable_records:
        stored_name = str(record.get("stored_name") or "")
        decision = decision_map.get(stored_name)
        if decision is None:
            decision = ReferenceCandidateDecision(
                stored_name=stored_name,
                accept=False,
                match_score=0.0,
                exact_identity_match=False,
                useful_for_geometry=False,
                reason=(
                    "Visual verifier did not return a decision for this candidate."
                    if selected_model
                    else "Visual verification failed across configured vision models: " + " | ".join(errors[-3:])
                ),
            )

        verified_record = {
            **record,
            "verified": True,
            "match_score": decision.match_score,
            "score_inferred": decision.score_inferred,
            "exact_identity_match": decision.exact_identity_match,
            "useful_for_geometry": decision.useful_for_geometry,
            "verification_reason": decision.reason,
            "verification_model": selected_model,
            "verification_query": query,
            "metadata_identity_support": _metadata_supports_reference_identity(plan, record),
            "verified_at": datetime.now(UTC).isoformat(),
        }
        metadata_identity = _metadata_supports_reference_identity(plan, record)
        title_identity = _metadata_title_supports_reference_identity(plan, record)
        should_accept = (
            decision.accept
            and decision.useful_for_geometry
            and decision.match_score >= 0.70
            and (
                decision.exact_identity_match
                or (metadata_identity and decision.match_score >= 0.78)
            )
            and (
                not decision.score_inferred
                or (decision.exact_identity_match and title_identity)
            )
        )
        if should_accept:
            accepted.append(verified_record)
        else:
            archived_name = _archive_reference_candidate(root, stored_name)
            rejected.append(
                {
                    **verified_record,
                    "candidate_artifact": archived_name,
                }
            )

    append_history(
        root,
        "reference_candidates_verified",
        query=query,
        model=selected_model,
        candidate_count=len(usable_records),
        accepted=len(accepted),
        rejected=len(rejected),
        verifier_errors=errors[-3:],
    )
    return accepted, rejected


async def _verify_reference_candidates(
    root: Path,
    *,
    plan: ReferenceSearchPlan,
    query: str,
    records: list[dict],
    max_images: int,
) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    for start in range(0, len(records), 4):
        batch = records[start : start + 4]
        batch_accepted, batch_rejected = await _verify_reference_batch(
            root,
            plan=plan,
            query=query,
            records=batch,
        )
        remaining_slots = max(0, max_images - len(accepted))
        accepted.extend(batch_accepted[:remaining_slots])
        rejected.extend(batch_rejected)
        for overflow in batch_accepted[remaining_slots:]:
            rejected.append(
                {
                    **overflow,
                    "verification_reason": (
                        str(overflow.get("verification_reason") or "")
                        + " Candidate was valid but exceeded the requested reference-pack size."
                    ).strip(),
                }
            )
            stored_name = str(overflow.get("stored_name") or "")
            if stored_name:
                _archive_reference_candidate(root, stored_name)
        if len(accepted) >= max_images:
            # Candidates not evaluated because we already have enough should not remain
            # loose in the references directory.
            for record in records[start + 4 :]:
                stored_name = str(record.get("stored_name") or "")
                if stored_name:
                    _archive_reference_candidate(root, stored_name)
            break
    return accepted, rejected


def _reference_search_queries(
    plan: ReferenceSearchPlan,
    requested_query: str,
    *,
    limit: int = 4,
) -> list[str]:
    """Order discovery for recall first while preserving strict identity for verification."""
    queries: list[str] = []
    discovery = str(plan.discovery_query or plan.primary_query or requested_query).strip()
    for value in [discovery, plan.primary_query, *plan.alternate_queries, requested_query]:
        value = str(value or "").strip()
        if value and value.casefold() not in {item.casefold() for item in queries}:
            queries.append(value)
        if len(queries) >= limit:
            return queries[:limit]

    # Deterministic view fallbacks stay attached to the broad still-correct subject,
    # not the user's full attribute sentence.
    base_query = discovery or str(plan.primary_query or requested_query).strip()
    for suffix in ("front view", "side view", "rear view"):
        value = f"{base_query} {suffix}".strip()
        if value.casefold() not in {item.casefold() for item in queries}:
            queries.append(value)
        if len(queries) >= limit:
            break
    return queries[:limit]


async def research_job(job_id: str, request: ResearchRequest) -> dict:
    root = _require_job(job_id)
    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    requested_query = (request.query or job_request.get("prompt") or "").strip()
    if not requested_query:
        raise HTTPException(status_code=400, detail="Reference research query is empty.")

    _write_status(root, state="running", stage="researching_references")
    plan = await _plan_reference_search(job_request, requested_query)
    normalized_primary = re.sub(
        r"^(?:a|an|the)\s+",
        "",
        str(plan.primary_query or requested_query).strip(),
        flags=re.IGNORECASE,
    ).strip()
    updates: dict[str, str] = {}
    if normalized_primary and normalized_primary != plan.primary_query:
        updates["primary_query"] = normalized_primary
    normalized_discovery = re.sub(
        r"^(?:a|an|the)\s+",
        "",
        str(plan.discovery_query or normalized_primary or requested_query).strip(),
        flags=re.IGNORECASE,
    ).strip()
    if normalized_discovery and normalized_discovery != plan.discovery_query:
        updates["discovery_query"] = normalized_discovery
    if updates:
        plan = plan.model_copy(update=updates)

    # Keep user-uploaded references, but remove old automatic references that were
    # never verified or previously failed identity matching.
    index = _prune_unverified_auto_references(root, _load_reference_index(root))
    known_hashes = {str(item.get("sha256")) for item in index if item.get("sha256")}
    evaluated_hashes = set(known_hashes)

    queries = _reference_search_queries(plan, requested_query)

    accepted_new: list[dict] = []
    rejected: list[dict] = []
    pages: list[dict] = []
    research_runs: list[dict] = []
    research_errors: list[str] = []

    for query in queries:
        if len(accepted_new) >= request.max_images:
            break
        try:
            payload = await research_web_references(
                query,
                root / "references",
                max_images=request.max_images,
            )
        except (httpx.HTTPError, ValueError) as exc:
            research_errors.append(f"{query}: {exc}")
            continue

        research_runs.append(payload)
        pages.extend(payload.get("pages", []))
        candidate_records = [
            record
            for record in payload.get("references", [])
            if record.get("sha256") not in evaluated_hashes
        ]
        evaluated_hashes.update(str(record["sha256"]) for record in candidate_records if record.get("sha256"))
        batch_accepted, batch_rejected = await _verify_reference_candidates(
            root,
            plan=plan,
            query=query,
            records=candidate_records,
            max_images=request.max_images - len(accepted_new),
        )
        rejected.extend(batch_rejected)
        for record in batch_accepted:
            digest = str(record.get("sha256") or "")
            if digest and digest in known_hashes:
                continue
            accepted_new.append(record)
            if digest:
                known_hashes.add(digest)

    index.extend(accepted_new)

    # Re-cohere the entire automatic pack, including references accepted by older
    # versions of the verifier. User uploads remain authoritative and untouched.
    coherence_decision: ReferenceCoherenceDecision | None = None
    uploaded_records = [
        record for record in index
        if isinstance(record, dict) and record.get("uploaded_at")
    ]
    automatic_records = [
        record for record in index
        if isinstance(record, dict)
        and not record.get("uploaded_at")
        and _is_auto_reference_record(record)
        and _is_usable_reference_record(record)
    ]
    other_records = [
        record for record in index
        if isinstance(record, dict)
        and not record.get("uploaded_at")
        and not _is_auto_reference_record(record)
    ]

    if len(automatic_records) > 1:
        coherent_auto, coherence_rejected, coherence_decision = await _cohere_reference_pack(
            root,
            plan=plan,
            records=automatic_records,
        )
        rejected.extend(coherence_rejected)
        index = [*uploaded_records, *other_records, *coherent_auto]
        kept_names = {
            str(record.get("stored_name") or "")
            for record in coherent_auto
        }
        accepted_new = [
            record
            for record in accepted_new
            if str(record.get("stored_name") or "") in kept_names
        ]

    index = _prune_unverified_auto_references(root, index)
    (root / "references.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "requested_query": requested_query,
        "search_plan": plan.model_dump(),
        "queries": queries,
        "provider": "wikimedia+vision-verifier",
        "created_at": datetime.now(UTC).isoformat(),
        "pages": pages,
        "accepted": accepted_new,
        "coherence": coherence_decision.model_dump() if coherence_decision else None,
        "rejected": rejected,
        "research_errors": research_errors,
        "runs": research_runs,
    }
    write_research_manifest(root / "research.json", manifest)

    usable = _usable_reference_index(root)
    append_history(
        root,
        "research",
        requested_query=requested_query,
        primary_query=plan.primary_query,
        discovery_query=plan.discovery_query,
        queries=queries,
        accepted=len(accepted_new),
        rejected=len(rejected),
        usable_reference_count=len(usable),
        provider="wikimedia+vision-verifier",
        errors=research_errors,
    )
    status = _write_status(
        root,
        state="ready",
        stage="references_researched" if usable else "references_unavailable",
        reference_count=len(usable),
        reference_search_query=plan.primary_query,
        rejected_reference_count=len(rejected),
        reference_gate={
            "required": True,
            "state": "ready" if usable else "blocked",
            "usable_reference_count": len(usable),
            "reason": "" if usable else "Reference research returned no usable verified images.",
        },
    )
    return {
        "job_id": job_id,
        "query": requested_query,
        "search_plan": plan.model_dump(),
        "added": accepted_new,
        "rejected": rejected,
        "pages": pages,
        "research_errors": research_errors,
        "status": status,
    }


async def _ensure_reference_pack(
    job_id: str,
    *,
    max_images: int = 5,
    attempts: int = 2,
) -> list[dict]:
    """Acquire verified references before any modeling/feature-planning work."""
    root = _require_job(job_id)
    usable = _usable_reference_index(root)
    if usable:
        _write_status(
            root,
            reference_count=len(usable),
            reference_gate={
                "required": True,
                "state": "ready",
                "usable_reference_count": len(usable),
                "reason": "",
            },
        )
        return usable

    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    requested_query = " ".join(str(job_request.get("prompt") or "").split())
    if job_request.get("component_job"):
        requested_query = f"{job_request.get('component_name', '')} {job_request.get('parent_prompt', '')}"
    requested_query = requested_query[:440].strip()
    errors: list[str] = []

    for attempt in range(1, max(1, attempts) + 1):
        _write_status(
            root,
            state="running",
            stage="researching_references",
            reference_gate={
                "required": True,
                "state": "searching",
                "attempt": attempt,
                "max_attempts": attempts,
                "usable_reference_count": 0,
            },
        )
        query = requested_query
        if attempt > 1:
            query = f"{requested_query} reference photo exterior multiple views".strip()
        try:
            result = await research_job(
                job_id,
                ResearchRequest(query=query or None, max_images=max_images),
            )
            usable = _usable_reference_index(root)
            if usable:
                append_history(
                    root,
                    "reference_gate_passed",
                    attempt=attempt,
                    usable_reference_count=len(usable),
                )
                _write_status(
                    root,
                    state="ready",
                    stage="references_ready",
                    reference_count=len(usable),
                    reference_gate={
                        "required": True,
                        "state": "ready",
                        "attempt": attempt,
                        "max_attempts": attempts,
                        "usable_reference_count": len(usable),
                    },
                )
                return usable
            detail = (
                f"Reference research returned zero verified images on attempt {attempt}. "
                f"Rejected={len(result.get('rejected') or [])}; "
                f"errors={len(result.get('research_errors') or [])}."
            )
            errors.append(detail)
        except HTTPException as exc:
            errors.append(str(exc.detail))

    reason = " | ".join(errors[-attempts:]) or "No verified reference images were found."
    append_history(
        root,
        "reference_gate_blocked",
        attempts=attempts,
        reason=reason,
    )
    _write_status(
        root,
        state="ready",
        stage="waiting_for_references",
        reference_count=0,
        reference_gate={
            "required": True,
            "state": "blocked",
            "attempt": attempts,
            "max_attempts": attempts,
            "usable_reference_count": 0,
            "reason": reason[:1200],
        },
    )
    raise HTTPException(
        status_code=424,
        detail=(
            "Modeling paused because no verified reference images are available. "
            "Reference research was retried automatically and will not be bypassed. "
            + reason
        ),
    )


@app.post("/v1/jobs/{job_id}/research", dependencies=[Depends(require_api_token)])
async def research_job_api(job_id: str, request: ResearchRequest) -> dict:
    return await _guard_job_action(job_id, lambda: research_job(job_id, request))


@app.get("/v1/jobs/{job_id}/history", dependencies=[Depends(require_api_token)])
async def job_history(job_id: str) -> dict:
    root = _require_job(job_id)
    return {"job_id": job_id, "history": load_history(root)}


@app.get("/v1/jobs/{job_id}/features", dependencies=[Depends(require_api_token)])
async def job_features(job_id: str) -> dict:
    root = _require_job(job_id)
    return {
        "job_id": job_id,
        "feature_plan": feature_plan_summary(root),
    }


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
    images = _encode_vision_images(image_paths)
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
    )[-3:]
    baseline_paths = (
        [
            root / "renders" / f"model-v{baseline_version}-{view}.png"
            for view in views
        ]
        if baseline_version is not None
        else []
    )
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
    images = _encode_vision_images(image_paths)
    labels = [f"{path.parent.name}/{path.name}" for path in image_paths]
    system = (
        "You are a strict visual regression gate for an autonomous 3D modeling system. "
        "Compare the BASELINE model against the CANDIDATE model for the user's exact request. "
        "Set candidate_is_better=true only when the candidate preserves all important recognizable parts "
        "and makes a clear net improvement in silhouette, proportions, connectivity or requested features. "
        "Reject the candidate if it loses a major part, turns detailed geometry into generic blobs, creates "
        "floating/disconnected parts, or is merely different without being clearly better. If uncertain, "
        "set candidate_is_better=false. This is a RELATIVE comparison: both versions may be unfinished. "
        "Missing later planned parts do not veto a clear improvement to the active feature, provided "
        "existing good geometry is preserved. Absolute feature and whole-object completion are checked separately. "
        "Return an explicit candidate_is_better and a non-empty summary explaining the visible difference. "
        "Return only JSON matching the schema."
    )
    prompt = (
        f"User request: {job_request.get('prompt', '')}\n"
        f"Intended use: {job_request.get('intended_use', '')}\n"
        f"Current feature plan and active pass: {json.dumps(feature_plan_summary(root) or {}, ensure_ascii=False)}\n"
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
            _write_llm_log(root, "version-compare-raw", {"model": candidate_model, "raw": result.data})
            normalized = _normalize_refinement_comparison_payload(result.data)
            if not str(normalized.get("summary") or "").strip():
                raise ValueError("Visual comparison omitted its judgment explanation.")
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




def _feature_diagnostic_views(feature_task: FeatureTask) -> tuple[str, ...]:
    if feature_task.strategy == "base_mesh_region":
        return ("front-left", "left", "back-right")
    text = " ".join(
        [
            feature_task.name,
            *feature_task.target_regions,
            *feature_task.owner_scope,
            *feature_task.acceptance_criteria,
        ]
    ).lower()
    if any(token in text for token in ("rear", "back", "tail", "exhaust", "diffuser", "spoiler")):
        return ("back", "back-right", "right")
    if any(token in text for token in ("front", "headlight", "grille", "hood", "bumper", "nose")):
        return ("front", "front-left", "left")
    if any(token in text for token in ("side", "door", "wheel", "arch", "mirror", "window", "skirt")):
        return ("left", "front-left", "back-left")
    if any(token in text for token in ("roof", "top", "scoop")):
        return ("top", "front-left", "back-right")
    return ("front-left", "left", "back-right")


def _feature_evaluation_accepts(
    feature_task: FeatureTask,
    evaluation: dict,
) -> bool:
    try:
        confidence = float(evaluation.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        match_score = float(evaluation.get("reference_match_score") or 0.0)
    except (TypeError, ValueError):
        match_score = 0.0

    feature_text = " ".join(
        [feature_task.name, feature_task.category, *feature_task.target_regions]
    ).lower()
    primary_shape = (
        feature_task.strategy == "base_mesh_region"
        or any(token in feature_text for token in ("silhouette", "body shell", "main body", "primary body"))
    )
    minimum_match = 0.82 if primary_shape else 0.72

    return bool(
        evaluation.get("passed") is True
        and evaluation.get("visible") is True
        and evaluation.get("criteria_satisfied") is True
        and evaluation.get("regression_detected") is not True
        and confidence >= 0.75
        and match_score >= minimum_match
    )


async def _evaluate_feature_candidate(
    job_id: str,
    feature_task: FeatureTask,
    *,
    baseline_version: int | None,
    candidate_version: int,
) -> dict:
    root = _require_job(job_id)
    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    views = _feature_diagnostic_views(feature_task)
    reference_paths: list[Path] = []
    for record in _usable_reference_index(root):
        stored_name = str(record.get("stored_name") or "")
        if not stored_name:
            continue
        path = root / "references" / Path(stored_name).name
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            reference_paths.append(path)
    reference_paths = sorted(
        reference_paths,
        key=lambda path: path.stat().st_mtime,
    )[-3:]
    if not reference_paths:
        return {
            "feature_id": feature_task.id,
            "passed": False,
            "visible": False,
            "criteria_satisfied": False,
            "subject_recognizable": False,
            "confidence": 0.0,
            "reference_match_score": 0.0,
            "regression_detected": False,
            "summary": "Strict feature QA has no verified reference image to compare against.",
            "problems": ["no verified reference images"],
            "protected_geometry_notes": [],
            "model": None,
        }
    baseline_paths = (
        [
            root / "renders" / f"model-v{baseline_version}-{view}.png"
            for view in views
        ]
        if baseline_version is not None
        else []
    )
    candidate_paths = [
        root / "renders" / f"model-v{candidate_version}-{view}.png"
        for view in views
    ]
    if not all(path.is_file() for path in baseline_paths + candidate_paths):
        return {
            "feature_id": feature_task.id,
            "passed": False,
            "visible": False,
            "criteria_satisfied": False,
            "subject_recognizable": False,
            "confidence": 0.0,
            "reference_match_score": 0.0,
            "regression_detected": True,
            "summary": "Feature QA could not run because comparison renders are missing.",
            "problems": ["missing comparison renders"],
            "protected_geometry_notes": [],
            "model": None,
        }

    image_paths = reference_paths + baseline_paths + candidate_paths
    images = _encode_vision_images(image_paths)
    labels = [f"{path.parent.name}/{path.name}" for path in image_paths]
    system = (
        "You are the strict visual QA reviewer for ONE feature sub-job in an autonomous 3D modeling pipeline. "
        "Compare the candidate against the exact references and, when supplied, the baseline. ACCEPTANCE IS ABSOLUTE, "
        "NOT RELATIVE: a feature being better than the baseline is never enough by itself. Set passed=true and "
        "criteria_satisfied=true only when the candidate visibly satisfies ALL acceptance criteria that can be judged "
        "from the supplied pixels. Set visible=true only when the feature itself is clearly visible in the candidate. "
        "subject_recognizable must indicate whether an unfamiliar viewer could recognize the requested overall subject "
        "from the candidate renders. Assess the ACTIVE feature itself; do not fail its local shape because "
        "separate planned components have not been built yet. Whole-object completion is checked separately. "
        "reference_match_score is "
        "an absolute 0..1 score for how closely this feature matches the reference appearance, "
        "shape, placement, count and proportions. If a criterion demands precision that cannot actually be verified "
        "from these images (for example exact millimetres or a 1% tolerance), do NOT pretend it was measured: set "
        "criteria_satisfied=false and explain the unverifiable criterion. If the candidate merely improved but remains "
        "wrong, passed MUST be false. If unrelated protected geometry regressed, regression_detected must be true. "
        "Return JSON only matching the supplied schema."
    )
    comparison_context = (
        f"Reference images come first. Then BASELINE v{baseline_version} views {list(views)}. "
        if baseline_version is not None
        else "Reference images come first. There is no trusted baseline yet. "
    )
    plan = load_feature_plan(root)
    other_features = [{"id": f.id, "name": f.name, "owner_scope": f.owner_scope}
                      for f in (plan.features if plan else []) if f.id != feature_task.id]
    prompt = (
        f"User request: {job_request.get('prompt', '')}\n"
        f"ACTIVE FEATURE SUB-JOB: {json.dumps(feature_task.model_dump(), ensure_ascii=False)}\n"
        f"Other planned features (owned by later passes): "
        f"{json.dumps(other_features)}\n"
        f"Images in order: {labels}\n"
        + comparison_context
        + f"Then CANDIDATE v{candidate_version} views {list(views)}.\n"
        + "Check every acceptance criterion explicitly. Passing means DONE, not just improved. "
        + "Mention any unmet/unverifiable criterion in problems and any protected geometry regression separately."
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
                schema=FeatureEvaluation.model_json_schema(),
                temperature=0.0,
                num_predict=4096,
            )
            _write_llm_log(root, "feature-qa-raw", {
                "model": candidate_model, "candidate_version": candidate_version,
                "raw": result.data, "images": labels,
            })
            payload = dict(result.data) if isinstance(result.data, dict) else {}
            for wrapper in ("evaluation", "feature_evaluation", "result"):
                if isinstance(payload.get(wrapper), dict):
                    payload = payload[wrapper]
                    break
            payload.setdefault("feature_id", feature_task.id)
            if payload["feature_id"] != feature_task.id:
                raise ValueError("Vision evaluated a different feature.")
            evaluation = FeatureEvaluation.model_validate(payload)
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            errors.append(f"{candidate_model}: {exc}")
            continue

        response = {
            **evaluation.model_dump(),
            "model": candidate_model,
            "baseline_version": baseline_version,
            "candidate_version": candidate_version,
            "images": labels,
            "created_at": datetime.now(UTC).isoformat(),
        }
        _write_llm_log(root, "feature-qa", response)
        append_history(
            root,
            "feature_subjob_qa",
            feature_id=feature_task.id,
            feature_name=feature_task.name,
            passed=evaluation.passed,
            visible=evaluation.visible,
            confidence=evaluation.confidence,
            regression_detected=evaluation.regression_detected,
            summary=evaluation.summary,
        )
        return response

    return {
        "feature_id": feature_task.id,
        "passed": False,
        "visible": False,
        "criteria_satisfied": False,
        "subject_recognizable": False,
        "confidence": 0.0,
        "reference_match_score": 0.0,
        "regression_detected": True,
        "summary": "Feature QA failed across configured vision models.",
        "problems": errors[-4:] or ["feature QA unavailable"],
        "protected_geometry_notes": [],
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
        "- Primitive location is its center. Prefer dimensions=[full width, full depth, full height] BEFORE rotation.\n"
        "- Legacy scale is NOT dimensions: a cube/cylinder/sphere has native size 2; scale=[1,1,1] is 2 units wide.\n"
        "- For a box, bottom Z = location[2] - dimensions[2]/2 and top Z = location[2] + dimensions[2]/2.\n"
        "- Lathe revolves a closed [radius,Z] material profile around local Z; include inner and outer walls to make cavities.\n"
        "- Sweep builds a smooth, capped round tube along a local XYZ path with an explicit radius.\n"
        "- Lathe/sweep/mesh use local coordinates plus location offset; normally keep scale=[1,1,1].\n"
        "- Visible details should sit on the intended surface instead of being buried inside another part.\n"
        "- Connected parts should touch or overlap when the real object is physically connected.\n"
        "- Use rod/beam only when start and end are explicitly defined; otherwise use a solid primitive.\n"
        "- Use mesh with local vertices and face indices for shaped panels that primitives cannot express.\n"
        "- Use the reference images, not object-name heuristics, to decide proportions, silhouette and part placement."
    )


async def _build_generic_scene_spec(
    job_id: str,
    auto_research: bool,
    *,
    feature_task: FeatureTask | None = None,
) -> GenericSceneSpec:
    root = _require_job(job_id)
    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))

    if auto_research and not _usable_reference_index(root):
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
    feature_plan = await _ensure_feature_plan(job_id, inventory)
    feature_plan_context = feature_plan.model_dump() if feature_plan is not None else {}
    active_feature_context = feature_task.model_dump() if feature_task is not None else {}

    system = (
        "You are the modeling agent for Blender. Inspect the user request and supplied reference images, then "
        "design the best declarative SceneSpec with primitives, lathe, sweep and arbitrary mesh parts. "
        "Lathe revolves a closed [radius,Z] material profile, preserving hollow interiors when the profile includes "
        "inner walls. Sweep makes smooth round-section curved parts from a short XYZ path and radius. "
        "Use these parametric tools instead of approximating curves with boxes. You own the "
        "geometry decisions: silhouette, proportions, primitive choice, number of parts, placement, rotation, "
        "connectivity and colors. Do not rely on hard-coded object-family templates. Return JSON only matching "
        "the supplied schema. Use rod or beam only when both start and end are explicitly provided. Coordinates "
        "should normally stay within -8..8, with Z up and negative Y facing the front camera. If primitives are "
        "a poor fit, still produce the strongest honest blockout you can; the AI director can switch the next "
        "pass to the continuous-mesh strategy. If an ACTIVE FEATURE is supplied, this is a feature-scoped "
        "construction pass: author only geometry owned by that active feature. Do not prebuild later features, "
        "supports, mounts, trim, repeated components, or other planned parts merely to make the whole object look "
        "more complete. The outer modeling loop adds later features after the active one passes review."
    )
    prompt = (
        f"User request: {job_request.get('prompt', '')}\n"
        f"Intended use: {job_request.get('intended_use', '')}\n"
        f"Target width mm: {job_request.get('target_width_mm')}\n"
        f"Visual reference analysis: {json.dumps(visual_context, ensure_ascii=False)}\n"
        f"Web research context: {json.dumps(research_context, ensure_ascii=False)}\n"
        f"Required subject inventory: {json.dumps(inventory_context, ensure_ascii=False)}\n"
        f"Coordinated visible-feature plan: {json.dumps(feature_plan_context, ensure_ascii=False)}\n"
        f"ACTIVE FEATURE FOR THIS PASS: {json.dumps(active_feature_context, ensure_ascii=False)}\n"
        f"Spatial/modeling guidance:\n{_generic_spatial_guidance(str(job_request.get('prompt') or ''))}\n"
        "Create the SceneSpec for the CURRENT construction state. When ACTIVE FEATURE is non-empty, include only "
        "geometry owned by that feature; every other non-accepted feature must remain absent until its own pass. "
        "Do not add placeholders for later features. Favor recognizability of the active feature and its reference "
        "evidence over whole-object completeness. IMPORTANT OWNERSHIP RULE: any FeaturePlan entry with "
        "build_mode=component_job is owned by its isolated child job. Do NOT author that component's final geometry "
        "in this parent SceneSpec. Shape a mounting region only when that mounting/support geometry itself belongs "
        "to the ACTIVE FEATURE; otherwise leave it for its later feature pass."
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
    # Vision has already supplied the observations and inventory above. Let the
    # reasoning model construct coordinates, as for continuous mesh planning.
    planner_models = (REASONING_MODEL, *VISION_MODELS)
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
                images=(reference_images or None) if candidate_model in VISION_MODELS else None,
                schema=GenericSceneSpec.model_json_schema(),
                temperature=0.0,
                num_predict=8192,
            )
            normalized = _normalize_scene_spec_payload(
                candidate_result.data,
                str(job_request.get("prompt") or "Generated model"),
            )
            candidate_spec = GenericSceneSpec.model_validate(normalized)
            if job_request.get("component_job") is True:
                candidate_spec.presentation_base = False
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
    activate_status: bool = True,
) -> dict:
    root = _require_job(job_id)
    require_unused_version(root, version)
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
    candidate_model = {
            "version": version,
            "title": spec.title,
            "blend": blend_path.name,
            "renders": expected,
            "qa": qa_path.name,
        }
    status = _write_status(
        root,
        state="ready",
        stage=f"generic_rendered_v{version}",
        **({"generic_model": candidate_model} if activate_status else {}),
    )
    return {
        "job_id": job_id,
        "status": status,
        "candidate_model": candidate_model,
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



def _normalize_hard_surface_cage_payload(
    data: object,
    fallback_title: str,
    *,
    complexity: str = "simple",
) -> dict:
    if not isinstance(data, dict):
        raise TypeError("Hard-surface cage response is not a JSON object.")

    normalized = dict(data)
    station_keys = (
        "stations",
        "sections",
        "profiles",
        "slices",
        "cross_sections",
        "cage_stations",
        "longitudinal_stations",
        "control_sections",
    )
    for wrapper in (
        "hard_surface_cage_spec",
        "hard_surface_cage",
        "cage_spec",
        "cage",
        "mesh_spec",
        "mesh",
        "spec",
        "result",
        "output",
        "data",
    ):
        candidate = normalized.get(wrapper)
        if isinstance(candidate, dict) and any(key in candidate for key in station_keys):
            normalized = dict(candidate)
            break

    normalized["title"] = str(normalized.get("title") or fallback_title or "Hard-surface cage")[:120]
    normalized["rationale"] = str(
        normalized.get("rationale")
        or normalized.get("reasoning")
        or normalized.get("description")
        or ""
    )[:2400]
    axis = str(normalized.get("axis") or normalized.get("length_axis") or "y").lower()
    normalized["axis"] = axis if axis in {"x", "y"} else "y"

    complexity = str(complexity or "simple").lower()
    if complexity == "complex":
        min_stations = 8
        min_profile_points = 6
    elif complexity == "moderate":
        min_stations = 6
        min_profile_points = 5
    else:
        min_stations = 4
        min_profile_points = 4

    color = str(normalized.get("color") or normalized.get("base_color") or "#B8BDC6")
    if not (
        len(color) == 7
        and color.startswith("#")
        and all(ch in "0123456789abcdefABCDEF" for ch in color[1:])
    ):
        color = "#B8BDC6"
    normalized["color"] = color.upper()

    try:
        subdivision = int(normalized.get("subdivision_levels", normalized.get("subdivision", 1)) or 0)
    except (TypeError, ValueError):
        subdivision = 1
    normalized["subdivision_levels"] = max(0, min(2, subdivision))
    try:
        bevel_width = float(normalized.get("bevel_width", 0.04) or 0.0)
    except (TypeError, ValueError):
        bevel_width = 0.04
    normalized["bevel_width"] = max(0.0, min(0.3, bevel_width))
    try:
        bevel_segments = int(normalized.get("bevel_segments", 2) or 2)
    except (TypeError, ValueError):
        bevel_segments = 2
    normalized["bevel_segments"] = max(1, min(4, bevel_segments))
    normalized["smooth"] = bool(normalized.get("smooth", True))
    # QA renders must contain only model geometry. A display pedestal changes
    # silhouette judgments and previously made failed vehicle cages look like
    # mushroom-shaped objects.
    normalized["presentation_base"] = False

    raw_stations = None
    for key in station_keys:
        candidate = normalized.get(key)
        if candidate:
            raw_stations = candidate
            break
    if isinstance(raw_stations, dict):
        for key in ("items", *station_keys):
            value = raw_stations.get(key)
            if isinstance(value, list):
                raw_stations = value
                break
        else:
            values = list(raw_stations.values())
            raw_stations = values if values and all(isinstance(v, dict) for v in values) else []
    if not isinstance(raw_stations, list):
        raise TypeError("Hard-surface cage stations could not be normalized to a list.")

    def resample_profile(profile: list[list[float]], target: int) -> list[list[float]]:
        if len(profile) == target:
            return [list(point) for point in profile]
        if len(profile) < 2:
            return profile
        result: list[list[float]] = []
        for index in range(target):
            t = index * (len(profile) - 1) / (target - 1)
            left = int(t)
            right = min(len(profile) - 1, left + 1)
            mix = t - left
            result.append(
                [
                    profile[left][0] * (1.0 - mix) + profile[right][0] * mix,
                    profile[left][1] * (1.0 - mix) + profile[right][1] * mix,
                ]
            )
        return result

    stations: list[dict] = []
    for raw in raw_stations[:20]:
        if not isinstance(raw, dict):
            continue
        raw_position = raw.get("position")
        if raw_position is None:
            for key in ("axis_position", "offset", "distance", "station", normalized["axis"]):
                if raw.get(key) is not None:
                    raw_position = raw.get(key)
                    break
        try:
            position = max(-10.0, min(10.0, float(raw_position)))
        except (TypeError, ValueError):
            continue

        raw_profile = None
        for key in (
            "profile",
            "half_profile",
            "points",
            "contour",
            "vertices",
            "control_points",
            "cross_section",
        ):
            candidate = raw.get(key)
            if candidate is not None:
                raw_profile = candidate
                break

        if raw_profile is None:
            widths = raw.get("half_widths") or raw.get("widths")
            heights = raw.get("heights") or raw.get("z_values")
            if isinstance(widths, list) and isinstance(heights, list):
                raw_profile = list(zip(widths, heights))

        if isinstance(raw_profile, dict):
            raw_profile = (
                raw_profile.get("points")
                or raw_profile.get("profile")
                or raw_profile.get("vertices")
                or raw_profile.get("control_points")
            )
        if not isinstance(raw_profile, list):
            continue

        profile: list[list[float]] = []
        saw_negative_width = False
        for point in raw_profile[:12]:
            if isinstance(point, dict):
                width = (
                    point.get("half_width")
                    if point.get("half_width") is not None
                    else point.get("width")
                    if point.get("width") is not None
                    else point.get("x")
                    if normalized["axis"] == "y"
                    else point.get("y")
                )
                height = (
                    point.get("height")
                    if point.get("height") is not None
                    else point.get("z")
                    if point.get("z") is not None
                    else point.get("v")
                )
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                width, height = point[0], point[1]
            else:
                continue
            try:
                raw_width = float(width)
                z = max(-10.0, min(10.0, float(height)))
            except (TypeError, ValueError):
                continue
            saw_negative_width = saw_negative_width or raw_width < 0
            profile.append([max(-10.0, min(10.0, raw_width)), z])

        if saw_negative_width:
            positive = [[abs(width), height] for width, height in profile if width >= -0.001]
            if len(positive) < 3:
                positive = [[abs(width), height] for width, height in profile]
            positive.sort(key=lambda point: point[1])
            profile = positive

        if len(profile) < 3:
            continue
        profile = [[max(0.0, point[0]), point[1]] for point in profile]
        profile[0][0] = 0.0
        profile[-1][0] = 0.0
        stations.append({"position": position, "profile": profile})

    stations.sort(key=lambda item: item["position"])
    deduped: list[dict] = []
    for station in stations:
        if deduped and abs(station["position"] - deduped[-1]["position"]) < 0.05:
            continue
        deduped.append(station)

    # Keep the existing human-style midpoint behavior for terse three-section
    # output, then apply a subject-dependent minimum geometry budget below.
    if 2 <= len(deduped) < 4:
        expanded: list[dict] = []
        for index, station in enumerate(deduped[:-1]):
            nxt = deduped[index + 1]
            expanded.append(station)
            target_size = max(
                min_profile_points,
                min(8, max(len(station["profile"]), len(nxt["profile"]))),
            )
            a = resample_profile(station["profile"], target_size)
            b = resample_profile(nxt["profile"], target_size)
            expanded.append(
                {
                    "position": (station["position"] + nxt["position"]) / 2.0,
                    "profile": [
                        [(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0]
                        for pa, pb in zip(a, b)
                    ],
                }
            )
        expanded.append(deduped[-1])
        deduped = expanded

    if len(deduped) < 4:
        raise ValueError("Hard-surface cage needs at least two usable reference stations.")

    target_size = max(
        min_profile_points,
        min(8, round(sum(len(s["profile"]) for s in deduped) / len(deduped))),
    )
    for station in deduped:
        station["profile"] = resample_profile(station["profile"], target_size)
        station["profile"][0][0] = 0.0
        station["profile"][-1][0] = 0.0

    # Do not fabricate missing shape information by interpolating a sparse
    # cage. If the planner did not provide enough independent control sections
    # for the declared complexity, reject the plan and let another model/pass
    # produce a genuinely richer cage.
    if len(deduped) < min_stations:
        raise ValueError(
            f"Hard-surface cage is under-specified for {complexity} geometry: "
            f"{len(deduped)} stations supplied; at least {min_stations} required."
        )

    axis_span = deduped[-1]["position"] - deduped[0]["position"]
    max_half_width = max(point[0] for station in deduped for point in station["profile"])
    min_z = min(point[1] for station in deduped for point in station["profile"])
    max_z = max(point[1] for station in deduped for point in station["profile"])
    height_span = max_z - min_z
    if axis_span < 0.5 or max_half_width < 0.1 or height_span < 0.2:
        raise ValueError("Hard-surface cage has degenerate body dimensions.")

    normalized["stations"] = deduped[:16]

    raw_cutters = (
        normalized.get("cutters")
        or normalized.get("boolean_cutters")
        or normalized.get("cutouts")
        or normalized.get("openings")
        or []
    )
    if isinstance(raw_cutters, dict):
        raw_cutters = list(raw_cutters.values())
    cutters: list[dict] = []
    if isinstance(raw_cutters, list):
        for index, item in enumerate(raw_cutters[:16]):
            if not isinstance(item, dict):
                continue
            shape = str(item.get("shape") or item.get("type") or "cube").lower()
            if shape not in {"cube", "cylinder", "sphere"}:
                shape = "cube"
            location = item.get("location") or item.get("center") or [0, 0, 0]
            scale = item.get("scale") or item.get("size") or [1, 1, 1]
            rotation = item.get("rotation_deg") or item.get("rotation") or [0, 0, 0]
            if not (
                isinstance(location, list) and len(location) >= 3
                and isinstance(scale, list) and len(scale) >= 3
                and isinstance(rotation, list) and len(rotation) >= 3
            ):
                continue
            try:
                cutters.append(
                    {
                        "name": str(item.get("name") or f"cut-{index + 1}")[:80],
                        "shape": shape,
                        "location": [max(-20.0, min(20.0, float(v))) for v in location[:3]],
                        "scale": [max(0.02, min(20.0, abs(float(v)))) for v in scale[:3]],
                        "rotation_deg": [max(-360.0, min(360.0, float(v))) for v in rotation[:3]],
                    }
                )
            except (TypeError, ValueError):
                continue

    # A planner that collapses several unrelated boolean operations onto the
    # same point is emitting placeholder geometry. Reject it rather than
    # silently guessing what the cutters were supposed to mean.
    if len(cutters) >= 3:
        cluster_tolerance = max(axis_span, max_half_width * 2.0, height_span) * 0.025
        for index, first in enumerate(cutters):
            collapsed = [first]
            for second in cutters[index + 1:]:
                distance = sum(
                    (float(first["location"][axis]) - float(second["location"][axis])) ** 2
                    for axis in range(3)
                ) ** 0.5
                if distance <= cluster_tolerance:
                    collapsed.append(second)
            if len(collapsed) >= 3:
                names = ", ".join(item["name"] for item in collapsed[:4])
                raise ValueError(
                    "Hard-surface cage contains three or more cutters collapsed "
                    f"onto one location ({names}); re-plan with deliberate cutter placement."
                )

    # Exact duplicate cutter transforms are redundant and can damage topology.
    deduped_cutters: list[dict] = []
    seen_cutters: set[tuple] = set()
    for item in cutters:
        key = (
            item["shape"],
            *(round(float(value), 4) for value in item["location"]),
            *(round(float(value), 4) for value in item["scale"]),
            *(round(float(value), 3) for value in item["rotation_deg"]),
        )
        if key in seen_cutters:
            continue
        seen_cutters.add(key)
        deduped_cutters.append(item)
    normalized["cutters"] = deduped_cutters

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


def _clean_initial_primary_blockout(
    spec: HardSurfaceCageSpec,
    feature_task: FeatureTask | None,
    *,
    has_existing_cage: bool,
) -> tuple[HardSurfaceCageSpec, int, int]:
    """Keep the first base-mesh pass as a clean primary mass.

    Feature decomposition owns openings and separate parts. An initial
    base_mesh_region should establish silhouette/proportion before later
    surface_cutout/attachment/component passes modify that mass.
    """

    if (
        has_existing_cage
        or feature_task is None
        or feature_task.strategy != "base_mesh_region"
    ):
        return spec, 0, 0

    deferred_cutters = len(spec.cutters)
    deferred_attachments = len(spec.attachments)
    if deferred_cutters == 0 and deferred_attachments == 0:
        return spec, 0, 0

    cleaned = spec.model_copy(deep=True)
    cleaned.cutters = []
    cleaned.attachments = []
    return cleaned, deferred_cutters, deferred_attachments


def _active_hard_surface_cage_spec(
    root: Path,
    status_payload: dict,
) -> tuple[int, HardSurfaceCageSpec] | None:
    model = status_payload.get("generic_model")
    if not isinstance(model, dict):
        return None
    version = model.get("version")
    if not isinstance(version, int):
        return None
    path = root / f"cage-spec-v{version}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec_payload = payload.get("spec") if isinstance(payload, dict) else None
        return version, HardSurfaceCageSpec.model_validate(spec_payload)
    except (OSError, json.JSONDecodeError, ValidationError, TypeError):
        return None


async def _build_hard_surface_cage_spec(
    job_id: str,
    *,
    reason: str,
    feature_task: FeatureTask | None = None,
) -> HardSurfaceCageSpec:
    root = _require_job(job_id)
    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    inventory = _load_subject_inventory(root)
    feature_plan = await _ensure_feature_plan(job_id, inventory)
    feature_plan_context = feature_plan.model_dump() if feature_plan is not None else {}
    feature_context = feature_task.model_dump() if feature_task is not None else {}
    subject_complexity = inventory.complexity if inventory is not None else "moderate"
    if subject_complexity == "complex":
        min_cage_stations, min_profile_points = 8, 6
    elif subject_complexity == "moderate":
        min_cage_stations, min_profile_points = 6, 5
    else:
        min_cage_stations, min_profile_points = 4, 4

    status_payload = _read_status(root)
    active = _working_hard_surface_cage_spec(root, status_payload)
    current_cage = geometry_context(active[1].model_dump()) if active is not None else {}

    latest_vision: dict = {}
    vision_path = root / "vision-latest.json"
    if vision_path.exists():
        try:
            payload = json.loads(vision_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                latest_vision = payload.get("report") or {}
        except (OSError, json.JSONDecodeError):
            pass

    images, labels = _collect_images(
        root,
        VisionAnalyzeRequest(
            stage="hard_surface_cage_planning",
            include_references=True,
            include_renders=True,
            max_images=8 if feature_task is not None else 14,
            render_version=active[0] if active is not None else None,
        ),
    )
    system = (
        "You are directing a human-style Blender hard-surface modeling pass. Return JSON only matching "
        "HardSurfaceCageSpec. Do NOT invent an external/generated base mesh. Model the primary object the way "
        "a competent Blender artist would: establish a low-poly HALF CAGE from orthographic/reference evidence, "
        "mirror it across the centerline, then use subdivision/bevel and bounded boolean cutters for visible openings. "
        "The cage stations run along the main length axis. Every station uses the SAME 4-8 point half-profile ordered "
        "from bottom centerline -> outer/lower side -> shoulder/upper side -> top centerline. The first and last profile "
        "points are on the mirror plane (half_width=0). Add stations where the silhouette changes; do not waste stations "
        "on tiny cosmetic detail. Use cutters only for genuine holes/recesses/openings that materially affect silhouette. "
        "Use attachments only for separate IN-PLACE parts. Any feature whose build_mode=component_job belongs to an "
        "isolated child job and must NOT be faked as an attachment. Preserve accepted/best geometry. If a current cage "
        "is supplied, revise it rather than starting over unless its topology is fundamentally unsuitable. Front is "
        "negative Y and Z is up. Favor proportion, silhouette and major construction lines over surface decoration. "
        f"Geometry budget for this subject: emit AT LEAST {min_cage_stations} stations and "
        f"AT LEAST {min_profile_points} points per half-profile. "
        "Place sections at the actual silhouette transitions visible in the references. "
        "Start with subdivision_levels=0 when the outline has crisp corners; subdivision shrinks and rounds "
        "the cage, so use it only when the visible form calls for it. Use bevel sparingly. "
        "Every cutter must have a deliberate location, scale and orientation. "
        "Do not include openings or details owned by later features in this first silhouette pass. "
        "If ACTIVE FEATURE strategy is base_mesh_region and there is no CURRENT HARD-SURFACE CAGE, emit a CLEAN "
        "continuous primary mass: no boolean cutters and no separate attachments. Those belong to later feature owners. "
        "Do not copy a stock profile or invent a display pedestal. "
        "Choose all section positions, widths and heights from the actual reference proportions. "
        "AXIS CONTRACT: axis=y means station.position is WORLD Y and profile=[HALF X WIDTH, ABSOLUTE WORLD Z]. "
        "axis=x means position is WORLD X and profile=[HALF Y WIDTH, ABSOLUTE WORLD Z]. "
        "A station's position is NEVER its height. The top height must occur in profile[*][1]. "
        "The visual artist's XYZ dimension brief is an approximate estimate from perspective photos, not a "
        "measurement. Correct implausible aspect ratios using the subject and silhouette evidence. Declare your "
        "chosen primary cage bounds as intended_dimensions_xyz=[world X width, world Y length, world Z height]; "
        "your emitted station/profile coordinates must agree with YOUR declared dimensions. Attachments are excluded."
    )
    prompt = (
        f"Exact user request: {job_request.get('prompt', '')}\n"
        f"Reference scope: {'parent object; focus only on component '+str(job_request.get('component_name')) if job_request.get('component_job') else 'requested whole object'}\n"
        f"Intended use: {job_request.get('intended_use', '')}\n"
        f"Reason for this Blender strategy/pass: {reason}\n"
        f"Subject inventory: {json.dumps(inventory.model_dump() if inventory else {}, ensure_ascii=False)}\n"
        f"Feature plan: {json.dumps(feature_plan_context, ensure_ascii=False)}\n"
        f"ACTIVE FEATURE: {json.dumps(feature_context, ensure_ascii=False)}\n"
        f"CURRENT HARD-SURFACE CAGE (if any): {json.dumps(current_cage, ensure_ascii=False)}\n"
        f"Latest visual critique: {json.dumps(latest_vision, ensure_ascii=False)}\n"
        f"Images in order: {labels}\n"
        "Choose coordinates from the visible references. For a focused feature pass, change the minimum cage stations, "
        "cutters or attachments necessary to make that feature visibly correct while protecting unrelated good geometry."
    )

    primary_form = feature_task is None or feature_task.strategy == "base_mesh_region"
    brief = (await _reference_geometry_brief(root, request=job_request, feature=feature_context,
                                            images=images, labels=labels)) if primary_form else None
    if brief:
        prompt += "\nReference artist's construction brief: " + json.dumps(brief)
    client = OllamaProxyClient()
    errors: list[str] = []
    cage_schema = HardSurfaceCageSpec.model_json_schema()
    if brief:
        cage_schema["properties"]["intended_dimensions_xyz"] = {
            "type": "array", "items": {"type": "number", "exclusiveMinimum": 0},
            "minItems": 3, "maxItems": 3,
            "description": "Planner's intended primary cage extent in world X, Y, Z; excludes attachments.",
        }
        cage_schema["required"].append("intended_dimensions_xyz")
    for candidate_model in ((REASONING_MODEL, *VISION_MODELS) if brief else VISION_MODELS):
        try:
            result = await client.chat_json(
                model=candidate_model,
                system=system,
                prompt=prompt + ("\nFix these previous construction errors: " + " | ".join(errors) if errors else ""),
                images=(images or None) if candidate_model in VISION_MODELS else None,
                schema=cage_schema,
                temperature=0.0,
                num_predict=8192,
            )
            _write_llm_log(
                root,
                "hard-surface-cage-raw",
                {
                    "job_id": job_id,
                    "model": candidate_model,
                    "endpoint": result.endpoint,
                    "usage": result.usage,
                    "images": labels,
                    "reason": reason,
                    "raw": result.data,
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
            normalized = _normalize_hard_surface_cage_payload(
                result.data,
                str(job_request.get("prompt") or "Hard-surface cage"),
                complexity=subject_complexity,
            )
            spec = HardSurfaceCageSpec.model_validate(normalized)
            if brief:
                if spec.intended_dimensions_xyz is None:
                    raise ValueError("Declare intended_dimensions_xyz for the primary cage.")
                _validate_cage_dimensions(spec, spec.intended_dimensions_xyz)
            spec, deferred_cutters, deferred_attachments = _clean_initial_primary_blockout(
                spec,
                feature_task,
                has_existing_cage=active is not None,
            )
            if job_request.get("component_job") is True:
                spec.presentation_base = False
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            errors.append(f"{candidate_model}: {exc}")
            continue

        if deferred_cutters or deferred_attachments:
            append_history(
                root,
                "primary_blockout_details_deferred",
                feature_id=feature_task.id if feature_task is not None else None,
                deferred_cutters=deferred_cutters,
                deferred_attachments=deferred_attachments,
                reason=(
                    "Initial base-mesh pass owns primary mass only; openings and separate "
                    "parts remain assigned to later feature passes."
                ),
            )

        _write_llm_log(
            root,
            "hard-surface-cage-spec",
            {
                "job_id": job_id,
                "model": candidate_model,
                "endpoint": result.endpoint,
                "usage": result.usage,
                "images": labels,
                "reason": reason,
                "spec": spec.model_dump(),
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        append_history(
            root,
            "hard_surface_cage_planned",
            model=candidate_model,
            axis=spec.axis,
            stations=len(spec.stations),
            cutters=len(spec.cutters),
            attachments=len(spec.attachments),
            reason=reason,
        )
        return spec

    raise HTTPException(
        status_code=502,
        detail="Hard-surface cage planning failed across configured vision models: "
        + " | ".join(errors[-4:]),
    )


async def _execute_hard_surface_cage(
    job_id: str,
    spec: HardSurfaceCageSpec,
    *,
    version: int,
    activate_status: bool = True,
) -> dict:
    root = _require_job(job_id)
    require_unused_version(root, version)
    prefix = f"model-v{version}"
    blend_path = root / "scene" / f"{prefix}.blend"
    qa_path = root / "exports" / f"{prefix}-qa.json"

    _write_status(
        root,
        state="running",
        stage=f"hard_surface_cage_build_v{version}",
        modeling_strategy="hard_surface_cage",
    )
    payload = {
        "tool": "blender_python_exec",
        "arguments": {
            "code": hard_surface_cage_script(),
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
        _write_status(root, state="failed", stage=f"hard_surface_cage_build_v{version}", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Hard-surface cage Blender build failed: {exc}") from exc

    blender_error = _worker_blender_error(result)
    if blender_error:
        _write_status(root, state="failed", stage=f"hard_surface_cage_build_v{version}", error=blender_error)
        raise HTTPException(status_code=502, detail=f"Hard-surface cage Blender script failed: {blender_error}")

    views = (
        "front", "front-left", "left", "back-left", "back",
        "back-right", "right", "front-right", "top",
    )
    expected = [f"{prefix}-{view}.png" for view in views]
    missing = [name for name in expected if not (root / "renders" / name).is_file()]
    if missing or not blend_path.is_file():
        raise HTTPException(
            status_code=502,
            detail=f"Hard-surface cage build completed but artifacts are missing: {missing}",
        )

    (root / f"cage-spec-v{version}.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "version": version,
                "strategy": "hard_surface_cage",
                "spec": spec.model_dump(),
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    append_history(
        root,
        "hard_surface_cage_generation",
        version=version,
        title=spec.title,
        stations=len(spec.stations),
        cutters=len(spec.cutters),
        attachments=len(spec.attachments),
        renders=expected,
        blend=blend_path.name,
        qa=qa_path.name,
    )
    candidate_model = {
        "version": version,
        "title": spec.title,
        "blend": blend_path.name,
        "renders": expected,
        "qa": qa_path.name,
        "strategy": "hard_surface_cage",
    }
    status_values: dict[str, object] = {
        "state": "ready",
        "stage": f"hard_surface_cage_rendered_v{version}",
        "modeling_strategy": "hard_surface_cage",
    }
    if activate_status:
        status_values["generic_model"] = candidate_model
    status = _write_status(root, **status_values)
    return {
        "job_id": job_id,
        "status": status,
        "candidate_model": candidate_model,
        "strategy": "hard_surface_cage",
        "spec": spec.model_dump(),
        "renders": expected,
        "worker_result": result,
    }


def _working_hard_surface_cage_spec(
    root: Path,
    status_payload: dict,
) -> tuple[int, HardSurfaceCageSpec] | None:
    """Return the editable cage track without confusing it with the best active model."""

    candidates: list[int] = []
    working_version = status_payload.get("working_cage_version")
    if isinstance(working_version, int):
        candidates.append(working_version)

    active = _active_hard_surface_cage_spec(root, status_payload)
    if active is not None and active[0] not in candidates:
        candidates.append(active[0])

    for path in sorted(
        [] if candidates else root.glob("cage-spec-v*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        match = re.search(r"cage-spec-v(\d+)\.json$", path.name)
        if match:
            version = int(match.group(1))
            if version not in candidates:
                candidates.append(version)

    for version in candidates:
        path = root / f"cage-spec-v{version}.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            spec_payload = payload.get("spec") if isinstance(payload, dict) else None
            return version, HardSurfaceCageSpec.model_validate(spec_payload)
        except (OSError, json.JSONDecodeError, ValidationError, TypeError):
            continue
    return None


def _recent_cage_edit_events(root: Path, limit: int = 6) -> list[dict]:
    events = [
        item
        for item in load_history(root)
        if item.get("event") in {
            "cage_edit_decision",
            "cage_edit_kept",
            "cage_edit_rejected",
        }
    ]
    return events[-limit:]


def _cage_edit_action_schema(
    *,
    allow_replan: bool,
    blocked_operations: set[str] | None = None,
) -> dict:
    """Expose only edit operations that are valid for the current recovery state."""

    schema = CageEditAction.model_json_schema()
    operation = schema.get("properties", {}).get("operation")
    if isinstance(operation, dict):
        values = operation.get("enum")
        if isinstance(values, list):
            blocked = set(blocked_operations or ())
            if not allow_replan:
                blocked.add("replan_representation")
            operation["enum"] = [value for value in values if value not in blocked]
    return schema


def _normalize_cage_edit_action_payload(data: object) -> dict:
    """Normalize harmless LLM field-name drift without inventing edit semantics."""

    if not isinstance(data, dict):
        raise TypeError("Visual cage edit decision is not a JSON object.")

    normalized = dict(data)
    if normalized.get("operation") is None and normalized.get("action") is not None:
        normalized["operation"] = normalized.get("action")
    if normalized.get("target_index") is None:
        for key in ("station_index", "cutter_index", "target_station", "target_cutter"):
            if normalized.get(key) is not None:
                normalized["target_index"] = normalized.get(key)
                break
    if normalized.get("point_index") is None:
        for key in ("profile_point_index", "target_point", "profile_index"):
            if normalized.get(key) is not None:
                normalized["point_index"] = normalized.get(key)
                break
    if not normalized.get("reason"):
        normalized["reason"] = str(
            normalized.get("summary")
            or normalized.get("diagnosis")
            or normalized.get("rationale")
            or "Visual edit director selected this bounded action."
        )
    if not normalized.get("expected_visual_effect"):
        normalized["expected_visual_effect"] = str(
            normalized.get("expected_effect")
            or normalized.get("expected_improvement")
            or normalized.get("goal")
            or ""
        )

    # The model owns edit intent; deterministic code owns safety bounds.
    # Preserve the proposed direction/magnitude as closely as possible while
    # clamping harmless numeric overshoot instead of wasting a vision round.
    scalar_bounds = {
        "length_scale": (0.75, 1.30),
        "width_scale": (0.65, 1.45),
        "height_scale": (0.65, 1.45),
        "height_offset_fraction": (-0.30, 0.30),
        "position_offset_fraction": (-0.20, 0.20),
        "width_offset_fraction": (-0.30, 0.30),
        "point_height_offset_fraction": (-0.30, 0.30),
        "insert_fraction": (0.15, 0.85),
    }
    for field, (lower, upper) in scalar_bounds.items():
        if normalized.get(field) is None:
            continue
        try:
            value = float(normalized[field])
        except (TypeError, ValueError):
            continue
        normalized[field] = max(lower, min(upper, value))

    if normalized.get("influence_radius") is not None:
        try:
            radius = round(float(normalized["influence_radius"]))
        except (TypeError, ValueError):
            radius = 1
        normalized["influence_radius"] = max(1, min(3, radius))

    return normalized


async def _decide_hard_surface_cage_edit(
    job_id: str,
    *,
    baseline_version: int,
    spec: HardSurfaceCageSpec,
    feature_task: FeatureTask | None,
    allow_replan: bool,
    blocked_operations: set[str] | None = None,
) -> CageEditAction:
    root = _require_job(job_id)
    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    status_payload = _read_status(root)
    quality_gate = (
        status_payload.get("quality_gate")
        if isinstance(status_payload.get("quality_gate"), dict)
        else {}
    )

    views = (
        _feature_diagnostic_views(feature_task)
        if feature_task is not None
        else ("front", "front-left", "left", "back", "right", "front-right", "top")
    )

    reference_paths: list[Path] = []
    for record in _usable_reference_index(root):
        stored_name = str(record.get("stored_name") or "")
        if not stored_name:
            continue
        path = root / "references" / Path(stored_name).name
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            reference_paths.append(path)
    reference_paths = sorted(reference_paths, key=lambda path: path.stat().st_mtime)[-4:]

    baseline_paths = [
        root / "renders" / f"model-v{baseline_version}-{view}.png"
        for view in views
    ]
    if not reference_paths:
        raise HTTPException(
            status_code=409,
            detail="Iterative visual editing requires at least one verified reference image.",
        )
    if not all(path.is_file() for path in baseline_paths):
        raise HTTPException(
            status_code=409,
            detail="Iterative visual editing requires complete baseline renders.",
        )

    image_paths = reference_paths + baseline_paths
    images = _encode_vision_images(image_paths)
    labels = [f"{path.parent.name}/{path.name}" for path in image_paths]

    station_summary = [
        {
            "index": index,
            "position": station.position,
            "profile": station.profile,
        }
        for index, station in enumerate(spec.stations)
    ]
    station_positions = [float(station.position) for station in spec.stations]
    position_min = min(station_positions)
    position_span = max(station_positions) - position_min
    station_envelope = []
    for index, station in enumerate(spec.stations):
        profile = station.profile
        station_envelope.append(
            {
                "index": index,
                "position_fraction": (
                    (float(station.position) - position_min) / position_span
                    if position_span > 0.0
                    else 0.0
                ),
                "half_width": max(float(point[0]) for point in profile),
                "bottom": min(float(point[1]) for point in profile),
                "top": max(float(point[1]) for point in profile),
            }
        )
    cutter_summary = [
        {
            "index": index,
            "name": cutter.name,
            "shape": cutter.shape,
            "location": cutter.location,
            "scale": cutter.scale,
            "rotation_deg": cutter.rotation_deg,
        }
        for index, cutter in enumerate(spec.cutters)
    ]
    max_half_width = max(float(item["half_width"]) for item in station_envelope)
    min_bottom = min(float(item["bottom"]) for item in station_envelope)
    max_top = max(float(item["top"]) for item in station_envelope)
    current_axis_span = position_span
    current_dimensions_xyz = (
        [current_axis_span, max_half_width * 2.0, max_top - min_bottom]
        if spec.axis == "x"
        else [max_half_width * 2.0, current_axis_span, max_top - min_bottom]
    )
    declared_dimensions_xyz = (
        [float(value) for value in spec.intended_dimensions_xyz]
        if spec.intended_dimensions_xyz is not None
        else None
    )
    envelope_relative_error = None
    envelope_matches_declared = False
    if declared_dimensions_xyz and all(value > 0.0 for value in declared_dimensions_xyz):
        envelope_relative_error = [
            abs(current - declared) / declared
            for current, declared in zip(current_dimensions_xyz, declared_dimensions_xyz, strict=True)
        ]
        envelope_matches_declared = max(envelope_relative_error) <= 0.12

    feature_context = feature_task.model_dump() if feature_task is not None else {}
    primary_form_pass = (
        quality_gate.get("recognizable") is not True
        or (
            feature_task is not None
            and str(feature_task.strategy or "") == "base_mesh_region"
        )
    )
    modeling_pass = "primary_form" if primary_form_pass else "secondary_form"
    recent_events = _recent_cage_edit_events(root)
    blocked_operations = set(blocked_operations or ())

    system = (
        "You are the visual edit director for an iterative Blender modeling agent. "
        "You are NOT generating a replacement mesh. Inspect the verified references and the CURRENT baseline renders, "
        "identify the single highest-impact visible geometric error, and choose exactly ONE bounded edit from the "
        "CageEditAction schema. The editing vocabulary is generic and index-based. Prefer the smallest edit likely to "
        "make a visible improvement. Use relative factors/fractions instead of inventing absolute coordinates. "
        "Work like a production 3D modeler: establish the blockout, overall proportions, and readable silhouette "
        "before spending edits on secondary forms or detail. Compare the side/front/top envelopes and major transitions. "
        "reshape_cage_proportions changes the whole length/width/height envelope and is ONLY appropriate when the "
        "global envelope itself is wrong. If the current computed XYZ envelope already agrees with the declared "
        "reference-driven target within about 12%, do not use whole-object scaling to manufacture local landmarks; "
        "use reshape_station_region, reshape_station, reshape_profile_point, or insert_station to shape the hood, "
        "roof peak, transitions, taper, or other regional silhouette structure instead. "
        "reshape_station_region behaves like bounded proportional editing around one station with smooth falloff into "
        "neighboring connected sections; "
        "reshape_station changes one entire cross-section; reshape_profile_point changes one local profile point; "
        "insert_station adds control where silhouette curvature is under-resolved; remove_station removes a harmful "
        "intermediate section; adjust_cutter changes one existing opening/cut; remove_cutter removes a clearly harmful "
        "cut. add_attachment adds a missing in-place primitive or shaped mesh panel; adjust_attachment or remove_attachment "
        "edits an existing part; add_cutter creates a deliberate opening. set_surface changes subdivision, "
        "bevel or shading when modifiers have rounded a crisp intended silhouette. "
        "Primitive transforms use full dimensions (preferred) or legacy local scales, then Euler XYZ rotation. "
        "Standard cubes/cylinders/spheres have native full size 2, so scale is half the final dimensions. "
        "Lathe and sweep describe curved parts directly; their profile/path coordinates are local. Locations use world X/Y/Z. "
        "Cylinders extend along local Z: rotation around Z does NOT change their axial direction. "
        "Protect unrelated good geometry. Do not repeat a recent rejected edit with effectively the same "
        "target and parameters. "
        + (
            "The following operations are temporarily blocked because that edit family already failed to improve "
            f"the preserved working model: {sorted(blocked_operations)}. Choose a different bounded operation. "
            if blocked_operations
            else ""
        )
        + (
            (
                "PRIMARY-FORM PASS: the subject is not yet visually established. Prioritize, in order: "
                "reshape_cage_proportions for a wrong global envelope; reshape_station_region for a broad silhouette "
                "or transition error spanning adjacent sections; reshape_station or insert_station for a local "
                "silhouette control problem. Avoid cutter/detail edits unless a large existing cut is itself the "
                "dominant silhouette error. Do not polish details on an unrecognizable blockout. "
            )
            if primary_form_pass
            else
            (
                "SECONDARY-FORM PASS: the broad subject already reads. Preserve the accepted silhouette while "
                "refining local profiles, openings, transitions, and attachments needed by the active feature. "
            )
        )
        + (
            "Representation replanning is now allowed because multiple bounded edits have already failed. "
            "Choose replan_representation only when the current cage topology truly cannot plausibly converge. "
            if allow_replan
            else
            "Representation replanning is LOCKED for this pass because the current editable representation has not "
            "yet had enough rejected bounded edits. You MUST choose one bounded station/profile/cutter edit; do not "
            "return replan_representation. "
        )
        + "Return JSON only matching the schema."
    )
    prompt = (
        f"Exact user request: {job_request.get('prompt', '')}\n"
        f"Reference scope: {'parent object; focus only on component '+str(job_request.get('component_name')) if job_request.get('component_job') else 'requested whole object'}\n"
        f"Intended use: {job_request.get('intended_use', '')}\n"
        f"Working baseline version: {baseline_version}\n"
        f"Coordinate frame: station position is world {spec.axis.upper()}; profile points are "
        f"[half-width on {'X' if spec.axis == 'y' else 'Y'}, world Z height]. "
        "World front is -Y, left is -X and up is +Z. Part locations/deltas are world XYZ; "
        "part scales are local XYZ before Euler rotation.\n"
        f"ACTIVE FEATURE (if any): {json.dumps(feature_context, ensure_ascii=False)}\n"
        f"Current quality critique: {json.dumps(quality_gate, ensure_ascii=False)}\n"
        f"MODELING PASS: {modeling_pass}\n"
        f"Declared reference-driven target dimensions XYZ: {json.dumps(declared_dimensions_xyz)}\n"
        f"Current cage dimensions XYZ computed from stations: {json.dumps(current_dimensions_xyz)}\n"
        f"Relative envelope error versus declared target: {json.dumps(envelope_relative_error)}; "
        f"envelope_matches_declared_within_12pct={envelope_matches_declared}\n"
        "If the envelope already matches the declared target, solve visible silhouette/transition problems locally "
        "instead of rescaling the entire object.\n"
        f"Station envelope (normalized longitudinal position, half-width, bottom, top): "
        f"{json.dumps(station_envelope, ensure_ascii=False)}\n"
        + f"Full station profile geometry: {json.dumps(station_summary, ensure_ascii=False)}\n"
        + f"Cutter indices and geometry: {json.dumps(cutter_summary, ensure_ascii=False)}\n"
        f"Attachments by index: {json.dumps(geometry_context([p.model_dump() for p in spec.attachments]))}\n"
        f"Current surface: subdivision={spec.subdivision_levels}, bevel={spec.bevel_width}, smooth={spec.smooth}\n"
        f"Recent edit outcomes: {json.dumps(recent_events, ensure_ascii=False)}\n"
        f"Images in order: {labels}\n"
        "Reference images come first, followed by CURRENT baseline renders. Choose one change only. "
        "State the visible error in reason and the expected improvement in expected_visual_effect."
    )

    client = OllamaProxyClient()
    errors: list[str] = []
    edit_brief = None
    for visual_model in VISION_MODELS:
        try:
            observed = await client.chat_json(
                model=visual_model,
                system=("Compare reference pixels and the CURRENT model as a 3D artist. Identify the single "
                        "largest visible shape error within the active feature, explain the geometric correction "
                        "and what must be preserved. Do not emit coordinates or an edit operation. A separate "
                        "geometry engineer will translate your observation using the actual mesh data. Return JSON."),
                prompt=prompt, images=images, schema=GeometryEditBrief.model_json_schema(),
                temperature=0.0, num_predict=2048,
            )
            edit_brief = GeometryEditBrief.model_validate(observed.data).model_dump()
            _write_llm_log(root, "geometry-edit-brief", {"model": visual_model, "images": labels, **edit_brief})
            break
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            errors.append(f"{visual_model}: {exc}")
    if edit_brief:
        prompt += "\nVisual artist's diagnosis: " + json.dumps(edit_brief)
    for candidate_model in ((REASONING_MODEL, *VISION_MODELS) if edit_brief else VISION_MODELS):
        try:
            result = await client.chat_json(
                model=candidate_model,
                system=system,
                prompt=prompt,
                images=images if candidate_model in VISION_MODELS else None,
                schema=_cage_edit_action_schema(
                    allow_replan=allow_replan,
                    blocked_operations=blocked_operations,
                ),
                temperature=0.0,
                num_predict=2048,
            )
            action = CageEditAction.model_validate(
                _normalize_cage_edit_action_payload(result.data)
            )
            if action.operation == "replan_representation" and not allow_replan:
                raise ValueError(
                    "Representation replan is locked until bounded cage edits have failed."
                )
            if action.operation in blocked_operations:
                raise ValueError(
                    f"{action.operation} is temporarily blocked after repeated rejected edits; "
                    "choose a different bounded edit family."
                )
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            errors.append(f"{candidate_model}: {exc}")
            continue

        payload = {
            "job_id": job_id,
            "model": candidate_model,
            "baseline_version": baseline_version,
            "images": labels,
            "feature": feature_context,
            "modeling_pass": modeling_pass,
            "action": action.model_dump(),
            "created_at": datetime.now(UTC).isoformat(),
        }
        _write_llm_log(root, "cage-edit-decision", payload)
        append_history(
            root,
            "cage_edit_decision",
            baseline_version=baseline_version,
            feature_id=feature_task.id if feature_task is not None else None,
            operation=action.operation,
            modeling_pass=modeling_pass,
            target_index=action.target_index,
            point_index=action.point_index,
            influence_radius=action.influence_radius,
            reason=action.reason,
            expected_visual_effect=action.expected_visual_effect,
            model=candidate_model,
            parameters=action.model_dump(),
        )
        return action

    raise HTTPException(
        status_code=502,
        detail="Visual cage edit decision failed across configured vision models: "
        + " | ".join(errors[-4:]),
    )


def _retain_working_cage_progress(root: Path, previous: dict, version: int,
                                  feature: FeatureTask | None, evaluation: dict | None,
                                  comparison: dict) -> dict:
    """Advance an improving construction track without replacing a better displayed model."""
    status = _write_status(
        root, state="ready", stage="hard_surface_cage_working_progress",
        modeling_strategy="hard_surface_cage", generic_model=previous.get("generic_model"),
        quality_gate=previous.get("quality_gate"), working_cage_version=version, cage_edit_stall_count=0,
        working_cage_evaluation={"evaluated_version": version, "feature": evaluation, "comparison": comparison},
    )
    if feature is not None:
        record_feature_progress(root, feature.id, version=version,
                                summary=str(comparison.get("summary") or "Working geometry improved."))
    append_history(root, "cage_working_progress", version=version,
                   active_version=(previous.get("generic_model") or {}).get("version"),
                   summary=comparison.get("summary"))
    return status


async def _refine_hard_surface_cage_incrementally(
    job_id: str,
    *,
    feature_task: FeatureTask | None,
) -> dict:
    """Run one observe -> edit -> render -> compare -> keep/revert iteration."""

    root = _require_job(job_id)
    previous_status = _read_status(root)
    working = _working_hard_surface_cage_spec(root, previous_status)
    if working is None:
        return await _generate_hard_surface_cage(
            job_id,
            reason=(
                "No editable hard-surface cage baseline exists yet. Create the initial "
                "reference-driven cage before iterative visual editing."
            ),
            feature_task=feature_task,
        )

    baseline_version, baseline_spec = working
    stall_count = int(previous_status.get("cage_edit_stall_count") or 0)
    if stall_count >= 4:
        append_history(
            root,
            "cage_edit_escalation",
            working_version=baseline_version,
            stall_count=stall_count,
            reason=(
                "Four bounded visual edits failed to improve the working cage; "
                "switching to a different editable mesh representation."
            ),
        )
        return await _generate_adaptive_mesh_fallback(
            job_id,
            reason=(
                "The current hard-surface cage stalled for four consecutive candidates. "
                "Do not regenerate another cage with the same representation. Build a different "
                "reference-driven adaptive loft candidate, compare it against the preserved "
                "best cage, and keep it only if the pixels clearly improve."
            ),
            feature_task=feature_task,
        )

    recent_rejections = [
        item for item in load_history(root)
        if item.get("event") == "cage_edit_rejected"
    ][-2:]
    repeated_global_rescale = (
        len(recent_rejections) == 2
        and all(item.get("operation") == "reshape_cage_proportions" for item in recent_rejections)
    )
    blocked_operations = {"reshape_cage_proportions"} if repeated_global_rescale else set()
    # Two failures from the same global edit family are not evidence that the
    # representation itself is wrong. Force a different bounded edit family
    # before representation replanning becomes eligible.
    allow_replan = stall_count >= 2 and not repeated_global_rescale
    action = await _decide_hard_surface_cage_edit(
        job_id,
        baseline_version=baseline_version,
        spec=baseline_spec,
        feature_task=feature_task,
        allow_replan=allow_replan,
        blocked_operations=blocked_operations,
    )
    if action.operation == "replan_representation":
        append_history(
            root,
            "cage_edit_escalation",
            working_version=baseline_version,
            stall_count=stall_count,
            reason=action.reason,
            next_strategy="adaptive_loft",
        )
        return await _generate_adaptive_mesh_fallback(
            job_id,
            reason=(
                "The visual edit director determined that the current cage representation "
                "cannot plausibly converge with bounded edits. Do not generate another cage. "
                "Try the different adaptive-loft representation while preserving the current "
                "best model until the candidate wins visual comparison. "
                + action.reason
            ),
            feature_task=feature_task,
        )

    try:
        edited_payload = apply_cage_edit_action(
            baseline_spec.model_dump(),
            action,
        )
        candidate_spec = HardSurfaceCageSpec.model_validate(edited_payload)
    except (ValidationError, ValueError, TypeError) as exc:
        append_history(
            root,
            "cage_edit_rejected",
            baseline_version=baseline_version,
            candidate_version=None,
            operation=action.operation,
            reason=f"Edit could not be applied safely: {exc}",
        )
        status = _write_status(
            root,
            state="ready",
            stage="hard_surface_cage_needs_refinement",
            modeling_strategy="hard_surface_cage",
            working_cage_version=baseline_version,
            cage_edit_stall_count=stall_count + 1,
        )
        return {
            "job_id": job_id,
            "strategy": "hard_surface_cage",
            "baseline_version": baseline_version,
            "candidate_version": None,
            "kept": False,
            "action": action.model_dump(),
            "comparison": None,
            "status": status,
        }

    version = reserve_model_version(root)
    build = await _execute_hard_surface_cage(
        job_id,
        candidate_spec,
        version=version,
        activate_status=False,
    )
    comparison = await _compare_generic_versions(
        root,
        baseline_version=baseline_version,
        candidate_version=version,
    )
    improved = comparison.get("candidate_is_better") is True

    feature_evaluation: dict | None = None
    feature_complete = False
    recognizability: dict | None = None
    if feature_task is not None:
        feature_evaluation = await _evaluate_feature_candidate(
            job_id,
            feature_task,
            baseline_version=baseline_version,
            candidate_version=version,
        )
        feature_complete = _feature_evaluation_accepts(feature_task, feature_evaluation)
    elif improved:
        try:
            recognizability = await _generic_recognizability_check(
                job_id,
                stage="iterative_cage_quality",
                render_version=version,
            )
        except HTTPException as exc:
            recognizability = {
                "recognizable": None,
                "subject_match_score": None,
                "summary": str(exc.detail),
                "director_action": "refine_mesh",
            }

    active_model = (
        dict(previous_status.get("generic_model"))
        if isinstance(previous_status.get("generic_model"), dict)
        else None
    )
    active_version = (
        active_model.get("version")
        if isinstance(active_model, dict) and isinstance(active_model.get("version"), int)
        else None
    )

    better_than_active = False
    if improved:
        if active_version is None or active_version == baseline_version:
            better_than_active = True
        elif active_version != version:
            active_comparison = await _compare_generic_versions(
                root,
                baseline_version=active_version,
                candidate_version=version,
            )
            better_than_active = active_comparison.get("candidate_is_better") is True

    if improved and not better_than_active:
        status = _retain_working_cage_progress(root, previous_status, version, feature_task,
                                               feature_evaluation, comparison)
        return {**build, "status": status, "baseline_version": baseline_version,
                "candidate_version": version, "kept": True, "promoted": False,
                "action": action.model_dump(), "comparison": comparison,
                "feature_evaluation": feature_evaluation}

    if improved:
        candidate_model = build.get("candidate_model")
        if recognizability and recognizability.get("recognizable") is True:
            better_than_active = True
        next_active_model = candidate_model if better_than_active else active_model
        quality = dict(
            previous_status.get("quality_gate")
            if isinstance(previous_status.get("quality_gate"), dict)
            else {}
        )
        if feature_evaluation:
            quality["summary"] = feature_evaluation.get("summary") or comparison.get("summary")
            quality["scope"] = "feature"
            quality["active_feature_id"] = feature_task.id
            quality["active_feature_passed"] = feature_complete
            if feature_evaluation.get("subject_recognizable") is not None:
                quality["recognizable"] = bool(feature_evaluation.get("subject_recognizable"))
            if feature_evaluation.get("reference_match_score") is not None:
                quality["subject_match_score"] = feature_evaluation.get("reference_match_score")
        elif recognizability:
            quality.update(recognizability)
            quality["summary"] = recognizability.get("summary") or comparison.get("summary")
        else:
            quality["summary"] = comparison.get("summary") or quality.get("summary")
        quality["representation"] = "hard_surface_cage"
        quality["working_cage_version"] = version
        quality["candidate_improved"] = True

        status = _write_status(
            root,
            state="ready",
            stage=(
                "hard_surface_cage_recognizable"
                if recognizability and recognizability.get("recognizable") is True
                else "hard_surface_cage_feature_complete"
                if feature_complete
                else "hard_surface_cage_needs_refinement"
            ),
            modeling_strategy="hard_surface_cage",
            generic_model=next_active_model,
            working_cage_version=version,
            cage_edit_stall_count=0,
            quality_gate=quality,
        )
        append_history(
            root,
            "cage_edit_kept",
            baseline_version=baseline_version,
            candidate_version=version,
            active_version=(
                next_active_model.get("version")
                if isinstance(next_active_model, dict)
                else None
            ),
            operation=action.operation,
            target_index=action.target_index,
            point_index=action.point_index,
            feature_id=feature_task.id if feature_task is not None else None,
            feature_complete=feature_complete,
            summary=comparison.get("summary"),
        )

        if feature_task is not None:
            if feature_complete:
                finish_feature(
                    root,
                    feature_task.id,
                    accepted=True,
                    version=version,
                    summary=str(feature_evaluation.get("summary") or ""),
                    verified=True,
                    acceptance_score=float(
                        feature_evaluation.get("reference_match_score") or 0.0
                    ),
                    acceptance_model=(
                        str(feature_evaluation.get("model"))
                        if feature_evaluation.get("model")
                        else None
                    ),
                )
            else:
                record_feature_progress(
                    root,
                    feature_task.id,
                    version=version,
                    summary=str(
                        (feature_evaluation or {}).get("summary")
                        or comparison.get("summary")
                        or action.expected_visual_effect
                    ),
                )
    else:
        status = _write_status(
            root,
            state="ready",
            stage="hard_surface_cage_needs_refinement",
            modeling_strategy="hard_surface_cage",
            generic_model=active_model,
            working_cage_version=baseline_version,
            cage_edit_stall_count=stall_count + 1,
            quality_gate={
                **(
                    previous_status.get("quality_gate")
                    if isinstance(previous_status.get("quality_gate"), dict)
                    else {}
                ),
                "representation": "hard_surface_cage",
                "working_cage_version": baseline_version,
                "candidate_improved": False,
                "last_rejected_candidate_version": version,
                "summary": comparison.get("summary"),
            },
        )
        append_history(
            root,
            "cage_edit_rejected",
            baseline_version=baseline_version,
            candidate_version=version,
            operation=action.operation,
            target_index=action.target_index,
            point_index=action.point_index,
            feature_id=feature_task.id if feature_task is not None else None,
            reason=comparison.get("summary"),
        )
        if feature_task is not None:
            record_feature_progress(
                root,
                feature_task.id,
                version=baseline_version,
                summary=(
                    "Candidate reverted; feature remains active. "
                    + str(comparison.get("summary") or "")
                ),
            )

    build["baseline_version"] = baseline_version
    build["candidate_version"] = version
    build["kept"] = improved
    build["action"] = action.model_dump()
    build["comparison"] = comparison
    build["feature_evaluation"] = feature_evaluation
    build["recognizability"] = recognizability
    build["status"] = status
    return build


async def _generate_hard_surface_cage(
    job_id: str,
    *,
    reason: str,
    feature_task: FeatureTask | None = None,
) -> dict:
    """Create/re-plan a cage while preserving the same visual keep/revert contract."""

    root = _require_job(job_id)
    previous_status = _read_status(root)
    previous_model = (
        dict(previous_status.get("generic_model"))
        if isinstance(previous_status.get("generic_model"), dict)
        else None
    )
    active_version = (
        previous_model.get("version")
        if isinstance(previous_model, dict) and isinstance(previous_model.get("version"), int)
        else None
    )

    feature_task = feature_task or begin_feature(root)
    working_before = _working_hard_surface_cage_spec(root, previous_status)
    baseline_version = working_before[0] if working_before is not None else active_version

    if feature_task is not None:
        append_history(
            root,
            "feature_subjob_started",
            feature_id=feature_task.id,
            feature_name=feature_task.name,
            attempt=feature_task.attempts,
            strategy=feature_task.strategy,
            dependencies=feature_task.depends_on,
            via="hard_surface_cage",
        )

    try:
        spec = await _build_hard_surface_cage_spec(
            job_id,
            reason=reason,
            feature_task=feature_task,
        )
        version = reserve_model_version(root)
        build = await _execute_hard_surface_cage(
            job_id,
            spec,
            version=version,
            activate_status=False,
        )
    except Exception as exc:
        if feature_task is not None:
            detail = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
            finish_feature(
                root,
                feature_task.id,
                accepted=False,
                version=None,
                error=detail,
            )
            append_history(
                root,
                "feature_subjob_failed",
                feature_id=feature_task.id,
                feature_name=feature_task.name,
                via="hard_surface_cage",
                error=detail,
            )
        raise

    feature_evaluation: dict | None = None
    comparison: dict | None = None

    if feature_task is not None:
        if baseline_version is not None and baseline_version != version:
            comparison = await _compare_generic_versions(
                root,
                baseline_version=baseline_version,
                candidate_version=version,
            )

        feature_evaluation = await _evaluate_feature_candidate(
            job_id,
            feature_task,
            baseline_version=baseline_version,
            candidate_version=version,
        )
        feature_complete = _feature_evaluation_accepts(feature_task, feature_evaluation)
        candidate_improved = bool(
            baseline_version is None
            or (comparison and comparison.get("candidate_is_better") is True)
        )

        working_improved = candidate_improved
        if candidate_improved and active_version is not None and active_version != baseline_version:
            active_comparison = await _compare_generic_versions(
                root, baseline_version=active_version, candidate_version=version,
            )
            candidate_improved = active_comparison.get("candidate_is_better") is True

        if working_improved and not candidate_improved:
            status = _retain_working_cage_progress(root, previous_status, version, feature_task,
                                                   feature_evaluation, comparison or {})
            return {**build, "status": status, "accepted": False, "kept": True, "promoted": False,
                    "comparison": comparison, "feature_evaluation": feature_evaluation}

        quality = dict(
            previous_status.get("quality_gate")
            if isinstance(previous_status.get("quality_gate"), dict)
            else {}
        )
        quality["summary"] = (
            feature_evaluation.get("summary")
            or (comparison or {}).get("summary")
            or quality.get("summary")
        )
        quality["scope"] = "feature"
        quality["active_feature_id"] = feature_task.id
        quality["active_feature_passed"] = feature_complete
        quality["representation"] = "hard_surface_cage"
        quality["baseline_version"] = baseline_version
        quality["candidate_version"] = version
        quality["candidate_improved"] = candidate_improved
        if feature_evaluation.get("subject_recognizable") is not None:
            quality["recognizable"] = bool(feature_evaluation.get("subject_recognizable"))
        if feature_evaluation.get("reference_match_score") is not None:
            quality["subject_match_score"] = feature_evaluation.get("reference_match_score")

        candidate_model = build.get("candidate_model")
        previous_stall_count = int(previous_status.get("cage_edit_stall_count") or 0)

        if feature_complete and candidate_improved:
            status = _write_status(
                root,
                state="ready",
                stage="hard_surface_cage_feature_complete",
                modeling_strategy="hard_surface_cage",
                generic_model=candidate_model,
                working_cage_version=version,
                cage_edit_stall_count=0,
                quality_gate=quality,
            )
            append_history(
                root,
                "hard_surface_cage_accepted",
                version=version,
                recognizable=feature_evaluation.get("subject_recognizable"),
                baseline_version=baseline_version,
                reason="strict feature acceptance passed",
            )
            finish_feature(
                root,
                feature_task.id,
                accepted=True,
                version=version,
                summary=str(feature_evaluation.get("summary") or ""),
                verified=True,
                acceptance_score=float(
                    feature_evaluation.get("reference_match_score") or 0.0
                ),
                acceptance_model=(
                    str(feature_evaluation.get("model"))
                    if feature_evaluation.get("model")
                    else None
                ),
            )
            append_history(
                root,
                "feature_subjob_accepted",
                feature_id=feature_task.id,
                feature_name=feature_task.name,
                version=version,
                via="hard_surface_cage",
            )

        elif candidate_improved:
            status = _write_status(
                root,
                state="ready",
                stage="hard_surface_cage_needs_refinement",
                modeling_strategy="hard_surface_cage",
                generic_model=candidate_model,
                working_cage_version=version,
                cage_edit_stall_count=0,
                quality_gate=quality,
            )
            append_history(
                root,
                "hard_surface_cage_progress",
                baseline_version=baseline_version,
                candidate_version=version,
                feature_id=feature_task.id,
                summary=quality.get("summary"),
            )
            record_feature_progress(
                root,
                feature_task.id,
                version=version,
                summary=str(
                    feature_evaluation.get("summary")
                    or (comparison or {}).get("summary")
                    or "Candidate improved the working representation but is not complete."
                ),
            )
            append_history(
                root,
                "feature_subjob_progress",
                feature_id=feature_task.id,
                feature_name=feature_task.name,
                version=version,
                via="hard_surface_cage",
            )

        else:
            kept_working_version = baseline_version
            status = _write_status(
                root,
                state="ready",
                stage="hard_surface_cage_needs_refinement",
                modeling_strategy="hard_surface_cage",
                generic_model=previous_model,
                working_cage_version=kept_working_version,
                cage_edit_stall_count=previous_stall_count + 1,
                quality_gate={
                    **_quality_snapshot(previous_status.get("quality_gate")),
                    "candidate_rejected": True,
                    "last_rejected_candidate_version": version,
                    "last_candidate_evaluation": _quality_snapshot(quality),
                },
            )
            append_history(
                root,
                "hard_surface_cage_rejected",
                baseline_version=baseline_version,
                candidate_version=version,
                feature_id=feature_task.id,
                summary=(
                    (comparison or {}).get("summary")
                    or feature_evaluation.get("summary")
                ),
            )
            record_feature_progress(
                root,
                feature_task.id,
                version=kept_working_version or version,
                summary=(
                    "Replanned candidate reverted; feature remains active. "
                    + str(
                        (comparison or {}).get("summary")
                        or feature_evaluation.get("summary")
                        or ""
                    )
                ),
            )

        accepted = feature_complete and candidate_improved

    else:
        if baseline_version is not None and baseline_version != version:
            comparison = await _compare_generic_versions(
                root,
                baseline_version=baseline_version,
                candidate_version=version,
            )
        try:
            quality = await _generic_recognizability_check(
                job_id,
                stage="hard_surface_cage_quality",
                render_version=version,
            )
        except HTTPException as exc:
            quality = {
                "recognizable": None,
                "subject_match_score": None,
                "summary": str(exc.detail),
                "director_action": "refine_mesh",
            }
        accepted = bool(
            baseline_version is None
            or (comparison and comparison.get("candidate_is_better"))
        )
        candidate_model = build.get("candidate_model")
        if accepted:
            status = _write_status(
                root,
                state="ready",
                stage=(
                    "hard_surface_cage_recognizable"
                    if quality.get("recognizable") is True
                    else "hard_surface_cage_needs_refinement"
                ),
                modeling_strategy="hard_surface_cage",
                generic_model=candidate_model,
                working_cage_version=version,
                cage_edit_stall_count=0,
                quality_gate={
                    **quality,
                    "representation": "hard_surface_cage",
                    "baseline_version": baseline_version,
                    "candidate_version": version,
                    "better_than_previous": (
                        comparison.get("candidate_is_better") if comparison else None
                    ),
                },
            )
            append_history(
                root,
                "hard_surface_cage_accepted",
                version=version,
                recognizable=quality.get("recognizable"),
                baseline_version=baseline_version,
            )
        else:
            status = _write_status(
                root,
                state="ready",
                stage="hard_surface_cage_needs_refinement",
                modeling_strategy="hard_surface_cage",
                generic_model=previous_model,
                working_cage_version=baseline_version,
                cage_edit_stall_count=int(previous_status.get("cage_edit_stall_count") or 0) + 1,
                quality_gate={
                    **_quality_snapshot(previous_status.get("quality_gate")),
                    "last_candidate_evaluation": _quality_snapshot(quality),
                    "representation": "hard_surface_cage",
                    "candidate_rejected": True,
                    "baseline_version": baseline_version,
                    "candidate_version": version,
                },
            )
            append_history(
                root,
                "hard_surface_cage_rejected",
                baseline_version=baseline_version,
                candidate_version=version,
                summary=(comparison or {}).get("summary"),
            )

    build["comparison"] = comparison
    build["feature_evaluation"] = feature_evaluation
    build["quality_gate"] = quality
    build["accepted"] = accepted
    build["status"] = status
    return build


async def _build_adaptive_loft_spec(
    job_id: str,
    *,
    reason: str,
    feature_task: FeatureTask | None = None,
) -> AdaptiveLoftSpec:
    root = _require_job(job_id)
    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    inventory = _load_subject_inventory(root)
    feature_plan = await _ensure_feature_plan(job_id, inventory)
    feature_plan_context = feature_plan.model_dump() if feature_plan is not None else {}
    feature_context = feature_task.model_dump() if feature_task is not None else {}

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
            max_images=8 if feature_task is not None else 14,
        ),
    )
    system = (
        "You are the mesh-reconstruction stage of an autonomous Blender system. Create a SAFE DECLARATIVE LOFT MESH "
        "using the selected construction strategy and reference evidence. "
        "Return JSON only matching the supplied AdaptiveLoftSpec schema. Choose the axis that best follows "
        "the subject's main length. Each cross section MUST contain exactly 8 perimeter points in consistent "
        "clockwise order when viewed along the positive axis; use the same semantic perimeter order in every "
        "section. Use 5-10 sections to capture major silhouette changes. Coordinates are Blender units and "
        "must remain roughly within -8..8. The loft is the main continuous body mass. Put visually separate "
        "identity-critical parts that are explicitly IN-PLACE features and should not be fused into the main silhouette "
        "into attachments using the safe primitive schema (for example handles, lenses, windows, feet, knobs, antennas). "
        "FeaturePlan entries with build_mode=component_job are owned by isolated child jobs: do NOT create those parts "
        "as attachments or placeholders here. Leave the supporting/mounting region suitable for later installation of "
        "the frozen child artifact. Do not approximate the whole subject with attachments; the loft must carry the "
        "primary silhouette. "
        "Front-facing subject direction is negative Y and Z is up. Favor reference geometry and recognizability "
        "over cosmetic detail. AXIS CONTRACT: axis=z means section.position is WORLD Z height and contour points "
        "are [world X, world Y]. axis=y means position is WORLD Y and contour=[X,Z]. axis=x means position is "
        "WORLD X and contour=[Y,Z]. Upright radial shapes usually need axis=z. Never confuse position with height "
        "on a horizontal axis."
    )
    prompt = (
        f"User request: {job_request.get('prompt', '')}\n"
        f"Intended use: {job_request.get('intended_use', '')}\n"
        f"Target width mm: {job_request.get('target_width_mm')}\n"
        f"Reason for strategy switch: {reason}\n"
        f"Subject inventory: {json.dumps(inventory.model_dump() if inventory else {}, ensure_ascii=False)}\n"
        f"Visible-feature sub-job plan: {json.dumps(feature_plan_context, ensure_ascii=False)}\n"
        f"ACTIVE FEATURE SUB-JOB: {json.dumps(feature_context, ensure_ascii=False)}\n"
        f"Latest visual critique: {json.dumps(latest_vision, ensure_ascii=False)}\n"
        f"Images in order: {labels}\n"
        "If an ACTIVE FEATURE SUB-JOB is supplied, make that feature the concrete goal of this pass while preserving "
        "geometry outside its ownership scope. Create a substantially more recognizable continuous base mesh. For an "
        "8-point cross section, a useful "
        "order is around the perimeter from lower-left -> mid-left -> upper-left/shoulder -> top-left -> "
        "top-right -> upper-right/shoulder -> mid-right -> lower-right, adjusted to the actual subject."
    )

    primary_form = feature_task is None or feature_task.strategy == "base_mesh_region"
    brief = (await _reference_geometry_brief(root, request=job_request, feature=feature_context,
                                            images=images, labels=labels)) if primary_form else None
    if brief:
        prompt += "\nReference artist's construction brief: " + json.dumps(brief)
    client = OllamaProxyClient()
    errors: list[str] = []
    for candidate_model in ((REASONING_MODEL, *VISION_MODELS) if brief else VISION_MODELS):
        try:
            result = await client.chat_json(
                model=candidate_model,
                system=system,
                prompt=prompt,
                images=(images or None) if candidate_model in VISION_MODELS else None,
                schema=AdaptiveLoftSpec.model_json_schema(),
                temperature=0.0,
                num_predict=8192,
            )
            normalized = _normalize_adaptive_loft_payload(
                result.data,
                str(job_request.get("prompt") or "Adaptive mesh"),
            )
            spec = AdaptiveLoftSpec.model_validate(normalized)
            if job_request.get("component_job") is True:
                spec.presentation_base = False
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
    require_unused_version(root, version)
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


def _active_adaptive_loft_spec(
    root: Path,
    status_payload: dict,
) -> tuple[int, AdaptiveLoftSpec] | None:
    model = status_payload.get("generic_model")
    if not isinstance(model, dict):
        return None
    version = model.get("version")
    if not isinstance(version, int):
        return None
    path = root / f"mesh-spec-v{version}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec_payload = payload.get("spec") if isinstance(payload, dict) else None
        return version, AdaptiveLoftSpec.model_validate(spec_payload)
    except (OSError, json.JSONDecodeError, ValidationError, TypeError):
        return None


async def _revise_adaptive_loft_spec(
    job_id: str,
    current_spec: AdaptiveLoftSpec,
    decision: dict,
    feature_task: FeatureTask | None = None,
) -> AdaptiveLoftSpec:
    root = _require_job(job_id)
    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    inventory = _load_subject_inventory(root)
    feature_plan, planned_task = _feature_task_context(root)
    feature_task = feature_task or planned_task
    feature_context = feature_task.model_dump() if feature_task is not None else {}
    plan_context = feature_plan.model_dump() if feature_plan is not None else {}
    images, labels = _collect_images(
        root,
        VisionAnalyzeRequest(
            stage="adaptive_mesh_revision",
            include_references=True,
            include_renders=True,
            max_images=8 if feature_task is not None else 14,
        ),
    )

    system = (
        "You revise an EXISTING adaptive loft mesh for an autonomous Blender system. Keep the useful topology and "
        "coordinate frame of the current AdaptiveLoftSpec and improve it toward the reference images. Return the full "
        "replacement AdaptiveLoftSpec JSON. Prefer targeted edits to section positions/contours and attachments over "
        "starting from an unrelated shape. Preserve good geometry. You may add/remove/reposition cross-sections and "
        "attachments when the director critique requires it, but keep exactly 8 consistently ordered contour points "
        "per section. The loft must remain the main continuous body. Use attachments for visually separate parts. "
        "Front is negative Y and Z is up. A feature sub-job may be supplied. When present, treat it as the primary "
        "owner for this pass: make the requested feature visibly better while preserving unrelated accepted geometry. "
        "Respect its target_regions, owner_scope, dependencies and acceptance_criteria. Never add, recreate or reshape "
        "geometry owned by any FeaturePlan entry whose build_mode=component_job; those parts are frozen/installed by the "
        "component assembler, not by this mesh revision worker. Return JSON only matching the schema."
    )
    prompt = (
        f"Exact user request: {job_request.get('prompt', '')}\n"
        f"Reference scope: {'parent object; focus only on component '+str(job_request.get('component_name')) if job_request.get('component_job') else 'requested whole object'}\n"
        f"Current AdaptiveLoftSpec: {json.dumps(current_spec.model_dump(), ensure_ascii=False)}\n"
        f"Modeling director critique/instructions: {json.dumps(decision, ensure_ascii=False)}\n"
        f"Subject inventory: {json.dumps(inventory.model_dump() if inventory else {}, ensure_ascii=False)}\n"
        f"Full coordinated feature plan: {json.dumps(plan_context, ensure_ascii=False)}\n"
        f"ACTIVE FEATURE SUB-JOB: {json.dumps(feature_context, ensure_ascii=False)}\n"
        f"Images in order: {labels}\n"
        "Make the smallest set of meaningful geometric changes that clearly improves the active feature and overall "
        "resemblance. Do not discard a recognizable body just because it is imperfect. Do not rewrite geometry outside "
        "the active feature's owner scope unless a dependency relationship makes that adjustment necessary."
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
                str(job_request.get("prompt") or current_spec.title),
            )
            revised = AdaptiveLoftSpec.model_validate(normalized)
            if job_request.get("component_job") is True:
                revised.presentation_base = False
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            errors.append(f"{candidate_model}: {exc}")
            continue

        payload = {
            "job_id": job_id,
            "model": candidate_model,
            "endpoint": result.endpoint,
            "usage": result.usage,
            "images": labels,
            "director": decision,
            "previous_spec": current_spec.model_dump(),
            "spec": revised.model_dump(),
            "created_at": datetime.now(UTC).isoformat(),
        }
        _write_llm_log(root, "adaptive-loft-revision", payload)
        append_history(
            root,
            "adaptive_mesh_revision_planned",
            model=candidate_model,
            sections=len(revised.sections),
            attachments=len(revised.attachments),
            director_summary=decision.get("summary"),
        )
        return revised

    raise HTTPException(
        status_code=502,
        detail="Adaptive mesh revision failed across configured vision models: "
        + " | ".join(errors[-4:]),
    )


async def _refine_adaptive_mesh(
    job_id: str,
    *,
    decision: dict,
    feature_task: FeatureTask | None = None,
) -> dict:
    root = _require_job(job_id)
    status_path = root / "status.json"
    try:
        previous_status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous_status = {}

    active = _active_adaptive_loft_spec(root, previous_status)
    if active is None:
        return await _generate_adaptive_mesh_fallback(
            job_id,
            reason=(decision.get("summary") or "No reusable adaptive mesh spec was available.")
            + "\n"
            + "\n".join(decision.get("instructions") or []),
            feature_task=feature_task,
        )

    baseline_version, current_spec = active
    previous_model = dict(previous_status.get("generic_model") or {})
    feature_task = feature_task or begin_feature(root)
    if feature_task is not None and feature_task.attempts > 0:
        append_history(
            root,
            "feature_subjob_started",
            feature_id=feature_task.id,
            feature_name=feature_task.name,
            attempt=feature_task.attempts,
            strategy=feature_task.strategy,
            dependencies=feature_task.depends_on,
        )
    _write_status(
        root,
        state="running",
        stage="adaptive_mesh_refining",
        modeling_strategy="adaptive_loft",
    )
    try:
        revised = await _revise_adaptive_loft_spec(
            job_id,
            current_spec,
            decision,
            feature_task=feature_task,
        )
    except Exception as exc:
        if feature_task is not None:
            finish_feature(
                root,
                feature_task.id,
                accepted=False,
                version=None,
                error=str(exc),
            )
            append_history(
                root,
                "feature_subjob_failed",
                feature_id=feature_task.id,
                feature_name=feature_task.name,
                error=str(exc),
            )
        raise
    if revised.model_dump() == current_spec.model_dump():
        append_history(
            root,
            "adaptive_mesh_refinement_stop",
            version=baseline_version,
            reason="AI returned an unchanged adaptive mesh spec",
        )
        if feature_task is not None:
            finish_feature(
                root,
                feature_task.id,
                accepted=False,
                version=None,
                summary="The feature worker returned an unchanged mesh specification.",
            )
            append_history(
                root,
                "feature_subjob_retry",
                feature_id=feature_task.id,
                feature_name=feature_task.name,
                reason="unchanged mesh specification",
            )
        return {
            "job_id": job_id,
            "status": _write_status(
                root,
                state="ready",
                stage="adaptive_mesh_needs_refinement",
                modeling_strategy="adaptive_loft",
                generic_model=previous_model,
                quality_gate=previous_status.get("quality_gate"),
            ),
            "strategy": "adaptive_loft",
            "unchanged": True,
        }

    version = reserve_model_version(root)
    build = await _execute_adaptive_loft(job_id, revised, version=version)
    if feature_task is not None:
        feature_evaluation = await _evaluate_feature_candidate(
            job_id,
            feature_task,
            baseline_version=baseline_version,
            candidate_version=version,
        )
        feature_passed = _feature_evaluation_accepts(
            feature_task,
            feature_evaluation,
        )
        comparison = {
            "candidate_is_better": feature_passed,
            "summary": feature_evaluation.get("summary") or "",
            "improvements": [],
            "regressions": feature_evaluation.get("protected_geometry_notes") or [],
            "model": feature_evaluation.get("model"),
            "baseline_version": baseline_version,
            "candidate_version": version,
            "focused_feature_qa": True,
            "scope": "feature",
        }
        previous_quality = _quality_snapshot(previous_status.get("quality_gate"))
        quality = dict(previous_quality)
        quality.setdefault("recognizable", False)
        quality["summary"] = feature_evaluation.get("summary") or quality.get("summary")
        quality["scope"] = "feature"
        quality["active_feature_id"] = feature_task.id
        quality["active_feature_passed"] = feature_passed
        recognizable = quality.get("recognizable")
        better = feature_passed
        accept_candidate = feature_passed
    else:
        feature_evaluation = None
        comparison = await _compare_generic_versions(
            root,
            baseline_version=baseline_version,
            candidate_version=version,
        )
        try:
            quality = await _generic_recognizability_check(
                job_id,
                stage="adaptive_mesh_refinement_quality",
            )
        except HTTPException as exc:
            append_history(root, "adaptive_mesh_quality_unavailable", error=str(exc.detail))
            quality = {
                "recognizable": None,
                "subject_match_score": None,
                "recommended_strategy": "base_mesh",
                "summary": str(exc.detail),
                "director_action": "refine_mesh",
                "instructions": [],
            }
        recognizable = quality.get("recognizable")
        better = bool(comparison.get("candidate_is_better"))
        feature_passed = False
        accept_candidate = recognizable is True or better

    if accept_candidate:
        stage = "adaptive_mesh_recognizable" if recognizable is True else "adaptive_mesh_needs_refinement"
        active_model = build["status"].get("generic_model")
        status = _write_status(
            root,
            state="ready",
            stage=stage,
            modeling_strategy="adaptive_loft",
            generic_model=active_model,
            quality_gate={
                **quality,
                "better_than_previous": better,
                "baseline_version": baseline_version,
                "candidate_version": version,
            },
        )
        append_history(
            root,
            "adaptive_mesh_refinement_accepted",
            baseline_version=baseline_version,
            candidate_version=version,
            recognizable=recognizable,
            comparison_summary=comparison.get("summary"),
        )
        if feature_task is not None:
            finish_feature(
                root,
                feature_task.id,
                accepted=True,
                version=version,
                summary=str(
                    (feature_evaluation or {}).get("summary")
                    or comparison.get("summary")
                    or quality.get("summary")
                    or ""
                ),
                verified=True,
                acceptance_score=float(
                    (feature_evaluation or {}).get("reference_match_score") or 0.0
                ),
                acceptance_model=(
                    str((feature_evaluation or {}).get("model"))
                    if (feature_evaluation or {}).get("model")
                    else None
                ),
            )
            append_history(
                root,
                "feature_subjob_accepted",
                feature_id=feature_task.id,
                feature_name=feature_task.name,
                version=version,
                comparison_summary=comparison.get("summary"),
            )
            feature_summary = feature_plan_summary(root) or {}
            if feature_summary.get("required_complete"):
                try:
                    final_quality = await _generic_recognizability_check(
                        job_id,
                        stage="feature_backlog_complete_quality",
                    )
                    quality = final_quality
                    recognizable = final_quality.get("recognizable")
                    stage = (
                        "adaptive_mesh_recognizable"
                        if recognizable is True
                        else "adaptive_mesh_needs_refinement"
                    )
                    status = _write_status(
                        root,
                        state="ready",
                        stage=stage,
                        modeling_strategy="adaptive_loft",
                        generic_model=active_model,
                        quality_gate=final_quality,
                    )
                except HTTPException as exc:
                    append_history(
                        root,
                        "feature_backlog_final_quality_unavailable",
                        error=str(exc.detail),
                    )
    else:
        status = _write_status(
            root,
            state="ready",
            stage="adaptive_mesh_needs_refinement",
            modeling_strategy="adaptive_loft",
            generic_model=previous_model,
            quality_gate={
                "recognizable": False,
                "subject_match_score": previous_status.get("quality_gate", {}).get("subject_match_score")
                if isinstance(previous_status.get("quality_gate"), dict)
                else None,
                "recommended_strategy": "base_mesh",
                "summary": (
                    "The candidate refinement was not a clear improvement, so the best previous mesh was preserved. "
                    + str(comparison.get("summary") or "")
                ).strip(),
                "director_action": "refine_mesh",
                "candidate_rejected": True,
                "baseline_version": baseline_version,
                "candidate_version": version,
            },
        )
        append_history(
            root,
            "adaptive_mesh_refinement_rejected",
            baseline_version=baseline_version,
            candidate_version=version,
            comparison_summary=comparison.get("summary"),
        )
        if feature_task is not None:
            finish_feature(
                root,
                feature_task.id,
                accepted=False,
                version=None,
                summary=str(comparison.get("summary") or ""),
                error="Candidate did not beat the best-so-far mesh.",
            )
            append_history(
                root,
                "feature_subjob_retry",
                feature_id=feature_task.id,
                feature_name=feature_task.name,
                reason="candidate did not beat best-so-far mesh",
            )

    build["comparison"] = comparison
    build["feature_evaluation"] = feature_evaluation
    build["quality_gate"] = quality
    build["status"] = status
    return build


async def _generate_adaptive_mesh_fallback(
    job_id: str,
    *,
    reason: str,
    feature_task: FeatureTask | None = None,
) -> dict:
    """Try a different adaptive representation without discarding the best-so-far model."""

    root = _require_job(job_id)
    previous_status = _read_status(root)
    previous_model = (
        dict(previous_status.get("generic_model"))
        if isinstance(previous_status.get("generic_model"), dict)
        else None
    )
    previous_strategy = str(previous_status.get("modeling_strategy") or "procedural")
    previous_working_cage_version = (
        previous_status.get("working_cage_version")
        if isinstance(previous_status.get("working_cage_version"), int)
        else None
    )
    baseline_version = (
        previous_model.get("version")
        if isinstance(previous_model, dict) and isinstance(previous_model.get("version"), int)
        else previous_working_cage_version
    )

    feature_task = feature_task or begin_feature(root)
    if feature_task is not None:
        append_history(
            root,
            "feature_subjob_started",
            feature_id=feature_task.id,
            feature_name=feature_task.name,
            attempt=feature_task.attempts,
            strategy=feature_task.strategy,
            dependencies=feature_task.depends_on,
            via="adaptive_mesh_fallback",
        )

    _write_status(
        root,
        state="running",
        stage="adaptive_mesh_planning",
        active_feature_id=feature_task.id if feature_task is not None else None,
    )
    spec = await _build_adaptive_loft_spec(
        job_id,
        reason=reason,
        feature_task=feature_task,
    )
    version = reserve_model_version(root)
    build = await _execute_adaptive_loft(job_id, spec, version=version)
    candidate_model = (
        build.get("status", {}).get("generic_model")
        if isinstance(build.get("status"), dict)
        else None
    )

    comparison: dict | None = None
    if baseline_version is not None and baseline_version != version:
        comparison = await _compare_generic_versions(
            root,
            baseline_version=baseline_version,
            candidate_version=version,
        )

    feature_evaluation: dict | None = None
    if feature_task is not None:
        feature_evaluation = await _evaluate_feature_candidate(
            job_id,
            feature_task,
            baseline_version=baseline_version,
            candidate_version=version,
        )

    if feature_task is not None:
        previous_quality = _quality_snapshot(previous_status.get("quality_gate"))
        quality = dict(previous_quality)
        feature_passed = bool(
            feature_evaluation
            and _feature_evaluation_accepts(feature_task, feature_evaluation)
        )
        better = bool(
            baseline_version is None
            or (comparison and comparison.get("candidate_is_better") is True)
        )
        accept_candidate = better
        recognizable = (
            (feature_evaluation or {}).get("subject_recognizable")
            if (feature_evaluation or {}).get("subject_recognizable") is not None
            else quality.get("recognizable")
        )
        quality.update(
            {
                "summary": (
                    (feature_evaluation or {}).get("summary")
                    or (comparison or {}).get("summary")
                    or quality.get("summary")
                    or f"Worked feature {feature_task.name}."
                ),
                "active_feature_id": feature_task.id,
                "active_feature_passed": feature_passed,
                "recognizable": recognizable,
                "subject_match_score": (
                    (feature_evaluation or {}).get("reference_match_score")
                    if (feature_evaluation or {}).get("reference_match_score") is not None
                    else quality.get("subject_match_score")
                ),
                "representation": "adaptive_loft",
                "baseline_version": baseline_version,
                "candidate_version": version,
                "candidate_improved": better,
            }
        )
        quality["scope"] = "feature"
    else:
        try:
            quality = await _generic_recognizability_check(
                job_id,
                stage="generic_mesh_fallback_quality",
                render_version=version,
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
        feature_passed = False
        better = bool(
            baseline_version is None
            or (comparison and comparison.get("candidate_is_better") is True)
        )
        accept_candidate = bool(
            baseline_version is None
            or better
            or (baseline_version is None and recognizable is True)
        )

    if not accept_candidate:
        restored_quality = _quality_snapshot(previous_status.get("quality_gate"))
        restored_quality.update(
            {
                "candidate_rejected": True,
                "last_rejected_candidate_version": version,
                "last_candidate_evaluation": _quality_snapshot(quality),
                "summary": (
                    (comparison or {}).get("summary")
                    or quality.get("summary")
                    or restored_quality.get("summary")
                ),
            }
        )
        status = _write_status(
            root,
            state="ready",
            stage=previous_status.get("stage") or "generic_needs_strategy_switch",
            modeling_strategy=previous_strategy,
            generic_model=previous_model,
            working_cage_version=previous_working_cage_version,
            cage_edit_stall_count=int(previous_status.get("cage_edit_stall_count") or 0) + 1,
            quality_gate=restored_quality,
        )
        append_history(
            root,
            "adaptive_mesh_rejected",
            baseline_version=baseline_version,
            candidate_version=version,
            comparison_summary=(comparison or {}).get("summary"),
            preserved_strategy=previous_strategy,
        )
        if feature_task is not None:
            record_feature_progress(
                root,
                feature_task.id,
                version=baseline_version or version,
                summary=(
                    "Alternative representation candidate reverted; feature remains active. "
                    + str(
                        (comparison or {}).get("summary")
                        or (feature_evaluation or {}).get("summary")
                        or ""
                    )
                ),
            )
    else:
        if feature_task is not None and feature_passed:
            stage = "adaptive_mesh_feature_complete"
        elif recognizable is True:
            stage = "adaptive_mesh_recognizable"
        else:
            stage = "adaptive_mesh_needs_refinement"

        status = _write_status(
            root,
            state="ready",
            stage=stage,
            modeling_strategy="adaptive_loft",
            generic_model=candidate_model if isinstance(candidate_model, dict) else build["status"].get("generic_model"),
            working_cage_version=None,
            cage_edit_stall_count=0,
            quality_gate={
                **quality,
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
        if previous_strategy != "adaptive_loft":
            append_history(
                root,
                "representation_strategy_switched",
                from_strategy=previous_strategy,
                to_strategy="adaptive_loft",
                baseline_version=baseline_version,
                candidate_version=version,
                feature_id=feature_task.id if feature_task is not None else None,
                reason=reason,
            )

        if feature_task is not None:
            if feature_passed:
                finish_feature(
                    root,
                    feature_task.id,
                    accepted=True,
                    version=version,
                    summary=str(
                        (feature_evaluation or {}).get("summary")
                        or quality.get("summary")
                        or ""
                    ),
                    error="",
                    verified=True,
                    acceptance_score=float(
                        (feature_evaluation or {}).get("reference_match_score") or 0.0
                    ),
                    acceptance_model=(
                        str((feature_evaluation or {}).get("model"))
                        if (feature_evaluation or {}).get("model")
                        else None
                    ),
                )
                append_history(
                    root,
                    "feature_subjob_accepted",
                    feature_id=feature_task.id,
                    feature_name=feature_task.name,
                    version=version,
                    via="adaptive_mesh_fallback",
                )
            else:
                record_feature_progress(
                    root,
                    feature_task.id,
                    version=version,
                    summary=str(
                        (feature_evaluation or {}).get("summary")
                        or (comparison or {}).get("summary")
                        or "Alternative representation improved the working model but the feature is not complete."
                    ),
                )
                append_history(
                    root,
                    "feature_subjob_progress",
                    feature_id=feature_task.id,
                    feature_name=feature_task.name,
                    version=version,
                    via="adaptive_mesh_fallback",
                )

        feature_summary = feature_plan_summary(root) or {}
        if feature_task is not None and feature_passed and feature_summary.get("required_complete"):
            try:
                final_quality = await _generic_recognizability_check(
                    job_id,
                    stage="feature_backlog_complete_quality",
                    render_version=version,
                )
                quality = final_quality
                recognizable = final_quality.get("recognizable")
                status = _write_status(
                    root,
                    state="ready",
                    stage=(
                        "adaptive_mesh_recognizable"
                        if recognizable is True
                        else "adaptive_mesh_needs_refinement"
                    ),
                    modeling_strategy="adaptive_loft",
                    generic_model=candidate_model if isinstance(candidate_model, dict) else build["status"].get("generic_model"),
                    quality_gate=final_quality,
                )
            except HTTPException as exc:
                append_history(
                    root,
                    "feature_backlog_final_quality_unavailable",
                    error=str(exc.detail),
                )

    build["comparison"] = comparison
    build["feature_evaluation"] = feature_evaluation
    build["quality_gate"] = quality
    build["status"] = status
    return build

async def _ask_modeling_director(
    job_id: str,
    *,
    stage: str,
    current_strategy: str,
    include_renders: bool = True,
    render_version: int | None = None,
) -> dict:
    root = _require_job(job_id)
    job_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    inventory = _load_subject_inventory(root)
    feature_plan, feature_task = _feature_task_context(root)
    feature_plan_context = feature_plan.model_dump() if feature_plan is not None else {}
    feature_task_context = feature_task.model_dump() if feature_task is not None else {}
    images, labels = _collect_images(
        root,
        VisionAnalyzeRequest(
            stage=stage,
            include_references=True,
            include_renders=include_renders,
            max_images=10,
            render_version=render_version,
        ),
    )

    system = (
        "You are the lead 3D modeling director. Make the next modeling decision from the actual user request, "
        "reference images and current renders. You are responsible for visual judgment; the Python application "
        "only orchestrates your decision. Choose exactly one action: accept, build_procedural, revise_procedural, "
        "build_mesh, refine_mesh, or rebuild_mesh. Before renders exist, choose build_procedural or build_mesh. "
        "After renders exist, choose accept only when the object is clearly recognizable as the requested subject "
        "and its main silhouette/proportions/identity-critical parts are credible. Choose revise_procedural when "
        "the existing declarative primitive strategy can plausibly be corrected. When the current strategy is an "
        "adaptive mesh and the visible result already has the correct broad subject/category and a usable body, "
        "prefer refine_mesh: keep the best-so-far mesh and correct its proportions, silhouette, roof/upper profile, "
        "front/rear shape, wheel/attachment placement and other visible geometry. Do NOT request rebuild_mesh merely "
        "because the current mesh is crude, low-detail, or has inaccurate proportions. Use rebuild_mesh only when "
        "the main topology/axis/body concept is fundamentally wrong enough that editing the current mesh is unlikely "
        "to converge. For any mesh action set mesh_representation explicitly: hard_surface_cage is a mirrored "
        "half-profile swept horizontally along X or Y, suited to bilateral shells. adaptive_loft uses full closed "
        "cross-sections along ANY axis, including Z for upright forms. Choose the construction that actually fits "
        "the feature; do not force an upright radial form into a horizontal half-cage. build_procedural supports "
        "assemblies of beams, boxes, cylinders, arbitrary polygon mesh parts, lathe (revolve a closed material "
        "profile around Z, including interior cavities) and sweep (a smooth capped tube along a 3D path). "
        "Use build_procedural for rotational or curved-tube forms that these tools describe directly, as well as assemblies. "
        "Do not struggle with hand-written loft ring coordinates when a parametric part fits. Ignore filenames and object names as proof "
        "of correctness: judge the visible geometry. "
        "Return JSON only matching the schema."
    )
    status_payload: dict = {}
    status_path = root / "status.json"
    if status_path.exists():
        try:
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status_payload = {}
    recent_director_history = [
        event
        for event in load_history(root)[-40:]
        if event.get("event") == "modeling_director"
    ][-6:]

    prompt = (
        f"Exact user request: {job_request.get('prompt', '')}\n"
        f"Reference scope: {'parent object; focus only on component '+str(job_request.get('component_name')) if job_request.get('component_job') else 'requested whole object'}\n"
        f"Intended use: {job_request.get('intended_use', '')}\n"
        f"Current strategy: {current_strategy}\n"
        f"Current stage: {stage}\n"
        f"Current quality gate: {json.dumps(status_payload.get('quality_gate') or {}, ensure_ascii=False)}\n"
        f"Recent director decisions: {json.dumps(recent_director_history, ensure_ascii=False)}\n"
        f"AI-generated subject inventory: "
        f"{json.dumps(inventory.model_dump() if inventory else {}, ensure_ascii=False)}\n"
        f"Coordinated visible-feature plan: {json.dumps(feature_plan_context, ensure_ascii=False)}\n"
        f"Next/active feature sub-job: {json.dumps(feature_task_context, ensure_ascii=False)}\n"
        f"Images in order: {labels}\n"
        "Give concrete instructions for the next modeling pass. When a feature sub-job is supplied, prioritize its "
        "acceptance criteria while protecting already-accepted features and the best-so-far silhouette. "
        "During an active feature pass, missing geometry owned by later features is expected; do not switch representation "
        "or rebuild the primary shape simply because those later components are absent. Preserve geometry "
        "that is already moving toward the reference instead of repeatedly restarting. Spend the available reasoning "
        "budget on visual comparison and specific geometry decisions rather than generic commentary."
    )

    client = OllamaProxyClient()
    errors: list[str] = []
    candidate_models = VISION_MODELS if images else (REASONING_MODEL, *VISION_MODELS)
    director_schema = ModelingDirectorDecision.model_json_schema()
    director_schema["required"].append("mesh_representation")
    for candidate_model in candidate_models:
        try:
            result = await client.chat_json(
                model=candidate_model,
                system=system,
                prompt=prompt,
                images=images or None,
                schema=director_schema,
                temperature=0.0,
                num_predict=4096,
            )
            decision = ModelingDirectorDecision.model_validate(
                _normalize_modeling_director_payload(result.data)
            )
        except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            errors.append(f"{candidate_model}: {exc}")
            continue

        action = decision.action
        if not include_renders and action == "accept":
            action = "build_procedural"
        elif include_renders and action == "accept" and feature_task is not None:
            # The whole object cannot be declared finished while the coordinator
            # still has an unresolved visible feature sub-job.
            action = "refine_mesh" if current_strategy == "adaptive_loft" else "revise_procedural"
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


async def _generic_recognizability_check(
    job_id: str, *, stage: str, render_version: int | None = None
) -> dict:
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
        render_version=render_version,
    )
    action = decision["action"]
    recognizable = action == "accept" and float(decision.get("subject_match_score") or 0) >= 0.75
    recommended_strategy = (
        "base_mesh"
        if action in {"build_mesh", "refine_mesh", "rebuild_mesh"}
        else "procedural"
    )
    return {
        "scope": "whole_object",
        "evaluated_version": render_version or (_read_status(root).get("generic_model") or {}).get("version"),
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


async def submit_model_design(job_id: str, spec: HardSurfaceCageSpec) -> dict:
    """Render an editable design supplied by a client/modeler, then run normal visual QA."""
    root = _require_job(job_id)
    if not _usable_reference_index(root):
        raise HTTPException(status_code=424, detail="A verified reference is required to review a design.")
    if _read_status(root).get("state") == "running":
        raise HTTPException(status_code=409, detail="Wait for the current modeling pass to finish.")
    previous = _read_status(root)
    baseline = (previous.get("generic_model") or {}).get("version")
    if spec.intended_dimensions_xyz is not None:
        _validate_cage_dimensions(spec, spec.intended_dimensions_xyz)
    spec.presentation_base = False
    version = reserve_model_version(root)
    build = await _execute_hard_surface_cage(job_id, spec, version=version, activate_status=False)
    (root / "exports" / f"model-v{version}-design.json").write_text(spec.model_dump_json(indent=2))
    comparison = (await _compare_generic_versions(root, baseline_version=baseline, candidate_version=version)
                  if baseline is not None else None)
    if comparison and comparison.get("candidate_is_better") is not True:
        status = _write_status(root, state="ready", stage="design_candidate_rejected",
                               generic_model=previous.get("generic_model"),
                               modeling_strategy=previous.get("modeling_strategy"),
                               quality_gate=previous.get("quality_gate"))
        return {**build, "status": status, "kept": False, "comparison": comparison}

    # Feature completion is earned from renders, never from part names in the supplied JSON.
    _write_status(root, state="running", stage="reviewing_design", generic_model=build["candidate_model"],
                  modeling_strategy="hard_surface_cage", working_cage_version=version, cage_edit_stall_count=0,
                  quality_gate={"evaluated_version": version, "recognizable": None, "scope": "whole_object"})
    plan = await _ensure_feature_plan(job_id, _load_subject_inventory(root))
    evaluations = []
    if plan is not None:
        for feature in plan.features:
            evaluation = await _evaluate_feature_candidate(job_id, feature, baseline_version=None,
                                                            candidate_version=version)
            evaluations.append(evaluation)
            passed = _feature_evaluation_accepts(feature, evaluation)
            finish_feature(root, feature.id, accepted=passed, version=version if passed else None,
                           summary=str(evaluation.get("summary") or ""), verified=passed,
                           acceptance_score=float(evaluation.get("reference_match_score") or 0),
                           acceptance_model=evaluation.get("model"),
                           error="" if passed else "Design still needs refinement for this feature.")
    quality = await _generic_recognizability_check(job_id, stage="submitted_design_quality", render_version=version)
    status = _write_status(root, state="ready",
                           stage="design_recognizable" if quality.get("recognizable") else "design_needs_refinement",
                           quality_gate=quality)
    append_history(root, "design_submitted", version=version, source="client_geometry",
                   parts=1 + len(spec.attachments), comparison=comparison,
                   features_passed=sum(_feature_evaluation_accepts(f, e)
                                       for f, e in zip(plan.features if plan else [], evaluations, strict=True)))
    return {**build, "status": status, "kept": True, "comparison": comparison,
            "feature_evaluations": evaluations, "quality_gate": quality}


@app.post("/v1/jobs/{job_id}/design", dependencies=[Depends(require_api_token)])
async def submit_model_design_api(job_id: str, request: HardSurfaceCageSpec) -> dict:
    return await _guard_job_action(job_id, lambda: submit_model_design(job_id, request))


async def _generate_directed_mesh(job_id: str, decision: dict, *, feature_task: FeatureTask | None = None) -> dict:
    builder = (_generate_adaptive_mesh_fallback if decision.get("mesh_representation") == "adaptive_loft"
               else _generate_hard_surface_cage)
    return await builder(job_id, reason=str(decision.get("summary") or "Reference-driven construction")
                         + "\n" + "\n".join(decision.get("instructions") or []), feature_task=feature_task)


async def generate_generic_scene(job_id: str, request: GenericGenerateRequest) -> dict:
    root = _require_job(job_id)
    previous = _read_status(root)
    try:
        return await _generate_generic_scene_candidate(job_id, request)
    except Exception as exc:
        # A failed strategy switch must not promote an intermediate procedural
        # draft over the user's previous model or leave its quality attached to it.
        if isinstance(previous.get("generic_model"), dict):
            _write_status(
                root, state="ready", stage=previous.get("stage") or "generation_failed",
                generic_model=previous["generic_model"],
                modeling_strategy=previous.get("modeling_strategy"),
                quality_gate=previous.get("quality_gate"),
                last_generation_error=str(getattr(exc, "detail", exc)),
            )
            append_history(root, "generation_failed_baseline_preserved",
                           version=previous["generic_model"].get("version"),
                           error=str(getattr(exc, "detail", exc)))
        raise


async def _generate_generic_scene_candidate(job_id: str, request: GenericGenerateRequest) -> dict:
    root = _require_job(job_id)

    if request.auto_research:
        await _ensure_reference_pack(job_id, max_images=5, attempts=2)
    elif not _usable_reference_index(root):
        _write_status(
            root,
            state="ready",
            stage="waiting_for_references",
            reference_gate={
                "required": True,
                "state": "blocked",
                "usable_reference_count": 0,
                "reason": "auto_research is disabled and no user reference was uploaded.",
            },
        )
        raise HTTPException(
            status_code=424,
            detail="Modeling requires at least one verified or user-uploaded reference image.",
        )

    await _ensure_feature_plan(job_id, _load_subject_inventory(root))

    _write_status(root, state="running", stage="planning_model_geometry")

    initial_decision = await _ask_modeling_director(
        job_id,
        stage="initial_modeling_strategy",
        current_strategy="none",
        include_renders=False,
    )
    initial_action = initial_decision["action"]
    if initial_action in {"build_mesh", "rebuild_mesh"}:
        return await _generate_directed_mesh(job_id, initial_decision)

    return await _generate_procedural_candidate(job_id)


async def _generate_procedural_candidate(job_id: str, *, feature_task: FeatureTask | None = None) -> dict:
    root = _require_job(job_id)
    previous = _read_status(root)
    feature_task = feature_task or begin_feature(root)
    _write_status(root, state="running", stage="planning_procedural_geometry")
    spec = await _build_generic_scene_spec(
        job_id, auto_research=False, feature_task=feature_task,
    )
    version = reserve_model_version(root)
    build = await _execute_generic_spec(job_id, spec, version=version, activate_status=False)
    return await _review_procedural_candidate(job_id, build, previous, feature_task)


async def _review_procedural_candidate(
    job_id: str, build: dict, previous: dict, feature_task: FeatureTask | None,
    *, evaluation: dict | None = None,
) -> dict:
    """Keep better construction drafts; accept features only at the unchanged strict QA gate."""
    root = _require_job(job_id)
    candidate_model = build["candidate_model"]
    version = candidate_model["version"]
    baseline = (previous.get("generic_model") or {}).get("version")
    comparison = None
    if baseline is not None and baseline != version:
        comparison = await _compare_generic_versions(root, baseline_version=baseline, candidate_version=version)
    kept = baseline is None or baseline == version or (comparison or {}).get("candidate_is_better") is True
    accepted = False
    if feature_task is not None:
        evaluation = evaluation or await _evaluate_feature_candidate(
            job_id, feature_task, baseline_version=baseline if baseline != version else None,
            candidate_version=version,
        )
        kept = kept and evaluation.get("regression_detected") is not True
        accepted = kept and _feature_evaluation_accepts(feature_task, evaluation)
        quality = {
            "scope": "feature", "active_feature_id": feature_task.id,
            "active_feature_passed": accepted, "representation": "procedural",
            "candidate_version": version, "baseline_version": baseline, "candidate_improved": kept,
            "recognizable": evaluation.get("subject_recognizable"),
            "subject_match_score": evaluation.get("reference_match_score"),
            "summary": evaluation.get("summary"), "problems": evaluation.get("problems") or [],
        }
        if accepted:
            finish_feature(root, feature_task.id, accepted=True, version=version, verified=True,
                           summary=str(evaluation.get("summary") or ""),
                           acceptance_score=float(evaluation.get("reference_match_score") or 0),
                           acceptance_model=evaluation.get("model"))
        elif kept:
            record_feature_progress(root, feature_task.id, version=version,
                                    summary=str(evaluation.get("summary") or ""))
        else:
            finish_feature(root, feature_task.id, accepted=False, version=None,
                           summary=str(evaluation.get("summary") or ""),
                           error="Candidate did not improve the preserved model.")
    else:
        quality = await _generic_recognizability_check(job_id, stage="generic_final_quality", render_version=version)
        accepted = kept and quality.get("recognizable") is True
    if kept:
        status = _write_status(root, state="ready", modeling_strategy="procedural",
                               stage="generic_feature_complete" if accepted else "generic_needs_refinement",
                               generic_model=candidate_model, working_cage_version=None, quality_gate=quality)
    else:
        status = _write_status(root, state="ready", stage=previous.get("stage") or "generic_needs_refinement",
                               modeling_strategy=previous.get("modeling_strategy") or "procedural",
                               generic_model=previous.get("generic_model"), quality_gate=previous.get("quality_gate"))
    append_history(root, "procedural_candidate_reviewed", candidate_version=version, baseline_version=baseline,
                   feature_id=feature_task.id if feature_task else None, kept=kept, accepted=accepted,
                   summary=quality.get("summary"))
    status = await _ensure_final_model_quality(job_id, status)
    return {**build, "status": status, "kept": kept, "accepted": accepted, "comparison": comparison,
            "feature_evaluation": evaluation, "quality_gate": status.get("quality_gate")}


@app.post("/v1/jobs/{job_id}/generate", dependencies=[Depends(require_api_token)])
async def generate_generic_scene_api(job_id: str, request: GenericGenerateRequest) -> dict:
    result = await _guard_job_action(job_id, lambda: generate_generic_scene(job_id, request))
    if request.auto_improve_rounds:
        result["auto_improve_started"] = _schedule_auto_improve(
            job_id,
            request.auto_improve_rounds,
        )
    return result


@app.post("/v1/jobs/{job_id}/auto-improve", dependencies=[Depends(require_api_token)])
async def auto_improve_job_api(job_id: str, request: AutoImproveRequest) -> dict:
    root = _require_job(job_id)
    started = _schedule_auto_improve(job_id, request.rounds)
    return {
        "job_id": job_id,
        "started": started,
        "auto_improve": _read_status(root).get("auto_improve"),
    }


def _catastrophic_visual_failure(quality_gate: object, *, threshold: float = 0.20) -> tuple[bool, float]:
    if not isinstance(quality_gate, dict):
        return False, 0.0
    raw_score = (
        quality_gate.get("subject_match_score")
        if quality_gate.get("subject_match_score") is not None
        else quality_gate.get("reference_match_score")
        if quality_gate.get("reference_match_score") is not None
        else 0.0
    )
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    return quality_gate.get("recognizable") is False and score <= threshold, score


async def refine_generic_scene(job_id: str, request: GenericRefineRequest) -> dict:
    root = _require_job(job_id)
    if not _usable_reference_index(root):
        append_history(root, "reference_research_retry", reason="refinement had no usable references")
        await _ensure_reference_pack(job_id, max_images=5, attempts=2)
    await _ensure_feature_plan(job_id, _load_subject_inventory(root))
    status_path = root / "status.json"
    status_payload: dict = {}
    if status_path.exists():
        try:
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status_payload = {}

    current_strategy = str(status_payload.get("modeling_strategy") or "procedural")
    if current_strategy == "hard_surface_cage" or status_payload.get("stage") in {
        "hard_surface_cage_needs_refinement",
        "hard_surface_cage_needs_replan",
        "hard_surface_cage_feature_complete",
    }:
        feature_task = begin_feature(root)
        if feature_task is not None and feature_task.build_mode == "component_job":
            return await _build_and_install_component_feature(job_id, feature_task)

        return await _refine_hard_surface_cage_incrementally(
            job_id,
            feature_task=feature_task,
        )

    if current_strategy == "adaptive_loft" or status_payload.get("stage") in {
        "generic_needs_strategy_switch",
        "adaptive_mesh_needs_refinement",
    }:
        feature_task = begin_feature(root)
        if feature_task is not None and feature_task.build_mode == "component_job":
            return await _build_and_install_component_feature(job_id, feature_task)
        if feature_task is not None:
            append_history(
                root,
                "feature_subjob_started",
                feature_id=feature_task.id,
                feature_name=feature_task.name,
                attempt=feature_task.attempts,
                strategy=feature_task.strategy,
                dependencies=feature_task.depends_on,
                via="auto_feature_queue",
            )
            quality_gate = status_payload.get("quality_gate") or {}
            severe_failure, _ = _catastrophic_visual_failure(quality_gate)
            if severe_failure or feature_task.attempts >= 2:
                decision = await _ask_modeling_director(
                    job_id, stage="adaptive_representation_review", current_strategy="adaptive_loft",
                    include_renders=True,
                )
                if decision["action"] in {"build_procedural", "revise_procedural"}:
                    return await _generate_procedural_candidate(job_id, feature_task=feature_task)
                if decision["action"] in {"build_mesh", "rebuild_mesh"}:
                    return await _generate_directed_mesh(job_id, decision, feature_task=feature_task)
            decision = {
                "action": "refine_mesh",
                "subject_match_score": (
                    status_payload.get("quality_gate", {}).get("subject_match_score")
                    if isinstance(status_payload.get("quality_gate"), dict)
                    else None
                ),
                "summary": f"Work the queued feature sub-job: {feature_task.name}.",
                "instructions": [
                    *feature_task.acceptance_criteria,
                    *[f"Target region: {region}" for region in feature_task.target_regions],
                    *[f"Protect/own scope: {scope}" for scope in feature_task.owner_scope],
                ],
                "major_problems": [],
            }
        else:
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
        if decision["action"] in {"build_procedural", "revise_procedural"}:
            return await _generate_procedural_candidate(job_id, feature_task=feature_task)
        if decision["action"] in {"build_mesh", "rebuild_mesh"}:
            return await _generate_directed_mesh(job_id, decision, feature_task=feature_task)
        return await _refine_adaptive_mesh(
            job_id,
            decision=decision,
            feature_task=feature_task,
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
        feature_task = begin_feature(root)
        if feature_task is not None and feature_task.build_mode == "component_job":
            return await _build_and_install_component_feature(job_id, feature_task)
        current_evaluation = None
        if feature_task is not None:
            # Existing assemblies may already contain a finished feature. Review
            # its actual pixels before spending tokens rebuilding good geometry.
            current_evaluation = await _evaluate_feature_candidate(
                job_id, feature_task, baseline_version=None, candidate_version=current_version,
            )
            if _feature_evaluation_accepts(feature_task, current_evaluation):
                result = await _review_procedural_candidate(
                    job_id, {"candidate_model": _read_status(root)["generic_model"], "spec": current_spec.model_dump()},
                    _read_status(root), feature_task, evaluation=current_evaluation,
                )
                completed.append(result)
                continue
        decision = await _ask_modeling_director(
            job_id, stage=f"agent_refinement_{offset + 1}", current_strategy="procedural",
            include_renders=True, render_version=current_version,
        )
        if decision["action"] in {"build_mesh", "rebuild_mesh"}:
            return await _generate_directed_mesh(job_id, decision, feature_task=feature_task)
        if decision["action"] == "accept" and feature_task is None:
            break
        decision["feature_evaluation"] = current_evaluation
        images, labels = _collect_images(root, VisionAnalyzeRequest(
            stage="agent_scene_revision", include_references=True, include_renders=True,
            max_images=10, render_version=current_version,
        ))
        plan = load_feature_plan(root)
        system = (
            "You construct geometry for an autonomous Blender modeler. The visual director and feature reviewer "
            "have already diagnosed the actual renders. Correct the diagnosed geometry and return the complete "
            "replacement SceneSpec. Preserve useful parts and their names. Use full dimensions for primitives, "
            "lathe for revolved material profiles (including hollow walls), sweep for curved round sections, "
            "or mesh for arbitrary polygons. Do not use subject templates. Return JSON only matching the schema."
        )
        prompt = (
            f"User request: {job_request.get('prompt', '')}\n"
            f"Current SceneSpec: {json.dumps(current_spec.model_dump())}\n"
            f"Visual diagnosis: {json.dumps(decision)}\n"
            f"Subject inventory: {json.dumps(inventory.model_dump() if inventory else {})}\n"
            f"Feature plan: {json.dumps(plan.model_dump() if plan else {})}\n"
            f"ACTIVE FEATURE: {json.dumps(feature_task.model_dump() if feature_task else {})}\n"
            f"Coordinate contract: {_generic_spatial_guidance('')}\n"
            f"Reference/current image order: {labels}\n"
            "Fix the active feature and protect other geometry. Never create parts owned by component_job entries; "
            "the assembler installs those separately. Check local dimensions and resulting world extents for every changed part."
        )
        revised, selected_model, errors = None, None, []
        client = OllamaProxyClient()
        for candidate_model in (REASONING_MODEL, *VISION_MODELS):
            try:
                result = await client.chat_json(
                    model=candidate_model, system=system, prompt=prompt,
                    images=(images or None) if candidate_model in VISION_MODELS else None,
                    schema=GenericSceneSpec.model_json_schema(), temperature=0.0, num_predict=8192,
                )
                revised = GenericSceneSpec.model_validate(_normalize_scene_spec_payload(result.data, current_spec.title))
                if job_request.get("component_job") is True:
                    revised.presentation_base = False
            except (OllamaProxyError, httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
                errors.append(f"{candidate_model}: {exc}")
                continue
            selected_model = candidate_model
            break
        if revised is None:
            raise HTTPException(status_code=502, detail="Scene revision failed: " + " | ".join(errors[-3:]))
        if revised.model_dump() == current_spec.model_dump():
            append_history(root, "agent_refinement_stop", reason="AI returned unchanged geometry")
            if feature_task:
                finish_feature(root, feature_task.id, accepted=False, version=None,
                               error="Modeler returned unchanged geometry after failed feature review.")
            break
        previous = _read_status(root)
        version = reserve_model_version(root)
        build = await _execute_generic_spec(job_id, revised, version=version, activate_status=False)
        result = await _review_procedural_candidate(job_id, build, previous, feature_task)
        append_history(root, "agent_revision", version=version, model=selected_model,
                       object_count=len(revised.objects), kept=result["kept"])
        completed.append(result)
        if not result["kept"]:
            break
        current_spec, current_version = revised, version
    status = await _ensure_final_model_quality(job_id, _read_status(root))
    return {"job_id": job_id, "iterations": completed, "status": status,
            "quality_gate": status.get("quality_gate")}


@app.post("/v1/jobs/{job_id}/improve", dependencies=[Depends(require_api_token)])
async def refine_generic_scene_api(job_id: str, request: GenericRefineRequest) -> dict:
    return await _guard_job_action(job_id, lambda: refine_generic_scene(job_id, request))


@app.post("/v1/jobs/{job_id}/quality-benchmark", dependencies=[Depends(require_api_token)])
async def quality_benchmark_api(job_id: str, request: QualityBenchmarkRequest) -> dict:
    return await _guard_job_action(job_id, lambda: evaluate_quality_benchmark(job_id, request))


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

    if request.auto_research and not _usable_reference_index(root):
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
    return await _guard_job_action(job_id, lambda: repair_print_model(job_id, request))


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
