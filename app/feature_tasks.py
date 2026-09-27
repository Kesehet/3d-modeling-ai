# ruff: noqa: I001
from __future__ import annotations

import json
import re
import typing
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


FeatureState = typing.Literal["pending", "ready", "running", "accepted", "retry", "blocked", "failed"]
FeatureStrategy = typing.Literal[
    "base_mesh_region",
    "attachment",
    "surface_cutout",
    "surface_detail",
    "mixed",
]


class FeatureTask(BaseModel):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=120)
    category: str = Field(default="visible_feature", max_length=80)
    required: bool = True
    priority: int = Field(default=5, ge=1, le=10)
    count: int = Field(default=1, ge=1, le=32)
    strategy: FeatureStrategy = "mixed"
    symmetry: typing.Literal["none", "bilateral", "paired", "radial"] = "none"
    parent: str | None = Field(default=None, max_length=80)
    depends_on: list[str] = Field(default_factory=list, max_length=16)
    target_regions: list[str] = Field(default_factory=list, max_length=16)
    acceptance_criteria: list[str] = Field(default_factory=list, max_length=12)
    owner_scope: list[str] = Field(default_factory=list, max_length=16)
    status: FeatureState = "pending"
    attempts: int = Field(default=0, ge=0, le=20)
    accepted_version: int | None = Field(default=None, ge=1)
    last_summary: str = Field(default="", max_length=1000)
    last_error: str = Field(default="", max_length=1000)


class FeatureEvaluation(BaseModel):
    feature_id: str = Field(min_length=1, max_length=80)
    passed: bool = False
    visible: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    regression_detected: bool = False
    summary: str = Field(default="", max_length=1600)
    problems: list[str] = Field(default_factory=list, max_length=12)
    protected_geometry_notes: list[str] = Field(default_factory=list, max_length=12)


class FeaturePlan(BaseModel):
    subject: str = Field(default="", max_length=160)
    coordinator_notes: str = Field(default="", max_length=2000)
    features: list[FeatureTask] = Field(min_length=1, max_length=48)
    active_feature_id: str | None = Field(default=None, max_length=80)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def _slug(value: str, fallback: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return (text or fallback)[:72]


def _string_list(
    source: dict,
    *keys: str,
    limit: int = 16,
) -> list[str]:
    value = None
    for key in keys:
        if key in source:
            value = source.get(key)
            break
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip()[:240] for item in value if str(item).strip()][:limit]


def normalize_feature_plan_payload(data: object, *, subject: str) -> dict:
    if not isinstance(data, dict):
        raise TypeError("Feature plan response is not a JSON object.")

    raw_features = (
        data.get("features")
        or data.get("tasks")
        or data.get("visible_features")
        or data.get("feature_inventory")
    )
    if isinstance(raw_features, dict):
        raw_features = list(raw_features.values())
    if not isinstance(raw_features, list):
        raise TypeError("Feature plan must include a feature list.")

    normalized: list[dict] = []
    used_ids: set[str] = set()
    alias_to_id: dict[str, str] = {}

    for index, raw in enumerate(raw_features[:48]):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or raw.get("feature") or raw.get("title") or f"Feature {index + 1}").strip()
        proposed = str(raw.get("id") or raw.get("feature_id") or name)
        feature_id = _slug(proposed, f"feature-{index + 1}")
        base = feature_id
        suffix = 2
        while feature_id in used_ids:
            feature_id = f"{base[:64]}-{suffix}"
            suffix += 1
        used_ids.add(feature_id)

        aliases = {name.lower(), proposed.lower(), feature_id.lower()}
        for alias in aliases:
            alias_to_id[alias] = feature_id

        strategy = str(raw.get("strategy") or raw.get("modeling_strategy") or "mixed").strip().lower()
        strategy_aliases = {
            "mesh": "base_mesh_region",
            "mesh_region": "base_mesh_region",
            "body": "base_mesh_region",
            "primitive": "attachment",
            "object": "attachment",
            "cutout": "surface_cutout",
            "opening": "surface_cutout",
            "detail": "surface_detail",
        }
        strategy = strategy_aliases.get(strategy, strategy)
        if strategy not in {
            "base_mesh_region",
            "attachment",
            "surface_cutout",
            "surface_detail",
            "mixed",
        }:
            strategy = "mixed"

        symmetry = str(raw.get("symmetry") or "none").strip().lower()
        if symmetry not in {"none", "bilateral", "paired", "radial"}:
            symmetry = "none"

        try:
            priority = int(raw.get("priority", 5))
        except (TypeError, ValueError):
            priority = 5
        try:
            count = int(raw.get("count", 1))
        except (TypeError, ValueError):
            count = 1

        normalized.append(
            {
                "id": feature_id,
                "name": name[:120],
                "category": str(raw.get("category") or raw.get("type") or "visible_feature")[:80],
                "required": bool(raw.get("required", True)),
                "priority": max(1, min(10, priority)),
                "count": max(1, min(32, count)),
                "strategy": strategy,
                "symmetry": symmetry,
                "parent": str(raw.get("parent") or "").strip()[:80] or None,
                "depends_on": _string_list(raw, "depends_on", "dependencies", "requires"),
                "target_regions": _string_list(raw, "target_regions", "regions", "target_region"),
                "acceptance_criteria": _string_list(
                    raw,
                    "acceptance_criteria",
                    "criteria",
                    "success_criteria",
                    limit=12,
                ),
                "owner_scope": _string_list(raw, "owner_scope", "ownership", "editable_regions"),
                "status": "pending",
                "attempts": 0,
                "accepted_version": None,
                "last_summary": "",
                "last_error": "",
            }
        )

    if not normalized:
        raise ValueError("Feature plan did not contain usable features.")

    valid_ids = {item["id"] for item in normalized}
    for item in normalized:
        resolved: list[str] = []
        for dependency in item["depends_on"]:
            key = dependency.lower()
            candidate = alias_to_id.get(key) or _slug(dependency, "")
            if candidate in valid_ids and candidate != item["id"] and candidate not in resolved:
                resolved.append(candidate)
        item["depends_on"] = resolved

        parent = item.get("parent")
        if parent:
            resolved_parent = alias_to_id.get(parent.lower()) or _slug(parent, "")
            item["parent"] = resolved_parent if resolved_parent in valid_ids else None

    return {
        "subject": str(data.get("subject") or subject)[:160],
        "coordinator_notes": str(
            data.get("coordinator_notes")
            or data.get("notes")
            or data.get("coordination")
            or ""
        )[:2000],
        "features": normalized,
        "active_feature_id": None,
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
    }


def feature_plan_path(root: Path) -> Path:
    return root / "feature-plan.json"


def load_feature_plan(root: Path) -> FeaturePlan | None:
    path = feature_plan_path(root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return FeaturePlan.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError, TypeError):
        return None


def save_feature_plan(root: Path, plan: FeaturePlan) -> FeaturePlan:
    plan.updated_at = datetime.now(UTC).isoformat()
    path = feature_plan_path(root)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(plan.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)
    return plan


def refresh_feature_states(plan: FeaturePlan) -> FeaturePlan:
    # A dependency that exhausted its own attempts should not freeze every
    # downstream visible feature forever. "Resolved" means the coordinator has
    # finished attempting that dependency, even if it could not be accepted.
    resolved = {
        feature.id
        for feature in plan.features
        if feature.status in {"accepted", "blocked", "failed"}
    }
    feature_ids = {feature.id for feature in plan.features}

    for feature in plan.features:
        if feature.status in {"accepted", "running", "failed"}:
            continue
        if any(dep not in feature_ids for dep in feature.depends_on):
            feature.status = "blocked"
            feature.last_error = "One or more dependencies are missing from the feature plan."
            continue
        if all(dep in resolved for dep in feature.depends_on):
            if feature.status in {"pending", "blocked"}:
                feature.status = "ready"
        elif feature.status == "ready":
            feature.status = "pending"
    return plan


def active_or_next_feature(plan: FeaturePlan) -> FeatureTask | None:
    plan = refresh_feature_states(plan)
    if plan.active_feature_id:
        active = next(
            (
                feature
                for feature in plan.features
                if feature.id == plan.active_feature_id and feature.status == "running"
            ),
            None,
        )
        if active is not None:
            return active

    candidates = [
        feature
        for feature in plan.features
        if feature.status in {"ready", "retry"}
    ]
    if not candidates:
        # Defensive deadlock recovery for an imperfect AI-authored dependency
        # graph. Break a cycle by releasing the highest-value unresolved task
        # rather than leaving the whole job permanently blocked.
        unresolved = [
            feature
            for feature in plan.features
            if feature.status == "pending"
        ]
        if unresolved:
            unresolved.sort(
                key=lambda feature: (
                    0 if feature.required else 1,
                    -feature.priority,
                    feature.attempts,
                    feature.name.lower(),
                )
            )
            unresolved[0].status = "ready"
            unresolved[0].last_error = (
                "Dependency deadlock was bypassed by the coordinator so feature work can continue."
            )
            candidates = [unresolved[0]]
    if not candidates:
        return None
    candidates.sort(
        key=lambda feature: (
            0 if feature.required else 1,
            -feature.priority,
            feature.attempts,
            feature.name.lower(),
        )
    )
    return candidates[0]


def begin_feature(root: Path) -> FeatureTask | None:
    plan = load_feature_plan(root)
    if plan is None:
        return None
    task = active_or_next_feature(plan)
    if task is None:
        save_feature_plan(root, plan)
        return None
    task.status = "running"
    task.attempts += 1
    task.last_error = ""
    plan.active_feature_id = task.id
    save_feature_plan(root, plan)
    return task


def finish_feature(
    root: Path,
    feature_id: str,
    *,
    accepted: bool,
    version: int | None,
    summary: str = "",
    error: str = "",
    max_attempts: int = 3,
) -> FeaturePlan | None:
    plan = load_feature_plan(root)
    if plan is None:
        return None
    task = next((feature for feature in plan.features if feature.id == feature_id), None)
    if task is None:
        return plan

    task.last_summary = summary[:1000]
    task.last_error = error[:1000]
    if accepted:
        task.status = "accepted"
        task.accepted_version = version
    elif task.attempts >= max_attempts:
        task.status = "blocked"
    else:
        task.status = "retry"

    if plan.active_feature_id == feature_id:
        plan.active_feature_id = None
    refresh_feature_states(plan)
    save_feature_plan(root, plan)
    return plan


def feature_plan_summary(root: Path) -> dict | None:
    plan = load_feature_plan(root)
    if plan is None:
        return None
    refresh_feature_states(plan)
    counts: dict[str, int] = {}
    for feature in plan.features:
        counts[feature.status] = counts.get(feature.status, 0) + 1
    next_task = active_or_next_feature(plan)
    return {
        "subject": plan.subject,
        "coordinator_notes": plan.coordinator_notes,
        "active_feature_id": plan.active_feature_id,
        "next_feature_id": next_task.id if next_task else None,
        "counts": counts,
        "complete": all(
            feature.status in {"accepted", "blocked", "failed"}
            for feature in plan.features
        ),
        "features": [feature.model_dump() for feature in plan.features],
        "updated_at": plan.updated_at,
    }
