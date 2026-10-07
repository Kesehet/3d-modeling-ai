# ruff: noqa: I001
from __future__ import annotations

import json
import re
import typing
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


FeatureState = typing.Literal["pending", "ready", "running", "component_ready", "accepted", "retry", "blocked", "failed"]
FeatureBuildMode = typing.Literal["in_place", "component_job"]
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
    build_mode: FeatureBuildMode = "in_place"
    symmetry: typing.Literal["none", "bilateral", "paired", "radial"] = "none"
    parent: str | None = Field(default=None, max_length=80)
    component_job_id: str | None = Field(default=None, max_length=80)
    component_depth: int = Field(default=0, ge=0, le=8)
    component_version: int | None = Field(default=None, ge=1)
    component_artifact: str | None = Field(default=None, max_length=240)
    assembly_anchor: str = Field(default="", max_length=240)
    assembly_notes: list[str] = Field(default_factory=list, max_length=12)
    depends_on: list[str] = Field(default_factory=list, max_length=16)
    target_regions: list[str] = Field(default_factory=list, max_length=16)
    acceptance_criteria: list[str] = Field(default_factory=list, max_length=12)
    owner_scope: list[str] = Field(default_factory=list, max_length=16)
    status: FeatureState = "pending"
    attempts: int = Field(default=0, ge=0, le=20)
    accepted_version: int | None = Field(default=None, ge=1)
    acceptance_verified: bool = False
    acceptance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    acceptance_model: str | None = Field(default=None, max_length=120)
    last_summary: str = Field(default="", max_length=1000)
    last_error: str = Field(default="", max_length=1000)


class FeatureEvaluation(BaseModel):
    feature_id: str = Field(min_length=1, max_length=80)
    passed: bool
    visible: bool
    criteria_satisfied: bool
    subject_recognizable: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reference_match_score: float = Field(ge=0.0, le=1.0)
    regression_detected: bool
    summary: str = Field(min_length=1, max_length=1600)
    problems: list[str] = Field(default_factory=list, max_length=12)
    protected_geometry_notes: list[str] = Field(default_factory=list, max_length=12)


class FeaturePlan(BaseModel):
    plan_version: int = Field(default=3, ge=1)
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


def _extract_feature_plan_payload(data: object) -> tuple[dict, list]:
    """Accept common model wrappers while still requiring a concrete feature list."""
    if isinstance(data, list):
        return {}, data
    if not isinstance(data, dict):
        raise TypeError("Feature plan response is not a JSON object or list.")

    feature_keys = (
        "features",
        "tasks",
        "visible_features",
        "feature_inventory",
        "components",
        "parts",
        "subjobs",
        "sub_jobs",
        "work_items",
        "children",
    )
    wrapper_keys = (
        "feature_plan",
        "plan",
        "build_plan",
        "decomposition",
        "result",
        "output",
        "response",
        "data",
        "analysis",
        "payload",
    )

    queue: list[dict] = [data]
    seen: set[int] = set()
    while queue:
        node = queue.pop(0)
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)

        for key in feature_keys:
            value = node.get(key)
            if isinstance(value, list):
                return node, value
            if isinstance(value, dict):
                nested_lists = [
                    value.get(nested_key)
                    for nested_key in feature_keys
                    if isinstance(value.get(nested_key), list)
                ]
                if nested_lists:
                    return value, nested_lists[0]
                if value and all(isinstance(item, dict) for item in value.values()):
                    return node, list(value.values())
                queue.append(value)

        for key in wrapper_keys:
            child = node.get(key)
            if isinstance(child, dict):
                queue.append(child)
            elif isinstance(child, list) and child and all(isinstance(item, dict) for item in child):
                # Some models use {"plan": [{...feature...}, ...]}.
                return node, child

    raise TypeError(
        "Feature plan must include a feature list under features/tasks/components/parts/subjobs."
    )


def normalize_feature_plan_payload(data: object, *, subject: str) -> dict:
    plan_container, raw_features = _extract_feature_plan_payload(data)

    normalized: list[dict] = []
    used_ids: set[str] = set()
    alias_to_id: dict[str, str] = {}

    for index, raw in enumerate(raw_features[:48]):
        if not isinstance(raw, dict):
            continue
        name = str(
            raw.get("name")
            or raw.get("feature_name")
            or raw.get("part_name")
            or raw.get("feature")
            or raw.get("title")
            or f"Feature {index + 1}"
        ).strip()
        proposed = str(raw.get("id") or raw.get("feature_id") or name)
        feature_id = _slug(proposed, f"feature-{index + 1}")
        # Models sometimes emit a useful semantic id but a placeholder display
        # label ("Feature 1"). Preserve the semantic identity for downstream
        # focused workers and visual QA instead of feeding them meaningless text.
        if re.fullmatch(r"feature[\s_-]*\d+", name, flags=re.IGNORECASE):
            semantic_name = re.sub(r"[-_]+", " ", proposed).strip()
            if semantic_name and not re.fullmatch(
                r"feature[\s_-]*\d+", semantic_name, flags=re.IGNORECASE
            ):
                name = semantic_name
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

        build_mode = str(
            raw.get("build_mode")
            or raw.get("execution_mode")
            or raw.get("worker_mode")
            or "in_place"
        ).strip().lower()
        build_mode_aliases = {
            "component": "component_job",
            "component-job": "component_job",
            "separate": "component_job",
            "isolated": "component_job",
            "recursive": "component_job",
            "subjob": "component_job",
            "sub-job": "component_job",
            "shared_scene": "in_place",
            "inline": "in_place",
        }
        build_mode = build_mode_aliases.get(build_mode, build_mode)
        if build_mode not in {"in_place", "component_job"}:
            build_mode = "in_place"

        try:
            priority = int(raw.get("priority", 5))
        except (TypeError, ValueError):
            priority = 5
        count_value = (
            raw.get("count")
            if raw.get("count") is not None
            else raw.get("quantity")
            if raw.get("quantity") is not None
            else raw.get("instances")
            if raw.get("instances") is not None
            else raw.get("instance_count", 1)
        )
        try:
            count = int(count_value)
        except (TypeError, ValueError):
            count = 1

        # Recover obvious repeated-instance intent when a model encoded it in
        # target/symmetry text instead of the numeric field.
        repetition_text = " ".join(
            str(raw.get(key) or "")
            for key in ("target_region", "target_regions", "notes", "description")
        ).lower()
        if count <= 1:
            if re.search(r"all[ _-]*four|four wheels|4 wheels", repetition_text):
                count = 4
            elif symmetry in {"bilateral", "paired"} and strategy != "base_mesh_region":
                # A mirrored primary/base mesh is one feature with bilateral
                # construction, not two independent feature instances.
                count = 2

        normalized.append(
            {
                "id": feature_id,
                "name": name[:120],
                "category": str(raw.get("category") or raw.get("type") or "visible_feature")[:80],
                "required": bool(raw.get("required", True)),
                "priority": max(1, min(10, priority)),
                "count": max(1, min(32, count)),
                "strategy": strategy,
                "build_mode": build_mode,
                "symmetry": symmetry,
                "parent": str(raw.get("parent") or "").strip()[:80] or None,
                "component_job_id": None,
                "component_depth": 0,
                "component_version": None,
                "component_artifact": None,
                "assembly_anchor": str(
                    raw.get("assembly_anchor")
                    or raw.get("anchor")
                    or raw.get("mount_point")
                    or ""
                ).strip()[:240],
                "assembly_notes": _string_list(
                    raw,
                    "assembly_notes",
                    "installation_notes",
                    "assembly",
                    limit=12,
                ),
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
        "plan_version": 3,
        "subject": str(
            plan_container.get("subject")
            or (data.get("subject") if isinstance(data, dict) else None)
            or subject
        )[:160],
        "coordinator_notes": str(
            plan_container.get("coordinator_notes")
            or plan_container.get("notes")
            or plan_container.get("coordination")
            or (data.get("coordinator_notes") if isinstance(data, dict) else None)
            or (data.get("notes") if isinstance(data, dict) else None)
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
    # Never trust a legacy/accidental accepted flag without strict visual proof.
    for feature in plan.features:
        if feature.status == "accepted" and not feature.acceptance_verified:
            feature.status = "retry"
            feature.accepted_version = None
            feature.acceptance_score = 0.0
            feature.acceptance_model = None
            feature.last_error = (
                "Accepted state lacked strict reference verification and was requeued."
            )

    feature_by_id = {feature.id: feature for feature in plan.features}
    accepted = {
        feature.id
        for feature in plan.features
        if feature.status == "accepted" and feature.acceptance_verified
    }
    failed = {
        feature.id
        for feature in plan.features
        if feature.status in {"blocked", "failed"}
    }

    for feature in plan.features:
        if feature.status in {"accepted", "running", "component_ready", "failed"}:
            continue

        missing = [dep for dep in feature.depends_on if dep not in feature_by_id]
        if missing:
            feature.status = "blocked"
            feature.last_error = (
                "Blocked because dependencies are missing from the plan: "
                + ", ".join(missing)
            )
            continue

        failed_dependencies = [dep for dep in feature.depends_on if dep in failed]
        if failed_dependencies:
            feature.status = "blocked"
            feature.last_error = (
                "Blocked because required dependency failed: "
                + ", ".join(failed_dependencies)
            )
            failed.add(feature.id)
            continue

        if all(dep in accepted for dep in feature.depends_on):
            if feature.status in {"pending", "blocked"}:
                feature.status = "ready"
                if feature.last_error.startswith("Blocked because"):
                    feature.last_error = ""
        elif feature.status in {"ready", "blocked", "retry"}:
            # A retry retains its attempt budget, but does not bypass dependencies.
            # Assembly repair can reopen an installed component after its dependent
            # finish task was requeued. That finish must wait for reinstallation.
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
        if feature.status in {"ready", "retry", "component_ready"}
    ]
    if not candidates:
        # Defensive deadlock recovery for an imperfect AI-authored dependency
        # graph. Break a cycle by releasing the highest-value unresolved task
        # rather than leaving the whole job permanently blocked.
        feature_by_id = {feature.id: feature for feature in plan.features}
        unresolved = [
            feature
            for feature in plan.features
            if feature.status == "pending"
            and feature.depends_on
            and all(
                dep in feature_by_id
                and feature_by_id[dep].status not in {"blocked", "failed"}
                for dep in feature.depends_on
            )
        ]
        if unresolved:
            order = {feature.id: index for index, feature in enumerate(plan.features)}
            unresolved.sort(
                key=lambda feature: (
                    0 if feature.required else 1,
                    order.get(feature.id, 10_000),
                    feature.attempts,
                    -feature.priority,
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
    order = {feature.id: index for index, feature in enumerate(plan.features)}
    candidates.sort(
        key=lambda feature: (
            0 if feature.required else 1,
            0 if feature.build_mode == "in_place" else 1,
            order.get(feature.id, 10_000),
            feature.attempts,
            -feature.priority,
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
    already_running = task.status == "running"
    installing_component = task.status == "component_ready"
    task.status = "running"
    if not already_running and not installing_component:
        task.attempts += 1
    task.last_error = ""
    plan.active_feature_id = task.id
    save_feature_plan(root, plan)
    return task


def record_feature_progress(
    root: Path,
    feature_id: str,
    *,
    version: int,
    summary: str = "",
) -> FeaturePlan | None:
    """Keep an improving feature active without falsely accepting or failing it."""

    plan = load_feature_plan(root)
    if plan is None:
        return None
    task = next((feature for feature in plan.features if feature.id == feature_id), None)
    if task is None:
        return plan

    task.status = "running"
    task.last_summary = summary[:1000]
    task.last_error = ""
    task.accepted_version = None
    task.acceptance_verified = False
    task.acceptance_score = 0.0
    task.acceptance_model = None
    plan.active_feature_id = task.id
    save_feature_plan(root, plan)
    return plan


def finish_feature(
    root: Path,
    feature_id: str,
    *,
    accepted: bool,
    version: int | None,
    summary: str = "",
    error: str = "",
    max_attempts: int = 3,
    verified: bool = False,
    acceptance_score: float = 0.0,
    acceptance_model: str | None = None,
) -> FeaturePlan | None:
    plan = load_feature_plan(root)
    if plan is None:
        return None
    task = next((feature for feature in plan.features if feature.id == feature_id), None)
    if task is None:
        return plan

    task.last_summary = summary[:1000]
    task.last_error = error[:1000]
    if accepted and verified:
        task.status = "accepted"
        task.accepted_version = version
        task.acceptance_verified = True
        task.acceptance_score = max(0.0, min(1.0, float(acceptance_score)))
        task.acceptance_model = acceptance_model
    elif task.attempts >= max_attempts:
        task.status = "failed"
    else:
        task.status = "retry"

    if not (accepted and verified):
        task.accepted_version = None
        task.acceptance_verified = False
        task.acceptance_score = 0.0
        task.acceptance_model = None

    if plan.active_feature_id == feature_id:
        plan.active_feature_id = None
    refresh_feature_states(plan)
    save_feature_plan(root, plan)
    return plan


def link_component_job(
    root: Path,
    feature_id: str,
    *,
    component_job_id: str,
    component_depth: int,
) -> FeaturePlan | None:
    plan = load_feature_plan(root)
    if plan is None:
        return None
    task = next((feature for feature in plan.features if feature.id == feature_id), None)
    if task is None:
        return plan
    task.build_mode = "component_job"
    task.component_job_id = component_job_id[:80]
    task.component_depth = max(0, min(8, int(component_depth)))
    task.status = "running"
    task.last_error = ""
    plan.active_feature_id = task.id
    save_feature_plan(root, plan)
    return plan


def mark_component_ready(
    root: Path,
    feature_id: str,
    *,
    component_version: int,
    component_artifact: str,
    summary: str = "",
) -> FeaturePlan | None:
    plan = load_feature_plan(root)
    if plan is None:
        return None
    task = next((feature for feature in plan.features if feature.id == feature_id), None)
    if task is None:
        return plan
    task.build_mode = "component_job"
    task.component_version = max(1, int(component_version))
    task.component_artifact = component_artifact[:240]
    task.status = "component_ready"
    task.last_summary = summary[:1000]
    task.last_error = ""
    if plan.active_feature_id == feature_id:
        plan.active_feature_id = None
    save_feature_plan(root, plan)
    return plan


def retry_feature(
    root: Path,
    feature_id: str,
    *,
    reset_attempts: bool = True,
) -> FeaturePlan | None:
    """Requeue one failed/blocked feature without discarding already accepted siblings."""

    plan = load_feature_plan(root)
    if plan is None:
        return None
    task = next((feature for feature in plan.features if feature.id == feature_id), None)
    if task is None:
        return plan

    task.status = "retry"
    if reset_attempts:
        task.attempts = 0
    task.accepted_version = None
    task.acceptance_verified = False
    task.acceptance_score = 0.0
    task.acceptance_model = None
    task.last_summary = ""
    task.last_error = ""
    plan.active_feature_id = None
    refresh_feature_states(plan)
    save_feature_plan(root, plan)
    return plan


def invalidate_unverified_acceptances(root: Path) -> int:
    """Requeue acceptances created before strict reference-based feature QA."""
    plan = load_feature_plan(root)
    if plan is None:
        return 0

    reset = 0
    for feature in plan.features:
        if feature.status == "accepted" and not feature.acceptance_verified:
            feature.status = "retry"
            feature.accepted_version = None
            feature.acceptance_score = 0.0
            feature.acceptance_model = None
            feature.last_error = (
                "Requeued because this feature was accepted before strict "
                "reference-based acceptance verification was enabled."
            )
            reset += 1

    if reset:
        plan.active_feature_id = None
        refresh_feature_states(plan)
        save_feature_plan(root, plan)
    return reset


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
        "plan_version": plan.plan_version,
        "subject": plan.subject,
        "coordinator_notes": plan.coordinator_notes,
        "active_feature_id": plan.active_feature_id,
        "next_feature_id": next_task.id if next_task else None,
        "counts": counts,
        "complete": all(
            (
                feature.status == "accepted"
                and feature.acceptance_verified
            )
            or feature.status in {"blocked", "failed"}
            for feature in plan.features
        ),
        "required_complete": all(
            (not feature.required)
            or (
                feature.status == "accepted"
                and feature.acceptance_verified
            )
            for feature in plan.features
        ),
        "required_unresolved": [
            feature.id
            for feature in plan.features
            if feature.required
            and not (
                feature.status == "accepted"
                and feature.acceptance_verified
            )
        ],
        "features": [feature.model_dump() for feature in plan.features],
        "updated_at": plan.updated_at,
    }
