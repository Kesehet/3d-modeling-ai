import json

import pytest
from fastapi import HTTPException

from app import main
from app.component_assembly import component_assembly_script
from app.feature_tasks import (
    FeaturePlan,
    FeatureTask,
    load_feature_plan,
    normalize_feature_plan_payload,
    save_feature_plan,
)


def _parent_job(tmp_path, monkeypatch, *, depth=0):
    jobs_root = tmp_path / "jobs"
    parent_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    root = jobs_root / parent_id
    root.mkdir(parents=True)
    for category in main.ARTIFACT_CATEGORIES:
        (root / category).mkdir(exist_ok=True)
    (root / "request.json").write_text(
        json.dumps(
            {
                "job_id": parent_id,
                "prompt": "Toyota Prius",
                "intended_use": "rendering",
                "component_depth": depth,
            }
        ),
        encoding="utf-8",
    )
    plan = FeaturePlan.model_validate(
        normalize_feature_plan_payload(
            {
                "subject": "Toyota Prius",
                "features": [
                    {
                        "id": "wheel",
                        "name": "Wheel assembly",
                        "build_mode": "component_job",
                        "count": 4,
                        "acceptance_criteria": [
                            "tire, rim and visible fasteners read as one finished wheel"
                        ],
                        "assembly_anchor": "front/rear axle wheel centers",
                    }
                ],
            },
            subject="Toyota Prius",
        )
    )
    save_feature_plan(root, plan)
    monkeypatch.setattr(main, "JOBS_ROOT", jobs_root)
    return parent_id, root, plan.features[0]


def test_component_child_job_is_linked_to_parent_feature(tmp_path, monkeypatch):
    parent_id, root, feature = _parent_job(tmp_path, monkeypatch)

    child_id = main._create_component_child_job(parent_id, feature)

    child_root = main.JOBS_ROOT / child_id
    child_request = json.loads((child_root / "request.json").read_text(encoding="utf-8"))
    assert child_request["component_job"] is True
    assert child_request["component_depth"] == 1
    assert child_request["parent_job_id"] == parent_id
    assert child_request["parent_feature_id"] == "wheel"
    assert "Model ONLY the isolated component 'Wheel assembly'" in child_request["prompt"]
    assert "Do NOT model the complete parent object" in child_request["prompt"]

    persisted = load_feature_plan(root)
    assert persisted is not None
    wheel = persisted.features[0]
    assert wheel.component_job_id == child_id
    assert wheel.component_depth == 1




def test_component_child_prompt_excludes_parent_owned_sibling_geometry():
    component = FeatureTask(
        id="hub",
        name="Central hub",
        build_mode="component_job",
        acceptance_criteria=[
            "Round cylindrical silhouette",
            "Includes three equidistant blade mounting sockets",
        ],
        target_regions=["central_hub"],
        owner_scope=["hub_shell"],
    )
    sibling = FeatureTask(
        id="blade-sockets",
        name="Blade Mounting Sockets",
        strategy="surface_cutout",
        target_regions=["hub_perimeter"],
        owner_scope=["socket_cutouts"],
    )

    prompt = main._component_child_prompt(
        {"prompt": "A simple three-blade fan"},
        component,
        excluded_features=[sibling],
    )

    assert "Round cylindrical silhouette" in prompt
    assert "Includes three equidistant blade mounting sockets" not in prompt
    assert "Parent-owned exclusions: Blade Mounting Sockets" in prompt
    assert "Do NOT model them" in prompt


def test_component_child_request_carries_structured_parent_ownership_exclusions(
    tmp_path, monkeypatch
):
    parent_id, root, feature = _parent_job(tmp_path, monkeypatch)
    plan = load_feature_plan(root)
    assert plan is not None
    plan.features.append(
        FeatureTask(
            id="wheel-opening",
            name="Wheel Arch Opening",
            strategy="surface_cutout",
            target_regions=["fender"],
            owner_scope=["wheel_arch_cutout"],
        )
    )
    save_feature_plan(root, plan)

    child_id = main._create_component_child_job(parent_id, feature)
    child_root = main.JOBS_ROOT / child_id
    child_request = json.loads((child_root / "request.json").read_text(encoding="utf-8"))

    exclusions = child_request["parent_owned_exclusions"]
    assert any(item["id"] == "wheel-opening" for item in exclusions)
    assert "Wheel Arch Opening" in child_request["prompt"]
    assert "context only" in child_request["prompt"]


def test_component_child_creation_respects_recursion_depth(tmp_path, monkeypatch):
    parent_id, _, feature = _parent_job(
        tmp_path,
        monkeypatch,
        depth=main.COMPONENT_MAX_DEPTH,
    )

    with pytest.raises(HTTPException) as exc:
        main._create_component_child_job(parent_id, feature)

    assert exc.value.status_code == 409
    assert "maximum" in str(exc.value.detail).lower()


def test_component_assembly_executor_imports_frozen_child_and_renders_parent():
    script = component_assembly_script()

    assert 'bpy.ops.wm.open_mainfile(filepath=PARENT_BLEND)' in script
    assert 'bpy.data.libraries.load(COMPONENT_BLEND, link=False)' in script
    assert 'clone.matrix_world = instance_matrix @ normalize @ source.matrix_world' in script
    assert '"front-right"' in script
    assert 'bpy.ops.wm.save_as_mainfile(filepath=BLEND)' in script
    assert '"installed_component": COMPONENT_NAME' in script



def test_parent_component_retry_reopens_failed_child_without_resetting_verified_siblings(
    tmp_path, monkeypatch
):
    _, root, _ = _parent_job(tmp_path, monkeypatch)
    child_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    child_root = main.JOBS_ROOT / child_id
    child_root.mkdir(parents=True)
    for category in main.ARTIFACT_CATEGORIES:
        (child_root / category).mkdir(exist_ok=True)
    (child_root / "request.json").write_text(
        json.dumps({"job_id": child_id, "prompt": "Isolated generic component"}),
        encoding="utf-8",
    )

    child_plan = FeaturePlan(
        features=[
            FeatureTask(
                id="verified-base",
                name="Verified base",
                status="accepted",
                accepted_version=2,
                acceptance_verified=True,
                acceptance_score=0.9,
            ),
            FeatureTask(
                id="failed-detail",
                name="Failed detail",
                status="failed",
                attempts=3,
                depends_on=["verified-base"],
            ),
            FeatureTask(
                id="dependent-finish",
                name="Dependent finish",
                status="blocked",
                depends_on=["failed-detail"],
            ),
        ]
    )
    save_feature_plan(child_root, child_plan)

    parent_plan = load_feature_plan(root)
    assert parent_plan is not None
    parent_feature = parent_plan.features[0]
    parent_feature.component_job_id = child_id
    parent_feature.status = "failed"
    save_feature_plan(root, parent_plan)

    reopened = main._requeue_linked_component_child(
        parent_feature,
        reset_attempts=True,
    )

    assert reopened == ["failed-detail"]
    persisted = load_feature_plan(child_root)
    assert persisted is not None
    verified, failed, dependent = persisted.features
    assert verified.status == "accepted"
    assert verified.acceptance_verified is True
    assert verified.accepted_version == 2
    assert failed.status == "retry"
    assert failed.attempts == 0
    assert dependent.status == "pending"
    child_status = main._read_status(child_root)
    assert child_status["state"] == "ready"
    assert child_status["stage"] == "component_feature_retry_ready"
