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
    assert "attachment/mounting anchor exactly at Blender world origin [0, 0, 0]" in child_request["prompt"]

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
    assert 'source_world_matrices' in script
    assert 'bpy.context.view_layer.update()' in script
    assert 'clone.parent = None' in script
    assert 'clone.matrix_world = instance_matrix @ source_world_matrices[source.name]' in script
    assert 'component_center' not in script
    assert '@ normalize @' not in script
    assert '"assembly_anchor": "component_global_origin"' in script
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



def _touch_preserved_model(root, version):
    (root / "scene" / f"model-v{version}.blend").write_bytes(b"blend")
    for view in (
        "front", "front-left", "left", "back-left", "back",
        "back-right", "right", "front-right", "top",
    ):
        (root / "renders" / f"model-v{version}-{view}.png").write_bytes(b"render")
    (root / f"scene-spec-v{version}.json").write_text(
        json.dumps({"version": version, "spec": {"title": "parent", "objects": []}}),
        encoding="utf-8",
    )


def test_component_assembly_chain_unwinds_consecutive_installs(tmp_path, monkeypatch):
    _, root, _ = _parent_job(tmp_path, monkeypatch)
    (root / "component-assembly-v6.json").write_text(
        json.dumps({
            "baseline_version": 4,
            "candidate_version": 6,
            "feature_id": "first-component",
            "child_job_id": "child-one",
            "assembly": {"instances": [{"location": [0, 0, 0]}]},
        }),
        encoding="utf-8",
    )
    (root / "component-assembly-v7.json").write_text(
        json.dumps({
            "baseline_version": 6,
            "candidate_version": 7,
            "feature_id": "second-component",
            "child_job_id": "child-two",
            "assembly": {"instances": [{"location": [1, 0, 0]}]},
        }),
        encoding="utf-8",
    )

    source, chain = main._component_assembly_chain(root, 7)

    assert source == 4
    assert [item["feature_id"] for item in chain] == [
        "first-component",
        "second-component",
    ]


def test_assembly_repair_sequences_selected_ancestor_and_dependent_interface():
    plan = FeaturePlan(
        subject="Generic assembly",
        features=[
            FeatureTask(
                id="body",
                name="Primary body",
                build_mode="in_place",
                status="accepted",
                accepted_version=3,
                acceptance_verified=True,
            ),
            FeatureTask(
                id="mount",
                name="Mounting interface",
                build_mode="in_place",
                parent="body",
                status="accepted",
                accepted_version=4,
                acceptance_verified=True,
            ),
            FeatureTask(
                id="detail",
                name="Independent detail",
                build_mode="in_place",
                status="accepted",
                accepted_version=4,
                acceptance_verified=True,
            ),
        ],
    )

    active, deferred = main._sequence_assembly_repair_parent_features(
        plan,
        ["body", "mount", "detail"],
    )

    assert active == ["body", "detail"]
    assert deferred == ["mount"]


def test_assembly_repair_keeps_descendant_active_when_ancestor_not_selected():
    plan = FeaturePlan(
        subject="Generic assembly",
        features=[
            FeatureTask(id="body", name="Body", build_mode="in_place"),
            FeatureTask(id="mount", name="Mount", build_mode="in_place", parent="body"),
        ],
    )

    active, deferred = main._sequence_assembly_repair_parent_features(plan, ["mount"])

    assert active == ["mount"]
    assert deferred == []


def test_coordinated_assembly_repair_reopens_parent_before_frozen_component(
    tmp_path, monkeypatch
):
    parent_id, root, _ = _parent_job(tmp_path, monkeypatch)
    plan = load_feature_plan(root)
    assert plan is not None
    component = plan.features[-1]
    parent_feature = FeatureTask(
        id="mount",
        name="Mounting interface",
        build_mode="in_place",
        strategy="surface_cutout",
        status="accepted",
        accepted_version=4,
        acceptance_verified=True,
        acceptance_score=0.95,
    )
    plan.features.insert(0, parent_feature)
    component.status = "accepted"
    component.accepted_version = 5
    component.acceptance_verified = True
    component.acceptance_score = 0.95
    component.component_job_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    save_feature_plan(root, plan)

    child_root = main.JOBS_ROOT / component.component_job_id
    child_root.mkdir(parents=True)
    for category in main.ARTIFACT_CATEGORIES:
        (child_root / category).mkdir(exist_ok=True)

    _touch_preserved_model(root, 4)
    _touch_preserved_model(root, 5)
    (root / "component-assembly-v5.json").write_text(
        json.dumps({
            "baseline_version": 4,
            "candidate_version": 5,
            "feature_id": component.id,
            "child_job_id": component.component_job_id,
            "assembly": {
                "rationale": "old placement",
                "instances": [{
                    "location": [1, 0, 0],
                    "rotation_deg": [0, 0, 0],
                    "scale": [1, 1, 1],
                }],
            },
        }),
        encoding="utf-8",
    )
    main._write_status(
        root,
        state="ready",
        modeling_strategy="procedural",
        generic_model={
            "version": 5,
            "title": "Assembled object",
            "blend": "model-v5.blend",
            "renders": [],
            "qa": "",
            "assembled_components": [{"feature_id": component.id}],
        },
        quality_gate={
            "scope": "whole_object",
            "evaluated_version": 5,
            "recognizable": False,
            "recognition_passed": True,
            "assembly_integrity_pass": False,
            "blocking_geometry_defects": ["Visible bad contact"],
            "instructions": ["Repair the mounting contact."],
        },
    )

    async def plan_repair(*args, **kwargs):
        return (
            main.AssemblyRepairDecision(
                rationale="The parent mounting interface also needs correction.",
                parent_feature_ids=["mount"],
            ),
            [],
        )

    monkeypatch.setattr(main, "_plan_assembled_parent_repair", plan_repair)
    prepared = __import__("asyncio").run(
        main._prepare_assembled_parent_repair(parent_id, main._read_status(root))
    )

    assert prepared["prepared"] is True
    assert prepared["source_parent_version"] == 4
    status = main._read_status(root)
    assert status["generic_model"]["version"] == 4
    assert status["assembly_repair"]["phase"] == "repair_parent"
    repaired_plan = load_feature_plan(root)
    assert repaired_plan is not None
    by_id = {feature.id: feature for feature in repaired_plan.features}
    assert by_id["mount"].status == "retry"
    assert by_id[component.id].status == "accepted"

    by_id["mount"].status = "accepted"
    by_id["mount"].accepted_version = 6
    by_id["mount"].acceptance_verified = True
    save_feature_plan(root, repaired_plan)
    advanced = main._advance_assembly_repair_if_ready(root, main._read_status(root))

    assert advanced["assembly_repair"]["phase"] == "reinstall_components"
    reloaded = load_feature_plan(root)
    assert reloaded is not None
    by_id = {feature.id: feature for feature in reloaded.features}
    assert by_id[component.id].status == "retry"
    assert by_id[component.id].component_job_id == component.component_job_id



def test_deferred_parent_repair_wave_continues_without_spending_global_attempt(
    tmp_path, monkeypatch
):
    _, root, _ = _parent_job(tmp_path, monkeypatch)
    plan = load_feature_plan(root)
    assert plan is not None
    component = plan.features[-1]
    body = FeatureTask(
        id="body",
        name="Primary body",
        build_mode="in_place",
        status="accepted",
        accepted_version=6,
        acceptance_verified=True,
        acceptance_score=0.95,
    )
    mount = FeatureTask(
        id="mount",
        name="Dependent mounting interface",
        build_mode="in_place",
        parent="body",
        status="accepted",
        accepted_version=6,
        acceptance_verified=True,
        acceptance_score=0.95,
    )
    plan.features.insert(0, body)
    plan.features.insert(1, mount)
    component.status = "accepted"
    component.accepted_version = 7
    component.acceptance_verified = True
    component.acceptance_score = 0.95
    component.component_job_id = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
    save_feature_plan(root, plan)

    child_root = main.JOBS_ROOT / component.component_job_id
    child_root.mkdir(parents=True)
    for category in main.ARTIFACT_CATEGORIES:
        (child_root / category).mkdir(exist_ok=True)

    status = main._write_status(
        root,
        state="ready",
        assembly_repair={
            "attempt": main.ASSEMBLY_REPAIR_MAX_ATTEMPTS,
            "phase": "repair_parent",
            "repair_wave": 1,
            "failed_version": 7,
            "source_parent_version": 4,
            "parent_feature_ids": ["body"],
            "deferred_parent_feature_ids": ["mount"],
            "component_feature_ids": [component.id],
        },
    )

    advanced = main._advance_assembly_repair_if_ready(root, status)

    assert advanced["assembly_repair"]["attempt"] == main.ASSEMBLY_REPAIR_MAX_ATTEMPTS
    assert advanced["assembly_repair"]["phase"] == "repair_parent"
    assert advanced["assembly_repair"]["repair_wave"] == 2
    assert advanced["assembly_repair"]["parent_feature_ids"] == ["mount"]
    assert advanced["assembly_repair"]["deferred_parent_feature_ids"] == []

    reloaded = load_feature_plan(root)
    assert reloaded is not None
    by_id = {feature.id: feature for feature in reloaded.features}
    assert by_id["mount"].status == "retry"
    assert by_id[component.id].status == "accepted"

    by_id["mount"].status = "accepted"
    by_id["mount"].accepted_version = 8
    by_id["mount"].acceptance_verified = True
    save_feature_plan(root, reloaded)

    finished = main._advance_assembly_repair_if_ready(root, main._read_status(root))

    assert finished["assembly_repair"]["attempt"] == main.ASSEMBLY_REPAIR_MAX_ATTEMPTS
    assert finished["assembly_repair"]["phase"] == "reinstall_components"
    final_plan = load_feature_plan(root)
    assert final_plan is not None
    final_by_id = {feature.id: feature for feature in final_plan.features}
    assert final_by_id[component.id].status == "retry"

def test_coordinated_assembly_repair_can_reinstall_without_parent_rebuild(
    tmp_path, monkeypatch
):
    parent_id, root, _ = _parent_job(tmp_path, monkeypatch)
    plan = load_feature_plan(root)
    assert plan is not None
    component = plan.features[-1]
    component.status = "accepted"
    component.accepted_version = 5
    component.acceptance_verified = True
    component.acceptance_score = 0.95
    component.component_job_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    save_feature_plan(root, plan)
    child_root = main.JOBS_ROOT / component.component_job_id
    child_root.mkdir(parents=True)
    for category in main.ARTIFACT_CATEGORIES:
        (child_root / category).mkdir(exist_ok=True)

    _touch_preserved_model(root, 4)
    _touch_preserved_model(root, 5)
    (root / "component-assembly-v5.json").write_text(
        json.dumps({
            "baseline_version": 4,
            "candidate_version": 5,
            "feature_id": component.id,
            "child_job_id": component.component_job_id,
            "assembly": {
                "instances": [{
                    "location": [2, 0, 0],
                    "rotation_deg": [0, 0, 0],
                    "scale": [1, 1, 1],
                }]
            },
        }),
        encoding="utf-8",
    )
    main._write_status(
        root,
        generic_model={
            "version": 5,
            "title": "Object",
            "blend": "model-v5.blend",
            "assembled_components": [{"feature_id": component.id}],
        },
        modeling_strategy="procedural",
        quality_gate={
            "recognizable": False,
            "recognition_passed": True,
            "assembly_integrity_pass": False,
            "blocking_geometry_defects": ["Only the component placement is wrong"],
        },
    )

    async def plan_repair(*args, **kwargs):
        return (
            main.AssemblyRepairDecision(
                rationale="Parent geometry is sound; only reinstall the frozen component.",
                parent_feature_ids=[],
            ),
            [],
        )

    monkeypatch.setattr(main, "_plan_assembled_parent_repair", plan_repair)
    prepared = __import__("asyncio").run(
        main._prepare_assembled_parent_repair(parent_id, main._read_status(root))
    )

    assert prepared["prepared"] is True
    status = main._read_status(root)
    assert status["assembly_repair"]["phase"] == "reinstall_components"
    assert status["assembly_repair"]["previous_assemblies"][component.id]["instances"][0]["location"] == [2, 0, 0]
    reloaded = load_feature_plan(root)
    assert reloaded is not None
    retried = next(feature for feature in reloaded.features if feature.id == component.id)
    assert retried.status == "retry"
    assert retried.acceptance_verified is False



def test_selected_parent_feature_cannot_be_reaccepted_unchanged_during_assembly_repair():
    feature = FeatureTask(id="joint", name="Joint", build_mode="in_place")
    status = {
        "assembly_repair": {
            "phase": "repair_parent",
            "parent_feature_ids": ["joint"],
        }
    }

    assert main._assembly_repair_forces_feature_edit(status, feature) is True
    assert main._assembly_repair_forces_feature_edit(status, FeatureTask(id="other", name="Other")) is False
    assert main._assembly_repair_forces_feature_edit({}, feature) is False


def test_unrecognizable_assembled_model_can_retry_frozen_component_transforms(
    tmp_path, monkeypatch
):
    parent_id, root, _ = _parent_job(tmp_path, monkeypatch)
    plan = load_feature_plan(root)
    assert plan is not None
    component = plan.features[-1]
    component.status = "accepted"
    component.accepted_version = 6
    component.acceptance_verified = True
    component.acceptance_score = 1.0
    component.component_job_id = "dddddddd-dddd-dddd-dddd-dddddddddddd"
    save_feature_plan(root, plan)

    child_root = main.JOBS_ROOT / component.component_job_id
    child_root.mkdir(parents=True)
    for category in main.ARTIFACT_CATEGORIES:
        (child_root / category).mkdir(exist_ok=True)

    _touch_preserved_model(root, 4)
    _touch_preserved_model(root, 6)
    (root / "component-assembly-v6.json").write_text(
        json.dumps({
            "baseline_version": 4,
            "candidate_version": 6,
            "feature_id": component.id,
            "child_job_id": component.component_job_id,
            "assembly": {
                "instances": [{
                    "location": [0.25, 0, 0.5],
                    "rotation_deg": [0, 0, 0],
                    "scale": [1, 1, 1],
                }]
            },
        }),
        encoding="utf-8",
    )
    main._write_status(
        root,
        generic_model={
            "version": 6,
            "title": "Assembled object",
            "blend": "model-v6.blend",
            "assembled_components": [{"feature_id": component.id}],
        },
        modeling_strategy="procedural",
        quality_gate={
            "scope": "whole_object",
            "evaluated_version": 6,
            "recognizable": False,
            "recognition_passed": False,
            "assembly_integrity_pass": None,
            "subject_match_score": 0.6,
            "director_action": "refine_mesh",
            "major_missing_parts": ["Installed component is visibly too thin."],
            "instructions": ["Increase installed component width/scale and clean up contact."],
        },
        assembly_repair={"attempt": 1, "phase": "reinstall_components"},
    )

    async def plan_repair(*args, **kwargs):
        return (
            main.AssemblyRepairDecision(
                rationale="Parent is acceptable; revise only frozen-component installation.",
                parent_feature_ids=[],
            ),
            [],
        )

    monkeypatch.setattr(main, "_plan_assembled_parent_repair", plan_repair)
    prepared = __import__("asyncio").run(
        main._prepare_assembled_parent_repair(parent_id, main._read_status(root))
    )

    assert prepared["prepared"] is True
    assert prepared["attempt"] == 2
    status = main._read_status(root)
    assert status["generic_model"]["version"] == 4
    assert status["assembly_repair"]["phase"] == "reinstall_components"
    assert "Installed component is visibly too thin." in status["assembly_repair"]["blocking_defects"]
    reloaded = load_feature_plan(root)
    assert reloaded is not None
    retried = next(feature for feature in reloaded.features if feature.id == component.id)
    assert retried.status == "retry"



def test_component_local_axis_context_exposes_xyz_scale_contract():
    context = main._component_local_axis_context(
        {"scene_dimensions_blender_units": [0.4, 2.5, 0.08]}
    )

    assert context["dimensions_xyz"] == [0.4, 2.5, 0.08]
    assert context["largest_extent_axis"] == "Y"
    assert context["middle_extent_axis"] == "X"
    assert context["smallest_extent_axis"] == "Z"
    assert "BEFORE instance rotation" in context["scale_contract"]


def test_component_local_axis_context_rejects_missing_or_degenerate_bounds():
    assert main._component_local_axis_context({}) == {}
    assert main._component_local_axis_context(
        {"scene_dimensions_blender_units": [1.0, 0.0, 2.0]}
    ) == {}


def test_coordinated_repair_remains_bounded_after_axis_aware_retry_extension():
    assert main.ASSEMBLY_REPAIR_MAX_ATTEMPTS == 5



def test_parent_assembly_repair_includes_failed_assembled_views(tmp_path, monkeypatch):
    _, root, _ = _parent_job(tmp_path, monkeypatch)
    for view in ("front", "front-left", "left", "front-right", "top"):
        (root / "renders" / f"model-v16-{view}.png").write_bytes(b"render")

    paths = main._assembly_repair_failed_render_paths(
        root,
        {
            "assembly_repair": {
                "phase": "repair_parent",
                "failed_version": 16,
            }
        },
        render_version=15,
    )

    assert [path.name for path in paths] == [
        "model-v16-front.png",
        "model-v16-front-left.png",
        "model-v16-left.png",
        "model-v16-front-right.png",
        "model-v16-top.png",
    ]
    assert main._assembly_repair_failed_render_paths(
        root,
        {"assembly_repair": {"phase": "reinstall_components", "failed_version": 16}},
        render_version=15,
    ) == []


def test_repair_director_context_fix_gets_one_bounded_retry():
    assert main.ASSEMBLY_REPAIR_MAX_ATTEMPTS == 5


@pytest.mark.parametrize("declared_symmetry", ["radial", "none"])
def test_radial_component_contract_aligns_long_axis_and_equalizes_instances(declared_symmetry):
    feature = FeatureTask(
        id="repeated-part",
        name="Repeated elongated part",
        count=3,
        symmetry=declared_symmetry,
        build_mode="component_job",
    )
    assembly = main.ComponentAssemblySpec(
        rationale="Place three repeated parts radially.",
        repeated_axis_alignment="radial",
        instances=[
            main.ComponentInstanceSpec(
                location=[0.12, 0.0, -0.31],
                rotation_deg=[0.0, 90.0, 0.0],
                scale=[0.05, 0.05, 0.05],
            ),
            main.ComponentInstanceSpec(
                location=[-0.06, 0.1039, -0.31],
                rotation_deg=[0.0, 210.0, 0.0],
                scale=[0.06, 0.06, 0.06],
            ),
            main.ComponentInstanceSpec(
                location=[-0.06, -0.1039, -0.31],
                rotation_deg=[0.0, 330.0, 0.0],
                scale=[0.07, 0.07, 0.07],
            ),
        ],
    )

    result = main._normalize_component_assembly_symmetry(
        assembly,
        feature,
        {
            "dimensions_xyz": [0.5958, 2.1998, 0.0911],
            "largest_extent_axis": "Y",
            "smallest_extent_axis": "Z",
        },
    )

    assert all(instance.scale == [0.05, 0.05, 0.05] for instance in result.instances)
    assert result.instances[0].rotation_deg == pytest.approx([0.0, 0.0, -90.0])
    assert result.instances[1].rotation_deg == pytest.approx([0.0, 0.0, 30.0], abs=0.1)
    assert result.instances[2].rotation_deg == pytest.approx([0.0, 0.0, 150.0], abs=0.1)
    assert result.instances[0].location == pytest.approx([0.12, 0.0, -0.31], abs=1e-4)
    assert result.instances[1].location == pytest.approx([-0.06, 0.103923, -0.31], abs=1e-4)
    assert result.instances[2].location == pytest.approx([-0.06, -0.103923, -0.31], abs=1e-4)


def test_tangential_radial_component_contract_uses_quarter_turn_offset():
    feature = FeatureTask(
        id="repeated-part",
        name="Repeated elongated part",
        count=2,
        symmetry="radial",
        build_mode="component_job",
    )
    assembly = main.ComponentAssemblySpec(
        repeated_axis_alignment="tangential",
        instances=[
            main.ComponentInstanceSpec(location=[2.0, 0.0, 1.0]),
            main.ComponentInstanceSpec(location=[-2.0, 0.0, 1.0]),
        ],
    )

    result = main._normalize_component_assembly_symmetry(
        assembly,
        feature,
        {"dimensions_xyz": [0.4, 3.0, 0.1]},
    )

    assert result.instances[0].rotation_deg == pytest.approx([0.0, 0.0, 0.0])
    assert abs(abs(result.instances[1].rotation_deg[2]) - 180.0) < 1e-6


def test_free_component_alignment_preserves_ai_transforms():
    feature = FeatureTask(
        id="free-part",
        name="Free repeated part",
        count=3,
        symmetry="radial",
        build_mode="component_job",
    )
    assembly = main.ComponentAssemblySpec(
        repeated_axis_alignment="free",
        instances=[
            main.ComponentInstanceSpec(location=[1, 0, 0], rotation_deg=[1, 2, 3]),
            main.ComponentInstanceSpec(location=[0, 1, 0], rotation_deg=[4, 5, 6]),
            main.ComponentInstanceSpec(location=[-1, 0, 0], rotation_deg=[7, 8, 9]),
        ],
    )

    result = main._normalize_component_assembly_symmetry(
        assembly,
        feature,
        {"dimensions_xyz": [0.4, 3.0, 0.1]},
    )

    assert result.model_dump() == assembly.model_dump()
