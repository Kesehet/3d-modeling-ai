from app.feature_tasks import (
    FeaturePlan,
    active_or_next_feature,
    begin_feature,
    finish_feature,
    invalidate_unverified_acceptances,
    link_component_job,
    load_feature_plan,
    mark_component_ready,
    normalize_feature_plan_payload,
    record_feature_progress,
    requeue_failed_structural_feature_as_component,
    retry_feature,
    save_feature_plan,
)


def _car_plan() -> FeaturePlan:
    payload = normalize_feature_plan_payload(
        {
            "subject": "Toyota Prius",
            "features": [
                {
                    "id": "body-shell",
                    "name": "Main body shell",
                    "priority": 10,
                    "strategy": "base_mesh_region",
                    "target_regions": ["whole body"],
                    "acceptance_criteria": ["Prius-like overall silhouette is visible"],
                },
                {
                    "id": "wheels",
                    "name": "Wheels",
                    "priority": 9,
                    "strategy": "attachment",
                    "depends_on": ["Main body shell"],
                    "count": 4,
                    "acceptance_criteria": ["Four wheels are visibly placed at the axles"],
                },
                {
                    "id": "windows",
                    "name": "Windows",
                    "priority": 8,
                    "strategy": "surface_cutout",
                    "depends_on": ["body-shell"],
                    "acceptance_criteria": ["Windshield and side glazing read clearly"],
                },
            ],
        },
        subject="Toyota Prius",
    )
    return FeaturePlan.model_validate(payload)


def test_normalizer_resolves_named_dependencies():
    plan = _car_plan()

    assert plan.features[1].depends_on == ["body-shell"]
    assert plan.features[2].depends_on == ["body-shell"]
    assert plan.features[0].status == "pending"


def test_feature_subjobs_follow_dependencies(tmp_path):
    plan = _car_plan()
    save_feature_plan(tmp_path, plan)

    first = begin_feature(tmp_path)
    assert first is not None
    assert first.id == "body-shell"
    assert first.status == "running"

    finish_feature(
        tmp_path,
        first.id,
        accepted=True,
        version=2,
        summary="Body silhouette matches the references.",
        verified=True,
        acceptance_score=0.91,
        acceptance_model="test-vision",
    )

    persisted = load_feature_plan(tmp_path)
    assert persisted is not None
    body = next(feature for feature in persisted.features if feature.id == "body-shell")
    assert body.status == "accepted"
    assert body.accepted_version == 2

    next_task = active_or_next_feature(persisted)
    assert next_task is not None
    assert next_task.id == "wheels"


def test_failed_required_feature_blocks_dependent_work(tmp_path):
    plan = _car_plan()
    save_feature_plan(tmp_path, plan)

    for _ in range(3):
        task = begin_feature(tmp_path)
        assert task is not None
        assert task.id == "body-shell"
        finish_feature(
            tmp_path,
            task.id,
            accepted=False,
            version=None,
            error="Candidate did not pass strict feature QA.",
        )

    persisted = load_feature_plan(tmp_path)
    assert persisted is not None
    body = next(feature for feature in persisted.features if feature.id == "body-shell")
    assert body.status == "failed"

    next_task = active_or_next_feature(persisted)
    assert next_task is None

    wheels = next(feature for feature in persisted.features if feature.id == "wheels")
    windows = next(feature for feature in persisted.features if feature.id == "windows")
    assert wheels.status == "blocked"
    assert windows.status == "blocked"
    assert "dependency failed" in wheels.last_error.lower()


def test_dependency_cycle_is_recovered_instead_of_deadlocking():
    payload = normalize_feature_plan_payload(
        {
            "features": [
                {
                    "id": "a",
                    "name": "Feature A",
                    "priority": 10,
                    "depends_on": ["b"],
                },
                {
                    "id": "b",
                    "name": "Feature B",
                    "priority": 9,
                    "depends_on": ["a"],
                },
            ]
        },
        subject="test",
    )
    plan = FeaturePlan.model_validate(payload)

    task = active_or_next_feature(plan)

    assert task is not None
    assert task.id == "a"
    assert "deadlock" in task.last_error.lower()



def test_feature_queue_preserves_authored_build_order_over_numeric_priority():
    payload = normalize_feature_plan_payload(
        {
            "subject": "Bugatti Veyron",
            "features": [
                {
                    "id": "body",
                    "name": "Primary Body Silhouette",
                    "priority": 1,
                    "strategy": "base_mesh_region",
                },
                {
                    "id": "grille",
                    "name": "Front Horseshoe Grille",
                    "priority": 10,
                    "strategy": "attachment",
                },
            ],
        },
        subject="Bugatti Veyron",
    )
    plan = FeaturePlan.model_validate(payload)

    task = active_or_next_feature(plan)

    assert task is not None
    assert task.id == "body"



def test_unverified_acceptance_is_not_accepted(tmp_path):
    plan = _car_plan()
    save_feature_plan(tmp_path, plan)
    task = begin_feature(tmp_path)
    assert task is not None

    finish_feature(
        tmp_path,
        task.id,
        accepted=True,
        version=2,
        summary="It got somewhat better, but no strict QA was performed.",
    )

    persisted = load_feature_plan(tmp_path)
    assert persisted is not None
    body = next(feature for feature in persisted.features if feature.id == "body-shell")
    assert body.status == "retry"
    assert body.accepted_version is None
    assert body.acceptance_verified is False


def test_legacy_accepted_features_are_requeued(tmp_path):
    plan = _car_plan()
    body = plan.features[0]
    body.status = "accepted"
    body.accepted_version = 2
    body.acceptance_verified = False
    save_feature_plan(tmp_path, plan)

    reset = invalidate_unverified_acceptances(tmp_path)

    assert reset == 1
    persisted = load_feature_plan(tmp_path)
    assert persisted is not None
    body = next(feature for feature in persisted.features if feature.id == "body-shell")
    assert body.status == "retry"
    assert body.accepted_version is None
    assert "strict" in body.last_error.lower()



def test_feature_name_aliases_are_preserved():
    payload = normalize_feature_plan_payload(
        {
            "subject": "Volkswagen Polo",
            "features": [
                {
                    "id": "body",
                    "feature_name": "Primary Body Silhouette",
                    "priority": 10,
                    "strategy": "base_mesh_region",
                },
                {
                    "id": "headlights",
                    "part_name": "Headlight Assemblies",
                    "priority": 8,
                    "depends_on": ["body"],
                },
            ],
        },
        subject="Volkswagen Polo",
    )
    plan = FeaturePlan.model_validate(payload)

    assert plan.features[0].name == "Primary Body Silhouette"
    assert plan.features[1].name == "Headlight Assemblies"



def test_component_job_mode_and_assembly_metadata_are_normalized():
    payload = normalize_feature_plan_payload(
        {
            "subject": "Toyota Prius",
            "features": [
                {
                    "id": "body",
                    "name": "Body shell",
                    "build_mode": "in_place",
                },
                {
                    "id": "wheel",
                    "name": "Wheel assembly",
                    "build_mode": "recursive",
                    "count": 4,
                    "depends_on": ["body"],
                    "assembly_anchor": "four wheel centers at front/rear axles",
                    "assembly_notes": ["Reuse one accepted wheel for all four positions"],
                },
            ],
        },
        subject="Toyota Prius",
    )
    plan = FeaturePlan.model_validate(payload)

    assert plan.plan_version == 3
    wheel = next(feature for feature in plan.features if feature.id == "wheel")
    assert wheel.build_mode == "component_job"
    assert wheel.count == 4
    assert "wheel centers" in wheel.assembly_anchor
    assert wheel.assembly_notes == ["Reuse one accepted wheel for all four positions"]


def test_frozen_component_is_scheduled_for_installation_without_spending_new_build_attempt(tmp_path):
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
                    }
                ],
            },
            subject="Toyota Prius",
        )
    )
    save_feature_plan(tmp_path, plan)

    first = begin_feature(tmp_path)
    assert first is not None
    assert first.attempts == 1

    link_component_job(
        tmp_path,
        "wheel",
        component_job_id="11111111-1111-1111-1111-111111111111",
        component_depth=1,
    )
    mark_component_ready(
        tmp_path,
        "wheel",
        component_version=3,
        component_artifact="child/scene/model-v3.blend",
        summary="Child passed strict QA.",
    )

    persisted = load_feature_plan(tmp_path)
    assert persisted is not None
    wheel = persisted.features[0]
    assert wheel.status == "component_ready"
    assert wheel.component_job_id is not None
    assert wheel.component_version == 3

    install = begin_feature(tmp_path)
    assert install is not None
    assert install.id == "wheel"
    assert install.status == "running"
    assert install.attempts == 1



def test_in_place_parent_work_is_scheduled_before_component_installation():
    payload = normalize_feature_plan_payload(
        {
            "subject": "car",
            "features": [
                {
                    "id": "wheel",
                    "name": "Wheel assembly",
                    "build_mode": "component_job",
                    "priority": 10,
                },
                {
                    "id": "windows",
                    "name": "Window surfaces",
                    "build_mode": "in_place",
                    "priority": 5,
                },
            ],
        },
        subject="car",
    )
    plan = FeaturePlan.model_validate(payload)

    task = active_or_next_feature(plan)

    assert task is not None
    assert task.id == "windows"



def test_feature_plan_normalizer_accepts_nested_plan_wrapper():
    payload = normalize_feature_plan_payload(
        {
            "result": {
                "feature_plan": {
                    "subject": "Toyota Prius",
                    "notes": "Build the shell before isolated assemblies.",
                    "components": [
                        {
                            "id": "body",
                            "name": "Primary body shell",
                            "build_mode": "in_place",
                        },
                        {
                            "id": "wheel",
                            "name": "Wheel assembly",
                            "build_mode": "component_job",
                            "count": 4,
                            "depends_on": ["body"],
                        },
                    ],
                }
            }
        },
        subject="fallback subject",
    )

    plan = FeaturePlan.model_validate(payload)

    assert plan.plan_version == 3
    assert plan.subject == "Toyota Prius"
    assert "shell" in plan.coordinator_notes.lower()
    assert [feature.id for feature in plan.features] == ["body", "wheel"]
    assert plan.features[1].build_mode == "component_job"
    assert plan.features[1].depends_on == ["body"]


def test_feature_plan_normalizer_accepts_plan_list_wrapper_and_top_level_list():
    wrapped = normalize_feature_plan_payload(
        {
            "plan": [
                {"id": "body", "name": "Body shell"},
                {"id": "wheel", "name": "Wheel assembly", "execution_mode": "recursive"},
            ]
        },
        subject="Toyota Prius",
    )
    direct = normalize_feature_plan_payload(
        [
            {"id": "body", "name": "Body shell"},
            {"id": "wheel", "name": "Wheel assembly", "worker_mode": "component"},
        ],
        subject="Toyota Prius",
    )

    assert FeaturePlan.model_validate(wrapped).features[1].build_mode == "component_job"
    assert FeaturePlan.model_validate(direct).features[1].build_mode == "component_job"


def test_feature_plan_preserves_semantic_id_when_model_emits_generic_name():
    payload = normalize_feature_plan_payload(
        {
            "subject": "Toyota Prius",
            "features": [
                {
                    "id": "main-body-shell",
                    "name": "Feature 1",
                    "target_regions": ["body"],
                },
                {
                    "id": "wheel-assembly",
                    "name": "Feature 2",
                    "quantity": 4,
                    "build_mode": "component_job",
                },
            ],
        },
        subject="Toyota Prius",
    )
    plan = FeaturePlan.model_validate(payload)
    assert plan.features[0].name == "main body shell"
    assert plan.features[1].name == "wheel assembly"
    assert plan.features[1].count == 4


def test_bilateral_primary_base_mesh_is_one_feature_instance():
    payload = normalize_feature_plan_payload(
        {
            "features": [
                {
                    "id": "primary-shell",
                    "name": "Primary shell",
                    "strategy": "base_mesh_region",
                    "symmetry": "bilateral",
                },
                {
                    "id": "paired-detail",
                    "name": "Paired detail",
                    "strategy": "attachment",
                    "symmetry": "bilateral",
                },
            ]
        },
        subject="generic object",
    )
    plan = FeaturePlan.model_validate(payload)

    assert plan.features[0].count == 1
    assert plan.features[1].count == 2


def test_feature_plan_recovers_repeated_instance_intent():
    payload = normalize_feature_plan_payload(
        {
            "features": [
                {
                    "id": "wheel-assembly",
                    "name": "Feature 1",
                    "target_region": "all_four_wheels",
                },
                {
                    "id": "side-mirrors",
                    "name": "Feature 2",
                    "symmetry": "bilateral",
                },
            ]
        },
        subject="car",
    )
    plan = FeaturePlan.model_validate(payload)
    assert plan.features[0].count == 4
    assert plan.features[1].count == 2



def test_feature_can_keep_partial_progress_without_consuming_another_attempt(tmp_path):
    plan = _car_plan()
    save_feature_plan(tmp_path, plan)

    first = begin_feature(tmp_path)
    assert first is not None
    assert first.attempts == 1

    record_feature_progress(
        tmp_path,
        first.id,
        version=2,
        summary="Silhouette improved but still needs another visual correction.",
    )

    persisted = load_feature_plan(tmp_path)
    assert persisted is not None
    body = next(feature for feature in persisted.features if feature.id == first.id)
    assert body.status == "running"
    assert body.attempts == 1
    assert persisted.active_feature_id == first.id

    again = begin_feature(tmp_path)
    assert again is not None
    assert again.id == first.id
    assert again.attempts == 1



def test_retry_feature_reopens_failed_dependency_chain(tmp_path):
    plan = _car_plan()
    body = plan.features[0]
    body.status = "failed"
    body.attempts = 3
    body.last_error = "Old strict QA failure."
    plan.features[1].status = "blocked"
    plan.features[1].last_error = "Blocked because required dependency failed: body-shell"
    plan.features[2].status = "blocked"
    plan.features[2].last_error = "Blocked because required dependency failed: body-shell"
    save_feature_plan(tmp_path, plan)

    retried = retry_feature(tmp_path, "body-shell", reset_attempts=True)

    assert retried is not None
    body = next(feature for feature in retried.features if feature.id == "body-shell")
    wheels = next(feature for feature in retried.features if feature.id == "wheels")
    windows = next(feature for feature in retried.features if feature.id == "windows")
    assert body.status == "retry"
    assert body.attempts == 0
    assert body.last_error == ""
    assert wheels.status == "pending"
    assert windows.status == "pending"
    assert retried.active_feature_id is None



def test_exhausted_structural_part_gets_one_isolated_fallback(tmp_path):
    plan = _car_plan()
    body, wheels, windows = plan.features
    body.status = "accepted"
    body.acceptance_verified = True
    body.accepted_version = 1
    wheels.status = "failed"
    wheels.attempts = 3
    wheels.last_error = "The wheel geometry is missing."
    windows.status = "blocked"
    save_feature_plan(tmp_path, plan)

    recovered = requeue_failed_structural_feature_as_component(tmp_path)
    assert recovered is not None
    assert recovered.id == "wheels"
    assert recovered.build_mode == "component_job"
    assert recovered.component_fallback_attempted is True
    assert recovered.attempts == 0
    assert recovered.status == "retry"

    state = load_feature_plan(tmp_path)
    assert state is not None
    assert state.features[0].status == "accepted"
    assert state.features[0].accepted_version == 1
    assert state.features[2].status == "ready"
    assert active_or_next_feature(state).id == "wheels"

    # A failed isolated retry must not trigger a second new child forever.
    state.features[1].status = "failed"
    state.features[1].attempts = 3
    save_feature_plan(tmp_path, state)
    assert requeue_failed_structural_feature_as_component(tmp_path) is None


def test_fallback_does_not_convert_primary_shell_or_subtractive_features(tmp_path):
    plan = _car_plan()
    plan.features[0].status = "failed"
    plan.features[0].attempts = 3
    plan.features[1].status = "blocked"
    plan.features[2].status = "blocked"
    save_feature_plan(tmp_path, plan)
    assert requeue_failed_structural_feature_as_component(tmp_path) is None

    plan.features[0].status = "accepted"
    plan.features[0].acceptance_verified = True
    plan.features[0].accepted_version = 1
    plan.features[1].status = "accepted"
    plan.features[1].acceptance_verified = True
    plan.features[1].accepted_version = 2
    plan.features[2].status = "failed"
    plan.features[2].attempts = 3
    save_feature_plan(tmp_path, plan)
    assert requeue_failed_structural_feature_as_component(tmp_path) is None


def test_fallback_respects_unresolved_deps_and_attempt_budget(tmp_path):
    plan = _car_plan()
    plan.features[0].status = "accepted"
    plan.features[0].acceptance_verified = True
    plan.features[0].accepted_version = 1
    plan.features[1].status = "failed"
    plan.features[1].attempts = 2
    save_feature_plan(tmp_path, plan)
    assert requeue_failed_structural_feature_as_component(tmp_path) is None

    plan.features[1].attempts = 3
    plan.features[1].depends_on = ["windows"]
    save_feature_plan(tmp_path, plan)
    assert requeue_failed_structural_feature_as_component(tmp_path) is None
