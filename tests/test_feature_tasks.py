from app.feature_tasks import (
    FeaturePlan,
    active_or_next_feature,
    begin_feature,
    finish_feature,
    invalidate_unverified_acceptances,
    load_feature_plan,
    normalize_feature_plan_payload,
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


def test_failed_feature_does_not_freeze_downstream_work(tmp_path):
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
            error="Candidate did not pass feature QA.",
        )

    persisted = load_feature_plan(tmp_path)
    assert persisted is not None
    body = next(feature for feature in persisted.features if feature.id == "body-shell")
    assert body.status == "blocked"

    next_task = active_or_next_feature(persisted)
    assert next_task is not None
    assert next_task.id in {"wheels", "windows"}


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
