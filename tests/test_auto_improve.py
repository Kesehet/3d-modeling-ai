import json

from app.feature_tasks import FeaturePlan, save_feature_plan
from app.main import _auto_improve_goal_reached, _auto_improve_progress_signature


def test_auto_improve_requires_quality_and_feature_completion(tmp_path):
    status = {
        "stage": "adaptive_mesh_recognizable",
        "quality_gate": {"recognizable": True, "subject_match_score": 0.91},
        "generic_model": {"version": 4},
    }

    plan = FeaturePlan.model_validate(
        {
            "subject": "car",
            "features": [
                {
                    "id": "body",
                    "name": "Body",
                    "priority": 10,
                    "status": "accepted",
                    "accepted_version": 4,
                    "acceptance_verified": True,
                    "acceptance_score": 0.90,
                },
                {
                    "id": "wheels",
                    "name": "Wheels",
                    "priority": 9,
                    "status": "ready",
                    "depends_on": ["body"],
                },
            ],
        }
    )
    save_feature_plan(tmp_path, plan)

    assert _auto_improve_goal_reached(tmp_path, status) is False

    plan.features[1].status = "accepted"
    plan.features[1].accepted_version = 4
    plan.features[1].acceptance_verified = True
    plan.features[1].acceptance_score = 0.90
    save_feature_plan(tmp_path, plan)

    assert _auto_improve_goal_reached(tmp_path, status) is True


def test_auto_improve_progress_signature_tracks_model_and_feature_progress(tmp_path):
    status = {
        "stage": "adaptive_mesh_needs_refinement",
        "quality_gate": {"recognizable": False, "subject_match_score": 0.42},
        "generic_model": {"version": 2},
    }
    plan = FeaturePlan.model_validate(
        {
            "subject": "car",
            "features": [
                {
                    "id": "body",
                    "name": "Body",
                    "priority": 10,
                    "status": "accepted",
                    "accepted_version": 2,
                    "acceptance_verified": True,
                    "acceptance_score": 0.90,
                },
                {
                    "id": "windows",
                    "name": "Windows",
                    "priority": 8,
                    "status": "ready",
                    "depends_on": ["body"],
                },
            ],
        }
    )
    save_feature_plan(tmp_path, plan)

    before = _auto_improve_progress_signature(tmp_path, status)

    status["generic_model"]["version"] = 3
    status["quality_gate"]["subject_match_score"] = 0.56
    plan.features[1].status = "accepted"
    plan.features[1].accepted_version = 3
    plan.features[1].acceptance_verified = True
    plan.features[1].acceptance_score = 0.90
    save_feature_plan(tmp_path, plan)

    after = _auto_improve_progress_signature(tmp_path, status)

    assert before != after


def test_auto_improve_without_feature_plan_stops_when_recognizable(tmp_path):
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "stage": "generic_initial_recognizable",
                "quality_gate": {"recognizable": True},
            }
        ),
        encoding="utf-8",
    )

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert _auto_improve_goal_reached(tmp_path, status) is True



def test_auto_improve_does_not_call_blocked_required_features_complete(tmp_path):
    status = {
        "stage": "adaptive_mesh_recognizable",
        "quality_gate": {"recognizable": True, "subject_match_score": 0.88},
        "generic_model": {"version": 5},
    }
    plan = FeaturePlan.model_validate(
        {
            "subject": "car",
            "features": [
                {
                    "id": "body",
                    "name": "Body",
                    "required": True,
                    "priority": 10,
                    "status": "accepted",
                    "accepted_version": 5,
                    "acceptance_verified": True,
                    "acceptance_score": 0.90,
                },
                {
                    "id": "wheels",
                    "name": "Wheels",
                    "required": True,
                    "priority": 9,
                    "status": "blocked",
                    "attempts": 3,
                    "depends_on": ["body"],
                },
            ],
        }
    )
    save_feature_plan(tmp_path, plan)

    assert _auto_improve_goal_reached(tmp_path, status) is False
