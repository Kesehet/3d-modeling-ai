import json

from app.feature_tasks import FeaturePlan, save_feature_plan
from app.main import (
    _assembled_parent_requires_safe_stop,
    _auto_improve_goal_reached,
    _auto_improve_progress_signature,
    _feature_queue_is_blocked,
    _persisted_auto_improve_no_progress_rounds,
    _quality_snapshot,
    _compact_quality_gate,
    _remaining_feature_attempt_budget,
)


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




def test_auto_improve_progress_signature_tracks_working_cage_hill_climb(tmp_path):
    status = {
        "stage": "hard_surface_cage_needs_refinement",
        "quality_gate": {"recognizable": False, "subject_match_score": 0.18},
        "generic_model": {"version": 2},
        "working_cage_version": 5,
    }

    before = _auto_improve_progress_signature(tmp_path, status)
    status["working_cage_version"] = 6
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



def test_remaining_feature_budget_counts_only_runnable_unfinished_attempts(tmp_path):
    plan = FeaturePlan.model_validate(
        {
            "plan_version": 2,
            "subject": "car",
            "features": [
                {
                    "id": "body",
                    "name": "Body",
                    "required": True,
                    "status": "accepted",
                    "accepted_version": 2,
                    "acceptance_verified": True,
                    "acceptance_score": 0.9,
                },
                {
                    "id": "wheels",
                    "name": "Wheels",
                    "required": True,
                    "status": "retry",
                    "attempts": 1,
                    "depends_on": ["body"],
                },
                {
                    "id": "badge",
                    "name": "Badge",
                    "required": False,
                    "status": "failed",
                    "attempts": 3,
                },
            ],
        }
    )
    save_feature_plan(tmp_path, plan)

    assert _remaining_feature_attempt_budget(tmp_path) == 2


def test_feature_queue_reports_failed_required_dependency(tmp_path):
    plan = FeaturePlan.model_validate(
        {
            "plan_version": 2,
            "subject": "car",
            "features": [
                {
                    "id": "body",
                    "name": "Body",
                    "required": True,
                    "status": "failed",
                    "attempts": 3,
                },
                {
                    "id": "wheels",
                    "name": "Wheels",
                    "required": True,
                    "status": "pending",
                    "depends_on": ["body"],
                },
            ],
        }
    )
    save_feature_plan(tmp_path, plan)

    blocked, unresolved = _feature_queue_is_blocked(tmp_path)

    assert blocked is True
    assert "body" in unresolved
    assert "wheels" in unresolved



def test_assembled_parent_stops_before_destructive_global_rebuild(tmp_path):
    plan = FeaturePlan.model_validate(
        {
            "plan_version": 3,
            "subject": "Toyota Prius",
            "features": [
                {
                    "id": "body",
                    "name": "Body shell",
                    "required": True,
                    "status": "accepted",
                    "accepted_version": 2,
                    "acceptance_verified": True,
                    "acceptance_score": 0.9,
                },
                {
                    "id": "wheel",
                    "name": "Wheel assembly",
                    "required": True,
                    "build_mode": "component_job",
                    "status": "accepted",
                    "accepted_version": 3,
                    "acceptance_verified": True,
                    "acceptance_score": 0.9,
                },
            ],
        }
    )
    save_feature_plan(tmp_path, plan)
    status = {
        "stage": "component_assembly_accepted",
        "quality_gate": {"recognizable": False, "subject_match_score": 0.67},
        "generic_model": {
            "version": 3,
            "assembled_components": [
                {
                    "feature_id": "wheel",
                    "child_job_id": "child-wheel",
                    "parent_version": 3,
                }
            ],
        },
    }

    assert _assembled_parent_requires_safe_stop(tmp_path, status) is True

    status["quality_gate"]["recognizable"] = True
    assert _assembled_parent_requires_safe_stop(tmp_path, status) is False



def test_progress_signature_ignores_stage_only_churn(tmp_path):
    status = {
        "stage": "hard_surface_cage_needs_refinement",
        "generic_model": {"version": 4},
        "working_cage_version": 7,
    }
    before = _auto_improve_progress_signature(tmp_path, status)
    status["stage"] = "hard_surface_cage_needs_replan"
    after = _auto_improve_progress_signature(tmp_path, status)

    assert before == after


def test_persisted_no_progress_streak_survives_new_auto_runs(tmp_path):
    (tmp_path / "history.json").write_text(
        json.dumps(
            [
                {"event": "auto_improve_round_completed", "progress_changed": False},
                {"event": "auto_improve_stopped", "reason": "short run ended"},
                {"event": "auto_improve_started", "requested_rounds": 1},
                {"event": "auto_improve_round_completed", "progress_changed": False},
            ]
        ),
        encoding="utf-8",
    )

    assert _persisted_auto_improve_no_progress_rounds(tmp_path) == 2


def test_persisted_no_progress_streak_resets_after_accepted_geometry(tmp_path):
    (tmp_path / "history.json").write_text(
        json.dumps(
            [
                {"event": "auto_improve_round_completed", "progress_changed": False},
                {"event": "hard_surface_cage_progress", "candidate_version": 8},
                {"event": "auto_improve_round_completed", "progress_changed": True},
            ]
        ),
        encoding="utf-8",
    )

    assert _persisted_auto_improve_no_progress_rounds(tmp_path) == 0


def test_quality_gate_compaction_never_recurses_candidate_history():
    nested = {
        "summary": "current",
        "last_candidate_evaluation": {
            "summary": "latest rejected",
            "last_candidate_evaluation": {
                "summary": "older rejected",
            },
        },
    }

    compact = _compact_quality_gate(nested)
    assert compact["last_candidate_evaluation"]["summary"] == "latest rejected"
    assert "last_candidate_evaluation" not in compact["last_candidate_evaluation"]

    snapshot = _quality_snapshot(compact)
    assert "last_candidate_evaluation" not in snapshot
