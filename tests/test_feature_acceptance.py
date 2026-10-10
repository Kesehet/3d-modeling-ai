from app.feature_tasks import FeatureTask
from app.main import (
    _apply_preservation_audit,
    _feature_candidate_review_decision,
    _feature_diagnostic_views,
    _feature_evaluation_accepts,
)


def _body_task() -> FeatureTask:
    return FeatureTask(
        id="body",
        name="Primary Body Silhouette",
        category="primary_silhouette",
        priority=10,
        strategy="base_mesh_region",
        acceptance_criteria=["The body silhouette visibly matches the reference proportions."],
    )


def test_improvement_without_completion_does_not_accept_feature():
    task = _body_task()
    evaluation = {
        "passed": True,
        "visible": True,
        "criteria_satisfied": False,
        "subject_recognizable": True,
        "confidence": 0.95,
        "reference_match_score": 0.90,
        "regression_detected": False,
    }

    assert _feature_evaluation_accepts(task, evaluation) is False


def test_primary_silhouette_requires_high_absolute_reference_match():
    task = _body_task()
    evaluation = {
        "passed": True,
        "visible": True,
        "criteria_satisfied": True,
        "subject_recognizable": True,
        "confidence": 0.90,
        "reference_match_score": 0.78,
        "regression_detected": False,
    }

    assert _feature_evaluation_accepts(task, evaluation) is False


def test_strict_primary_silhouette_can_pass_when_all_gates_are_met():
    task = _body_task()
    evaluation = {
        "passed": True,
        "visible": True,
        "criteria_satisfied": True,
        "subject_recognizable": True,
        "confidence": 0.90,
        "reference_match_score": 0.88,
        "regression_detected": False,
    }

    assert _feature_evaluation_accepts(task, evaluation) is True



def test_feature_completion_ignores_model_pass_when_whole_subject_is_incomplete():
    task = FeatureTask(
        id="mount",
        name="Local mounting feature",
        strategy="attachment",
        acceptance_criteria=["The mounting feature visibly matches the reference."],
    )
    evaluation = {
        "passed": False,
        "visible": True,
        "criteria_satisfied": True,
        "subject_recognizable": False,
        "confidence": 0.96,
        "reference_match_score": 0.90,
        "regression_detected": False,
    }

    assert _feature_evaluation_accepts(task, evaluation) is True


def test_feature_completion_still_rejects_bad_local_reference_match():
    task = FeatureTask(
        id="mount",
        name="Local mounting feature",
        strategy="attachment",
        acceptance_criteria=["The mounting feature visibly matches the reference."],
    )
    evaluation = {
        "passed": False,
        "visible": True,
        "criteria_satisfied": True,
        "subject_recognizable": False,
        "confidence": 0.96,
        "reference_match_score": 0.55,
        "regression_detected": False,
    }

    assert _feature_evaluation_accepts(task, evaluation) is False


def test_generic_geometry_reference_does_not_veto_completed_local_feature():
    task = _body_task()
    evaluation = {
        "passed": False,
        "visible": True,
        "criteria_satisfied": True,
        "subject_recognizable": False,
        "confidence": 0.96,
        "reference_match_score": 0.10,
        "reference_match_required": False,
        "regression_detected": False,
    }

    assert _feature_evaluation_accepts(task, evaluation) is True


def test_generic_reference_never_bypasses_unmet_feature_criteria():
    task = _body_task()
    evaluation = {
        "passed": False,
        "visible": True,
        "criteria_satisfied": False,
        "subject_recognizable": False,
        "confidence": 0.96,
        "reference_match_score": 0.95,
        "reference_match_required": False,
        "regression_detected": False,
    }

    assert _feature_evaluation_accepts(task, evaluation) is False


def test_completed_local_shape_can_pass_before_other_components_are_built():
    task = _body_task()
    evaluation = {
        "passed": True,
        "visible": True,
        "criteria_satisfied": True,
        "subject_recognizable": False,
        "confidence": 0.95,
        "reference_match_score": 0.95,
        "regression_detected": False,
    }

    assert _feature_evaluation_accepts(task, evaluation) is True



def test_radial_repeated_feature_qa_includes_top_view():
    task = FeatureTask(
        id="radial-openings",
        name="Repeated radial openings",
        count=3,
        strategy="surface_cutout",
        symmetry="radial",
        acceptance_criteria=["Three evenly spaced openings"],
    )

    views = _feature_diagnostic_views(task)

    assert "top" in views
    assert "front-left" in views
    assert "back-right" in views
    assert len(views) >= 3



def test_strict_feature_pass_overrides_relative_equivalence():
    task = FeatureTask(
        id="local-part",
        name="Generic local part",
        strategy="attachment",
        acceptance_criteria=["Requested part is visible and complete"],
    )
    evaluation = {
        "passed": True,
        "visible": True,
        "criteria_satisfied": True,
        "subject_recognizable": False,
        "confidence": 0.96,
        "reference_match_score": 0.90,
        "regression_detected": False,
    }

    complete, keep = _feature_candidate_review_decision(
        task,
        evaluation,
        relative_improved=False,
    )

    assert complete is True
    assert keep is True


def test_relative_improvement_keeps_incomplete_regression_free_draft():
    task = FeatureTask(
        id="local-part",
        name="Generic local part",
        strategy="attachment",
    )
    evaluation = {
        "passed": False,
        "visible": True,
        "criteria_satisfied": False,
        "subject_recognizable": False,
        "confidence": 0.90,
        "reference_match_score": 0.60,
        "regression_detected": False,
    }

    complete, keep = _feature_candidate_review_decision(
        task,
        evaluation,
        relative_improved=True,
    )

    assert complete is False
    assert keep is True


def test_relative_improvement_does_not_keep_regressing_feature_draft():
    task = FeatureTask(
        id="local-part",
        name="Generic local part",
        strategy="attachment",
    )
    evaluation = {
        "passed": False,
        "visible": True,
        "criteria_satisfied": False,
        "subject_recognizable": False,
        "confidence": 0.90,
        "reference_match_score": 0.60,
        "regression_detected": True,
    }

    complete, keep = _feature_candidate_review_decision(
        task,
        evaluation,
        relative_improved=True,
    )

    assert complete is False
    assert keep is False



def test_repeated_component_qa_includes_plan_view_without_declared_symmetry():
    # A feature planner may leave symmetry as 'none' even when the later
    # assembly uses exact radial placement; the QA must still see the outline.
    for symmetry in ("none", "radial"):
        task = FeatureTask(
            id="repeated-panel",
            name="Repeated tapered panel",
            count=3,
            build_mode="component_job",
            strategy="mixed",
            symmetry=symmetry,
            acceptance_criteria=[
                "Three identical panels spaced evenly",
                "Each panel widens toward its free end",
            ],
        )
        assert _feature_diagnostic_views(task) == (
            "top", "front-left", "left", "back-right"
        )


def test_single_feature_without_planar_criteria_keeps_oblique_qa_views():
    task = FeatureTask(
        id="single-insert",
        name="Small insert",
        count=1,
        strategy="attachment",
        symmetry="none",
    )
    assert _feature_diagnostic_views(task) == ("front-left", "left", "back-right")



def test_bilateral_shape_feature_qa_includes_top_view():
    task = FeatureTask(
        id="symmetric-panel",
        name="Symmetric panel",
        strategy="base_mesh_region",
        symmetry="bilateral",
        acceptance_criteria=["Outline visibly varies along its length"],
    )

    views = _feature_diagnostic_views(task)

    assert views[0] == "top"
    assert "front-left" in views
    assert "left" in views
    assert "back-right" in views



def test_negative_preservation_audit_forces_regression_and_rejection():
    evaluation = {
        "passed": True,
        "visible": True,
        "criteria_satisfied": True,
        "subject_recognizable": True,
        "confidence": 0.96,
        "reference_match_score": 0.92,
        "regression_detected": False,
        "summary": "Active feature looks complete.",
        "problems": [],
        "protected_geometry_notes": [],
    }
    audit = {
        "preserved": False,
        "damaged_feature_ids": ["body"],
        "summary": "The previously accepted body was opened and shortened.",
        "notes": ["Top cap disappeared."],
    }

    merged = _apply_preservation_audit(evaluation, audit)

    assert merged["passed"] is False
    assert merged["regression_detected"] is True
    assert "body" in " ".join(merged["protected_geometry_notes"])
    assert "opened and shortened" in " ".join(merged["problems"])


def test_positive_preservation_audit_leaves_feature_verdict_unchanged():
    evaluation = {
        "passed": True,
        "regression_detected": False,
        "problems": [],
        "protected_geometry_notes": [],
    }

    merged = _apply_preservation_audit(
        evaluation,
        {
            "preserved": True,
            "damaged_feature_ids": [],
            "summary": "Protected geometry is unchanged.",
            "notes": [],
        },
    )

    assert merged == evaluation
