from app.feature_tasks import FeatureTask
from app.main import (
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
