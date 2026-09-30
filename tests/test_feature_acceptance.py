from app.feature_tasks import FeatureTask
from app.main import _feature_diagnostic_views, _feature_evaluation_accepts


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
