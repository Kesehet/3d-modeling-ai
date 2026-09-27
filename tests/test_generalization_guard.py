import pytest
from pydantic import ValidationError

from app.main import (
    ModelingDirectorDecision,
    _normalize_vision_report_payload,
)


def test_vision_normalizer_preserves_nested_recognizability_signal():
    payload = {
        "analysis": {
            "recognizable": False,
            "subject_match_score": 22,
            "geometry": "The blockout does not read as the requested subject.",
        },
        "recommended_modeling_strategy": "hybrid",
        "summary": "Major silhouette mismatch.",
    }

    normalized = _normalize_vision_report_payload(payload)

    assert normalized["recognizable"] is False
    assert normalized["subject_match_score"] == 0.22
    assert normalized["recommended_modeling_strategy"] == "hybrid"


def test_modeling_director_can_choose_full_strategy_switch():
    decision = ModelingDirectorDecision(
        action="rebuild_mesh",
        subject_match_score=0.18,
        summary="The primitive blockout has the wrong primary silhouette.",
        instructions=[
            "Discard the current body shell.",
            "Reconstruct the main continuous mass from the references.",
        ],
        major_problems=["silhouette", "proportions"],
    )

    assert decision.action == "rebuild_mesh"
    assert decision.subject_match_score == 0.18
    assert "silhouette" in decision.major_problems


def test_modeling_director_acceptance_is_explicit():
    decision = ModelingDirectorDecision(
        action="accept",
        subject_match_score=0.91,
        summary="The rendered model is recognizable and structurally credible.",
    )
    assert decision.action == "accept"


def test_modeling_director_rejects_unknown_actions():
    with pytest.raises(ValidationError):
        ModelingDirectorDecision(
            action="hardcoded_car_fix",
            subject_match_score=0.5,
        )
