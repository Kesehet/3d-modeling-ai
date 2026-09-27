from app.main import (
    GenericSceneSpec,
    SceneObjectSpec,
    SubjectInventory,
    SubjectPartSpec,
    _aggregate_generic_quality_verdicts,
    _normalize_generic_quality_payload,
    _normalize_vision_report_payload,
    _scene_inventory_coverage,
)


def obj(name: str) -> SceneObjectSpec:
    return SceneObjectSpec(
        name=name,
        shape="cube",
        location=[0.0, 0.0, 0.0],
        scale=[1.0, 1.0, 1.0],
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


def test_subject_inventory_detects_missing_repeated_parts_without_subject_hacks():
    inventory = SubjectInventory(
        subject_family="passenger vehicle",
        silhouette_summary="Low body with a raised cabin and four wheels.",
        complexity="complex",
        recommended_strategy="hybrid",
        minimum_distinct_parts=7,
        major_parts=[
            SubjectPartSpec(name="main body"),
            SubjectPartSpec(name="cabin"),
            SubjectPartSpec(name="windshield"),
            SubjectPartSpec(name="wheels", count=4),
        ],
    )
    bad = GenericSceneSpec(
        title="bad holdout",
        objects=[
            obj("main body"),
            obj("front left wheel"),
            obj("front right wheel"),
        ],
    )

    coverage = _scene_inventory_coverage(bad, inventory)

    assert coverage["passes"] is False
    assert coverage["required_coverage"] < 0.8
    assert any("cabin" in item for item in coverage["missing_required"])
    assert any("windshield" in item for item in coverage["missing_required"])
    assert any("wheels" in item for item in coverage["missing_required"])


def test_subject_inventory_accepts_semantically_complete_holdout_parts():
    inventory = SubjectInventory(
        subject_family="passenger vehicle",
        silhouette_summary="Low body with a raised cabin and four wheels.",
        complexity="complex",
        recommended_strategy="hybrid",
        minimum_distinct_parts=7,
        major_parts=[
            SubjectPartSpec(name="main body"),
            SubjectPartSpec(name="cabin"),
            SubjectPartSpec(name="windshield"),
            SubjectPartSpec(name="wheels", count=4),
        ],
    )
    good = GenericSceneSpec(
        title="generalized holdout",
        objects=[
            obj("main body"),
            obj("cabin roof"),
            obj("windshield"),
            obj("front left wheel"),
            obj("front right wheel"),
            obj("rear left wheel"),
            obj("rear right wheel"),
        ],
    )

    coverage = _scene_inventory_coverage(good, inventory)

    assert coverage["passes"] is True
    assert coverage["required_coverage"] == 1.0


def test_quality_normalizer_requires_explicit_recognizability_and_score():
    payload = {
        "recognizable": False,
        "subject_match_score": 18,
        "recommended_strategy": "hybrid",
        "summary": "The candidate is a flat wedge, not a recognizable vehicle.",
        "major_missing_parts": ["cabin", "windshield", "rear body volume"],
    }

    normalized = _normalize_generic_quality_payload(payload)

    assert normalized["recognizable"] is False
    assert normalized["subject_match_score"] == 0.18
    assert normalized["recommended_strategy"] == "hybrid"
    assert "cabin" in normalized["major_missing_parts"]


def test_quality_ensemble_needs_two_positive_votes_when_multiple_models_answer():
    verdicts = [
        {
            "recognizable": True,
            "subject_match_score": 0.86,
            "recommended_strategy": "procedural",
            "summary": "Looks correct.",
            "major_missing_parts": [],
        },
        {
            "recognizable": False,
            "subject_match_score": 0.42,
            "recommended_strategy": "hybrid",
            "summary": "Silhouette is still wrong.",
            "major_missing_parts": ["cabin"],
        },
        {
            "recognizable": False,
            "subject_match_score": 0.31,
            "recommended_strategy": "hybrid",
            "summary": "Not recognizable.",
            "major_missing_parts": ["windshield"],
        },
    ]

    aggregate = _aggregate_generic_quality_verdicts(verdicts)

    assert aggregate["recognizable"] is False
    assert aggregate["positive_votes"] == 1
    assert aggregate["negative_votes"] == 2
    assert aggregate["recommended_strategy"] == "hybrid"


def test_quality_ensemble_accepts_strong_multimodel_consensus():
    verdicts = [
        {
            "recognizable": True,
            "subject_match_score": 0.88,
            "recommended_strategy": "procedural",
            "summary": "Recognizable.",
            "major_missing_parts": [],
        },
        {
            "recognizable": True,
            "subject_match_score": 0.83,
            "recommended_strategy": "procedural",
            "summary": "Strong match.",
            "major_missing_parts": [],
        },
    ]

    aggregate = _aggregate_generic_quality_verdicts(verdicts)

    assert aggregate["recognizable"] is True
    assert aggregate["positive_votes"] == 2
    assert aggregate["subject_match_score"] >= 0.83
