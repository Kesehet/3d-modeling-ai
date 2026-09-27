from app.main import (
    GenericSceneSpec,
    SceneObjectSpec,
    SubjectInventory,
    SubjectPartSpec,
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
