from app.feature_tasks import FeaturePlan
from app.main import SubjectInventory, _resolve_initial_modeling_action


def test_base_mesh_inventory_overrides_procedural_first_pass():
    inventory = SubjectInventory.model_validate(
        {
            "subject_family": "vehicle",
            "silhouette_summary": "A smooth continuous liftback body.",
            "complexity": "complex",
            "recommended_strategy": "base_mesh",
            "minimum_distinct_parts": 5,
            "major_parts": [
                {"name": "Body", "importance": "required"},
                {"name": "Wheels", "count": 4, "importance": "required"},
            ],
        }
    )

    assert (
        _resolve_initial_modeling_action("build_procedural", inventory, None)
        == "build_mesh"
    )


def test_primary_base_mesh_feature_overrides_procedural_first_pass():
    plan = FeaturePlan.model_validate(
        {
            "plan_version": 2,
            "subject": "car",
            "features": [
                {
                    "id": "body",
                    "name": "Primary body",
                    "required": True,
                    "priority": 10,
                    "strategy": "base_mesh_region",
                }
            ],
        }
    )

    assert (
        _resolve_initial_modeling_action("build_procedural", None, plan)
        == "build_mesh"
    )


def test_simple_procedural_subject_can_stay_procedural():
    inventory = SubjectInventory.model_validate(
        {
            "subject_family": "simple furniture",
            "silhouette_summary": "Straight constructed members.",
            "complexity": "simple",
            "recommended_strategy": "procedural",
            "minimum_distinct_parts": 2,
            "major_parts": [
                {"name": "Top", "importance": "required"},
                {"name": "Legs", "count": 4, "importance": "required"},
            ],
        }
    )

    assert (
        _resolve_initial_modeling_action("build_procedural", inventory, None)
        == "build_procedural"
    )
