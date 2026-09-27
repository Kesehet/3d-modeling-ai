from app.main import _generic_spatial_guidance


def test_spatial_guidance_is_universal_not_subject_specific():
    character = _generic_spatial_guidance("Create a character")
    vehicle = _generic_spatial_guidance("Create a vehicle")
    chair = _generic_spatial_guidance("Create a chair")

    assert character == vehicle == chair
    assert "negative Y" in character
    assert "reference images" in character
    assert "rod/beam" in character


def test_spatial_guidance_does_not_encode_object_family_templates():
    guidance = _generic_spatial_guidance("anything")

    for subject_specific_term in (
        "pikachu",
        "sneaker",
        "lamp rule",
        "chair rule",
        "quadruped rule",
        "five radial spokes",
        "lace rods",
    ):
        assert subject_specific_term not in guidance.lower()
