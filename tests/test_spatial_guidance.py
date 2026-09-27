from app.main import _generic_spatial_guidance


def test_character_guidance_exposes_face_on_negative_y():
    guidance = _generic_spatial_guidance("Create a stylized character figurine")
    assert "negative Y" in guidance
    assert "eyes/cheeks" in guidance
    assert "tail" in guidance


def test_lamp_guidance_requires_connected_chain():
    guidance = _generic_spatial_guidance("Create an articulated desk lamp")
    assert "base -> vertical stem -> pivot/joint -> angled neck -> shade" in guidance
    assert "never floating" in guidance


def test_sneaker_guidance_avoids_vertical_blob_stack():
    guidance = _generic_spatial_guidance("Create a low-top sneaker")
    assert "sole" in guidance
    assert "lace rods" in guidance
    assert "rather than stacking round primitives vertically" in guidance


def test_chair_guidance_places_backrest_behind_seat():
    guidance = _generic_spatial_guidance("Create a modern office chair")
    assert "backrest sits behind the seat" in guidance
    assert "five radial spokes" in guidance
