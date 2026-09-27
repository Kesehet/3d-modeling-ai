from app.main import _enforce_character_visibility, _generic_spatial_guidance


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



def test_character_visibility_repair_moves_face_to_front_and_fixes_ears_tail():
    spec = {
        "objects": [
            {"name": "body", "shape": "sphere", "location": [0, 0, 1], "scale": [1, 0.8, 1.2]},
            {"name": "head", "shape": "sphere", "location": [0, 0, 2.7], "scale": [1.2, 1.0, 1.0]},
            {"name": "left eye", "shape": "sphere", "location": [-0.3, 0.6, 2.8], "scale": [0.2, 0.2, 0.2]},
            {"name": "right eye", "shape": "sphere", "location": [0.3, 0.6, 2.8], "scale": [0.2, 0.2, 0.2]},
            {"name": "left cheek", "shape": "sphere", "location": [-0.6, 0.5, 2.5], "scale": [0.25, 0.2, 0.2]},
            {"name": "right cheek", "shape": "sphere", "location": [0.6, 0.5, 2.5], "scale": [0.25, 0.2, 0.2]},
            {"name": "left ear", "shape": "rod", "location": [-0.4, 0, 3.3], "scale": [1, 1, 1]},
            {"name": "right ear", "shape": "rod", "location": [0.4, 0, 3.3], "scale": [1, 1, 1]},
            {"name": "left ear tip", "shape": "sphere", "location": [-0.4, 0, 3.6], "scale": [0.2, 0.2, 0.2]},
            {"name": "right ear tip", "shape": "sphere", "location": [0.4, 0, 3.6], "scale": [0.2, 0.2, 0.2]},
            {"name": "left arm", "shape": "rod", "location": [-0.4, 0, 1.2], "scale": [1, 1, 1]},
            {"name": "right arm", "shape": "rod", "location": [0.4, 0, 1.2], "scale": [1, 1, 1]},
            {"name": "left foot", "shape": "rod", "location": [-0.3, 0, 0.2], "scale": [1, 1, 1]},
            {"name": "right foot", "shape": "rod", "location": [0.3, 0, 0.2], "scale": [1, 1, 1]},
            {"name": "tail segment 1", "shape": "sphere", "location": [0, 0, 2], "scale": [1, 1, 1]},
            {"name": "tail segment 2", "shape": "sphere", "location": [0, 0, 2], "scale": [1, 1, 1]},
            {"name": "tail segment 3", "shape": "sphere", "location": [0, 0, 2], "scale": [1, 1, 1]},
        ]
    }

    repaired = _enforce_character_visibility(spec, "Create a stylized Pikachu character figurine")
    by_name = {item["name"]: item for item in repaired["objects"]}

    assert by_name["left eye"]["location"][1] < -0.8
    assert by_name["right cheek"]["location"][1] < -0.8
    assert by_name["left ear"]["shape"] == "cone"
    assert by_name["right ear"]["shape"] == "cone"
    assert by_name["left ear tip"]["color"] == "#111111"
    assert by_name["left arm"]["shape"] == "sphere"
    assert by_name["left foot"]["shape"] == "sphere"
    assert by_name["tail segment 1"]["shape"] == "rod"
    assert by_name["tail segment 1"]["start"] != by_name["tail segment 1"]["end"]
    assert by_name["tail segment 3"]["end"][0] > by_name["tail segment 3"]["start"][0]
