from app.main import _enforce_subject_geometry


def _part(name: str) -> dict:
    return {
        "name": name,
        "shape": "sphere",
        "location": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "rotation_deg": [0.0, 0.0, 0.0],
        "start": None,
        "end": None,
        "radius": None,
        "color": "#808080",
        "bevel": True,
        "smooth": True,
    }


def test_lamp_assembly_connects_neck_to_shade():
    data = {
        "objects": [
            _part("base"),
            _part("vertical stem"),
            _part("pivot joint"),
            _part("angled neck"),
            _part("dome shade"),
        ]
    }
    result = _enforce_subject_geometry(data, "Create a compact retro desk lamp")
    by_name = {item["name"]: item for item in result["objects"]}
    neck = by_name["angled neck"]
    shade = by_name["dome shade"]
    assert neck["shape"] == "rod"
    assert shade["shape"] == "frustum"
    assert neck["end"][0] == shade["location"][0]
    assert neck["end"][2] > shade["location"][2]


def test_chair_assembly_adds_five_real_caster_wheels():
    data = {
        "objects": [
            _part("seat"),
            _part("backrest"),
            _part("left armrest"),
            _part("right armrest"),
            _part("gas lift column"),
        ]
    }
    result = _enforce_subject_geometry(data, "Create a modern ergonomic office chair")
    wheels = [item for item in result["objects"] if "caster wheel" in item["name"]]
    spokes = [item for item in result["objects"] if "base spoke" in item["name"]]
    assert len(wheels) == 5
    assert len(spokes) == 5
    assert all(item["shape"] == "torus" for item in wheels)
    assert all(item["shape"] == "rod" for item in spokes)


def test_quadruped_assembly_builds_articulated_leg_chain():
    names = [
        "torso",
        "leg_front_left_upper",
        "leg_front_left_knee_joint",
        "leg_front_left_lower",
        "leg_front_left_foot",
        "head",
        "camera_lens",
        "antenna_mast",
        "battery_pack",
        "tool_arm_upper",
        "tool_arm_joint",
        "tool_arm_lower",
        "tool_end_effector",
    ]
    data = {"objects": [_part(name) for name in names]}
    result = _enforce_subject_geometry(
        data,
        "Create a quadruped robot with articulated legs and a tool arm",
    )
    by_name = {item["name"]: item for item in result["objects"]}
    assert by_name["leg_front_left_upper"]["shape"] == "rod"
    assert by_name["leg_front_left_lower"]["shape"] == "rod"
    assert by_name["leg_front_left_knee_joint"]["shape"] == "sphere"
    assert by_name["leg_front_left_foot"]["shape"] == "cube"
    assert by_name["camera_lens"]["location"][1] < 0


def test_sneaker_assembly_uses_wedge_and_lace_rods():
    data = {
        "objects": [
            _part("outsole"),
            _part("upper"),
            _part("toe box"),
            _part("heel counter"),
            _part("tongue"),
            _part("opening collar"),
            _part("lace 1"),
            _part("lace 2"),
            _part("lace 3"),
        ]
    }
    result = _enforce_subject_geometry(data, "Create a low-top sneaker")
    by_name = {item["name"]: item for item in result["objects"]}
    assert by_name["upper"]["shape"] == "wedge"
    assert by_name["toe box"]["shape"] == "sphere"
    assert by_name["opening collar"]["shape"] == "torus"
    assert by_name["lace 1"]["shape"] == "rod"
    assert by_name["lace 1"]["start"][1] < by_name["lace 1"]["end"][1]
