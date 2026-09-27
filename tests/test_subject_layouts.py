from app.layout import enforce_subject_layout


def _item(name: str) -> dict:
    return {
        "name": name,
        "shape": "sphere",
        "location": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "rotation_deg": [0.0, 0.0, 0.0],
        "color": "#808080",
        "bevel": True,
        "smooth": True,
    }


def _by_name(spec: dict) -> dict[str, dict]:
    return {item["name"]: item for item in spec["objects"]}


def test_pikachu_tail_becomes_visible_zigzag():
    names = [
        "body", "head", "left eye", "right eye", "left cheek", "right cheek",
        "left ear", "right ear", "left ear tip", "right ear tip",
        "left arm", "right arm", "left foot", "right foot",
        "tail seg1", "tail seg2", "tail seg3", "tail seg4", "tail tip",
    ]
    spec = {"objects": [_item(name) for name in names]}
    fixed = enforce_subject_layout(spec, "Create a stylized Pikachu test model")
    objects = _by_name(fixed)
    tail = [objects[f"tail seg{i}"] for i in range(1, 5)] + [objects["tail tip"]]
    assert all(item["shape"] == "rod" for item in tail)
    assert len({tuple(item["end"]) for item in tail}) == 5
    assert objects["left eye"]["location"][1] < 0
    assert objects["left ear"]["shape"] == "cone"


def test_lamp_has_unbroken_endpoint_chain():
    names = ["base", "stem", "pivot joint", "neck", "shade"]
    fixed = enforce_subject_layout(
        {"objects": [_item(name) for name in names]},
        "Create a compact retro desk lamp",
    )
    objects = _by_name(fixed)
    assert objects["stem"]["end"] == objects["neck"]["start"]
    assert objects["pivot joint"]["location"] == objects["stem"]["end"]
    neck_end = objects["neck"]["end"]
    shade = objects["shade"]
    assert abs(shade["location"][0] - neck_end[0]) < shade["scale"][0]
    assert abs(shade["location"][2] - neck_end[2]) < shade["scale"][2]


def test_sneaker_is_long_low_and_laces_cross_upper():
    names = [
        "sole base", "sole mid layer", "sole top layer", "rounded toe box",
        "upper body", "heel counter", "tongue", "foot opening",
        "lace 1", "lace 2", "lace 3", "lace 4",
    ]
    fixed = enforce_subject_layout(
        {"objects": [_item(name) for name in names]},
        "Create a stylized low-top sneaker",
    )
    objects = _by_name(fixed)
    assert objects["sole base"]["scale"][0] > objects["sole base"]["scale"][1] * 2
    assert objects["rounded toe box"]["location"][0] > 1
    assert objects["heel counter"]["location"][0] < -1
    assert objects["lace 1"]["shape"] == "rod"
    assert objects["lace 1"]["start"][1] < 0 < objects["lace 1"]["end"][1]


def test_chair_uses_radial_spokes_and_torus_casters():
    names = [
        "seat", "backrest", "left armrest", "right armrest",
        "left arm support", "right arm support", "gas lift column", "base hub",
        "spoke 0", "spoke 1", "spoke 2", "spoke 3", "spoke 4",
        "wheel 0", "wheel 1", "wheel 2", "wheel 3", "wheel 4",
    ]
    fixed = enforce_subject_layout(
        {"objects": [_item(name) for name in names]},
        "Create a modern office chair with five caster wheels",
    )
    objects = _by_name(fixed)
    assert all(objects[f"spoke {i}"]["shape"] == "rod" for i in range(5))
    assert all(objects[f"wheel {i}"]["shape"] == "torus" for i in range(5))
    assert len({tuple(objects[f"wheel {i}"]["location"][:2]) for i in range(5)}) == 5


def test_quadruped_has_articulated_leg_chains_and_external_details():
    names = [
        "torso", "front camera head", "camera lens", "left sensor pod",
        "right sensor pod", "antenna mast", "antenna tip", "rear battery pack",
        "leg fl upper", "leg fl joint", "leg fl lower", "leg fl foot",
        "leg fr upper", "leg fr joint", "leg fr lower", "leg fr foot",
        "leg rl upper", "leg rl joint", "leg rl lower", "leg rl foot",
        "leg rr upper", "leg rr joint", "leg rr lower", "leg rr foot",
        "tool arm upper", "tool arm joint", "tool arm lower", "tool end",
    ]
    fixed = enforce_subject_layout(
        {"objects": [_item(name) for name in names]},
        "Create a complex quadruped robot with four articulated legs",
    )
    objects = _by_name(fixed)
    for code in ("fl", "fr", "rl", "rr"):
        upper = objects[f"leg {code} upper"]
        joint = objects[f"leg {code} joint"]
        lower = objects[f"leg {code} lower"]
        assert upper["shape"] == "rod"
        assert lower["shape"] == "rod"
        assert upper["end"] == joint["location"] == lower["start"]
    assert objects["front camera head"]["location"][1] < -1
    assert objects["rear battery pack"]["location"][1] > 1
    assert objects["tool arm upper"]["shape"] == "rod"
