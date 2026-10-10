from app.component_direction import (
    required_taper_direction,
    reverse_failed_axis_contract,
)


def test_explicit_generic_taper_requirement_not_object_name():
    assert required_taper_direction(
        ["Blade has a tapered shape (wider at the end than the root)"]
    ) == "widens_outward"
    assert required_taper_direction(["Panel is narrower at the distal end"]) == "narrows_outward"
    assert required_taper_direction(["Three evenly spaced panels"]) is None


def test_proven_bad_radial_direction_is_flipped_before_reassembly():
    evidence = {\n        "requirement": "widens_outward",
        "proven_direction": "narrows_outward",
        "previous_sign": "negative",
        "previous_alignment": "radial",
        "proposed_sign": "negative",
        "proposed_alignment": "radial",\n    }
    assert reverse_failed_axis_contract(**evidence) == "positive"

    # Already-corrected AI plans survive, not flipped again.
    assert reverse_failed_axis_contract(
        **{**evidence, "proposed_sign": "positive"}
    ) is None

    # Symmetric opposite case must work for arbitrary part geometry.
    assert reverse_failed_axis_contract(
        **{**evidence, "requirement": "narrows_outward",
           "proven_direction": "widens_outward", "previous_sign": "positive",
           "proposed_sign": "positive"}
    ) == "negative"


def test_missing_or_ambiguous_evidence_never_changes_orientation():
    previous = {\n        "requirement": "widens_outward", "proven_direction": "narrows_outward",
        "previous_sign": "negative", "previous_alignment": "radial",
        "proposed_sign": "negative", "proposed_alignment": "radial",\n    }
    for changes in (
        {"requirement": None},
        {"proven_direction": None},
        {"proven_direction": "widens_outward"},
        {"previous_sign": None},
        {"previous_alignment": "free"},
        {"proposed_alignment": "tangential"},
        {"proposed_sign": "positive"},
    ):
        assert reverse_failed_axis_contract(**{**previous, **changes}) is None
