"""Generic signed-axis assembly feedback based on verified geometry."""

import re


def required_taper_direction(criteria: list[str]) -> str | None:
    text = " ".join(criteria).lower()
    if re.search(r"wider\s+at\s+(?:the\s+)?(?:end|tip|distal)", text):
        return "widens_outward"
    if re.search(r"narrower\s+at\s+(?:the\s+)?(?:end|tip|distal)", text):
        return "narrows_outward"
    return None


def reverse_failed_axis_contract(
    *, requirement: str | None, proven_direction: str | None,
    previous_sign: str | None, previous_alignment: str | None,
    proposed_sign: str, proposed_alignment: str,
) -> str | None:
    if (
        requirement in {"widens_outward", "narrows_outward"}
        and proven_direction in {"widens_outward", "narrows_outward", "approximately_uniform"}
        and requirement != proven_direction
        and previous_alignment == proposed_alignment == "radial"
        and proposed_sign == previous_sign
    ):
        if proposed_sign == "negative":
            return "positive"
        if proposed_sign == "positive":
            return "negative"
    return None
