"""Object-agnostic feedback for assembly direction; never certifies a model."""

import re


def required_taper_direction(acceptance_criteria: list[str]) -> str | None:
    """Recognize an explicit root/distal width requirement, otherwise do nothing."""
    criteria = " ".join(acceptance_criteria).lower()
    if re.search(
        r"\b(?:wider\s+at|widen(?:s|ing)?\s+toward(?:s)?)\s+"
        r"(?:the\s+)?(?:free\s+)?(?:end|tip|distal|outer)\b", criteria,
    ):
        return "widens_outward"
    if re.search(
        r"\b(?:narrower\s+at|narrow(?:s|ing)?\s+toward(?:s)?)\s+"
        r"(?:the\s+)?(?:free\s+)?(?:end|tip|distal|outer)\b", criteria,
    ):
        return "narrows_outward"
    return None


def reverse_failed_axis_contract(
    *, requirement: str | None, proven_direction: str | None,
    previous_sign: str | None, previous_alignment: str | None,
    proposed_sign: str, proposed_alignment: str,
) -> str | None:
    """Propose the opposite *sign* only after proven contradictory geometry.

    Corrects repeated planner decisions without touching frozen meshes,
    object-specific proportions or mandatory strict QA.
    """
    if (
        requirement in ("widens_outward", "narrows_outward")
        and proven_direction in ("widens_outward", "narrows_outward", "approximately_uniform")
        and proven_direction != requirement
        and previous_alignment == proposed_alignment == "radial"
        and previous_sign == proposed_sign
        and proposed_sign in ("positive", "negative")
    ):
        return "negative" if proposed_sign == "positive" else "positive"
    return None
