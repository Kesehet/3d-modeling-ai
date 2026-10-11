"""Generic required-component contract derived from inspected reference pixels.

This does not choose or generate geometry. It prevents a candidate from gaming
connectivity/silhouette gates by omitting independently requested features.
Names, primitive types and count language are evidence, not a visual pass.
"""
from __future__ import annotations

import math
import re

_WORDS = re.compile(r"[a-z0-9]+")
_SKIP = {
    "a", "an", "the", "one", "two", "three", "four", "five", "six",
    "main", "central", "primary", "part", "parts", "component", "components",
    "piece", "pieces", "structure", "assembly", "assemblies", "system",
    "surface", "plane", "outer", "inner", "visible", "pair", "pairs",
    "panel", "panels", "protrusion", "lid", "fascia", "fasciae",
    "housing", "housings", "view", "and", "or", "volume", "object", "group",
}
_COUNTS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _words(value: str) -> set[str]:
    result: set[str] = set()
    for word in _WORDS.findall(value.lower().replace("_", " ")):
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("s") and len(word) > 3 and not word.endswith(("ss", "us")):
            word = word[:-1]
        if word not in _SKIP:
            result.add(word)
    return result


def evaluate_required_component_coverage(blueprint, requirements: list[str]) -> dict:
    """Conservative name/kind coverage, independent of subject category."""
    if not requirements:
        return {"pass": True, "status": "no_requirements", "checked": 0, "missing": []}
    descriptions = [
        _words(part.name) | _words(part.kind)
        for part in blueprint.parts
    ]
    missing = []
    satisfied = []
    for requirement in requirements[:24]:
        terms = _words(requirement)
        if not terms:
            continue
        low = requirement.lower()
        paired = re.search(
            r"\b(front|rear|left|right|upper|lower)\s+and\s+"
            r"(front|rear|left|right|upper|lower)\b", low
        )
        if paired:
            first, second = paired.groups()
            core = terms - {first, second}
            matches = [
                part for part, tokens in zip(blueprint.parts, descriptions, strict=True)
                if core.issubset(tokens) and (first in tokens or second in tokens)
            ]
            paired_satisfied = (
                any(core.issubset(tokens) and first in tokens for tokens in descriptions)
                and any(core.issubset(tokens) and second in tokens for tokens in descriptions)
            )
        else:
            matches = [part for part, tokens in zip(blueprint.parts, descriptions, strict=True)
                       if terms.issubset(tokens)]
            paired_satisfied = True
        count_match = re.search(r"\b([2-6])\b", low)
        named = next((n for word, n in _COUNTS.items()
                      if re.search(rf"\b{word}\b", low)), None)
        minimum_count = max(2, named or 1) if paired else (
            int(count_match.group(1)) if count_match else named or 1
        )
        if paired_satisfied and len(matches) >= minimum_count:
            satisfied.append(requirement)
        else:
            missing.append({
                "required": requirement,
                "matched_parts": [part.name for part in matches],
                "minimum_count": minimum_count,
            })
    total = len(satisfied) + len(missing)
    # A multi-feature visual brief can include minor details, but cannot lose
    # most of its defining components. Two-item briefs require both.
    minimum = total if total <= 2 else math.ceil(total * 0.60)
    passed = len(satisfied) >= minimum
    return {
        "pass": passed,
        "status": "passed" if passed else "missing_required_components",
        "checked": total,
        "matched": len(satisfied),
        "minimum_matched": minimum,
        "missing": missing,
        "limitations": "Part names and kinds are not proof of actual geometry",
    }
