"""Fixed quality-regression benchmarks for the generic 3D pipeline.

These profiles are intentionally explicit. They are deploy gates, not general prompt
classification rules: a benchmark only runs when its key is requested directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkProfile:
    key: str
    title: str
    prompt: str
    min_objects: int
    required_part_groups: dict[str, tuple[str, ...]]
    visual_requirements: tuple[str, ...]


BENCHMARKS: dict[str, BenchmarkProfile] = {
    "pikachu": BenchmarkProfile(
        key="pikachu",
        title="Stylized Pikachu",
        prompt=(
            "Create a stylized Pikachu test model with a large head, yellow rounded body, "
            "two long pointed ears with black tips, two black eyes, two red cheek circles, "
            "two short arms, two feet, and a large lightning-bolt tail."
        ),
        min_objects=12,
        required_part_groups={
            "head": ("head", "face"),
            "body": ("body", "torso"),
            "ears": ("ear",),
            "eyes": ("eye",),
            "cheeks": ("cheek", "face patch", "red patch"),
            "arms": ("arm", "forelimb", "hand"),
            "feet": ("foot", "feet", "hind leg"),
            "tail": ("tail",),
        },
        visual_requirements=(
            "large head and rounded yellow body",
            "two long pointed ears",
            "black eyes",
            "red cheek circles",
            "short arms and visible feet",
            "lightning-bolt tail",
        ),
    ),
    "desk-lamp": BenchmarkProfile(
        key="desk-lamp",
        title="Articulated retro desk lamp",
        prompt=(
            "Create a compact retro desk lamp with a round weighted base, a connected vertical stem, "
            "a clearly angled neck with a visible pivot joint, and a dome-shaped shade physically "
            "attached to the neck. The lamp must read as one connected articulated object."
        ),
        min_objects=5,
        required_part_groups={
            "base": ("base",),
            "stem": ("stem", "post", "upright", "pole", "shaft", "support"),
            "neck": ("neck", "arm", "boom"),
            "joint": ("joint", "pivot", "hinge"),
            "shade": ("shade", "dome", "lamphead", "lamp head", "hood"),
        },
        visual_requirements=(
            "round weighted base",
            "connected vertical stem",
            "clearly angled neck",
            "visible pivot or articulation joint",
            "dome shade attached to the neck",
            "all major lamp parts visibly connected rather than floating",
        ),
    ),
    "sneaker": BenchmarkProfile(
        key="sneaker",
        title="Low-top sneaker",
        prompt=(
            "Create a stylized low-top sneaker with a layered sole, rounded toe box, shaped upper, "
            "heel counter, tongue, foot opening, and several visible laces crossing the tongue."
        ),
        min_objects=8,
        required_part_groups={
            "sole": ("sole", "midsole", "outsole", "platform"),
            "toe": ("toe",),
            "upper": ("upper", "shoe body", "body"),
            "heel": ("heel",),
            "tongue": ("tongue",),
            "opening": ("opening", "collar"),
            "laces": ("lace", "laces", "cord", "string"),
        },
        visual_requirements=(
            "layered sole",
            "rounded toe box",
            "recognizable shoe upper and heel",
            "tongue and foot opening",
            "multiple visible laces crossing the tongue",
        ),
    ),
    "office-chair": BenchmarkProfile(
        key="office-chair",
        title="Office chair",
        prompt=(
            "Create a modern ergonomic office chair with a padded seat, upright backrest, two armrests, "
            "central gas-lift column, five-spoke wheeled base, and five caster wheels."
        ),
        min_objects=11,
        required_part_groups={
            "seat": ("seat", "cushion"),
            "backrest": ("backrest", "back"),
            "armrests": ("armrest", "arm rest", "arm support"),
            "column": ("column", "lift", "post"),
            "base": ("base", "spoke", "star"),
            "wheels": ("wheel", "caster"),
        },
        visual_requirements=(
            "distinct seat and upright backrest",
            "two armrests",
            "central support column",
            "five-spoke base",
            "caster wheels at the ends of the base",
        ),
    ),
    "quadruped-robot": BenchmarkProfile(
        key="quadruped-robot",
        title="Complex quadruped robot",
        prompt=(
            "Create a complex retro-futuristic quadruped exploration robot with an armored rectangular "
            "torso, four articulated two-segment legs with round joints and feet, an offset camera head, "
            "twin side sensor pods, top antenna mast, rear battery pack, and one asymmetric tool arm. "
            "Use a white, dark gray and safety-orange color scheme."
        ),
        min_objects=18,
        required_part_groups={
            "torso": ("torso", "body", "chassis"),
            "legs": ("leg", "thigh", "shin"),
            "joints": ("joint", "knee", "hip"),
            "feet": ("foot", "feet"),
            "camera head": ("camera", "head", "optic"),
            "sensors": ("sensor", "pod"),
            "antenna": ("antenna", "mast"),
            "battery": ("battery", "pack"),
            "tool arm": ("tool", "arm", "manipulator"),
        },
        visual_requirements=(
            "armored torso",
            "four articulated multi-segment legs with joints and feet",
            "offset camera head",
            "side sensor pods",
            "top antenna mast",
            "rear battery pack",
            "one asymmetric tool arm",
        ),
    ),
}


def get_benchmark(key: str) -> BenchmarkProfile:
    try:
        return BENCHMARKS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark: {key}") from exc


def _normalize_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def evaluate_scene_spec_structural(spec: dict, profile: BenchmarkProfile) -> dict:
    objects = spec.get("objects") if isinstance(spec, dict) else None
    if not isinstance(objects, list):
        objects = []

    object_names = [
        _normalize_name(str(item.get("name") or ""))
        for item in objects
        if isinstance(item, dict)
    ]
    searchable = " | ".join(object_names)

    matched: dict[str, bool] = {}
    evidence: dict[str, list[str]] = {}
    for group, aliases in profile.required_part_groups.items():
        group_hits = [
            name
            for name in object_names
            if any(_normalize_name(alias) in name for alias in aliases)
        ]
        matched[group] = bool(group_hits)
        evidence[group] = group_hits[:8]

    missing = [group for group, present in matched.items() if not present]
    object_count = len(objects)
    passed = object_count >= profile.min_objects and not missing

    return {
        "passed": passed,
        "benchmark": profile.key,
        "object_count": object_count,
        "minimum_object_count": profile.min_objects,
        "required_parts": matched,
        "missing_parts": missing,
        "evidence": evidence,
        "object_names": object_names,
        "searchable_names": searchable,
    }
