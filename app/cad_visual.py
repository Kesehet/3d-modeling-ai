"""Vision-model visual acceptance of real CAD preview renders against verified reference images.

Scoring is explicitly model-reported, not independent human assessment.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from .cad_preview import render_cad_views
from .config import VISION_MODEL


class VisualAssessment(BaseModel):
    recognizable: bool
    confidence: float = Field(ge=0, le=1)
    subject_match_score: float = Field(ge=0, le=1)
    silhouette_match_score: float = Field(ge=0, le=1)
    missing_features: list[str] = Field(default_factory=list, max_length=12)
    defects: list[str] = Field(default_factory=list, max_length=12)
    repair_component_ids: list[str] = Field(default_factory=list, max_length=3)
    explanation: str = Field(min_length=8, max_length=1400)

    @property
    def accepted(self) -> bool:
        return (self.recognizable and self.confidence >= 0.7
                and self.subject_match_score >= 0.75
                and self.silhouette_match_score >= 0.7
                and not self.missing_features)


async def inspect_candidate(*, client, prompt: str, plan, research: dict,
                            model_dir: Path, root: Path, round_number: int) -> VisualAssessment:
    """Compare multiple actual CAD renders to previously pixel-verified references."""
    from .cad_agents import _reference_image

    render_paths = render_cad_views(model_dir / "assembly.glb", root / "previews" / f"round-{round_number}")
    valid_refs = research["verified_visual_references"][:2]
    if not valid_refs:
        raise ValueError("Cannot perform CAD visual QA without verified reference images")
    images = [_reference_image(root / "reference-candidates" / ref["stored_name"])
              for ref in valid_refs]
    images.extend(_reference_image(path) for path in render_paths)
    payload = {
        "requested_subject": prompt,
        "expected_identity": plan.identity,
        "expected_silhouette": plan.silhouette,
        "overall_dimensions_mm": plan.dimensions_mm,
        "components": [{"id": c.id, "purpose": c.purpose} for c in plan.components],
        "reference_image_count": len(valid_refs),
        "render_view_names": [p.stem for p in render_paths],
        "candidate_version": round_number,
        "image_order": ("Reference images first, then CAD render views. "
                        "Do not count references as generated model views."),
    }
    answer = await client.chat_json(
        model=VISION_MODEL,
        system=("You are an independent 3D-model critic. Compare the actual CAD preview "
                "renders to reference image pixels and the requested subject. "
                "Do not certify a model because it exports, has many parts, or resembles "
                "a generic category. Identify significant missing identity features. "
                "Only suggest component IDs from the provided list for bounded repairs. "
                "Use low scores for unrecognizable blobs and avoid optimistic guesses. "
                "Silhouette, proportions, and part placement matter more than colors."),
        prompt=json.dumps(payload, ensure_ascii=False),
        images=images,
        schema=VisualAssessment.model_json_schema(),
        temperature=0,
    )
    result = VisualAssessment.model_validate(answer.data)
    valid_ids = {c.id for c in plan.components}
    if not set(result.repair_component_ids) <= valid_ids:
        raise ValueError("Visual critic requested a non-existent component")
    record = result.model_dump()
    record["accepted"] = result.accepted
    record["preview_files"] = [str(p.relative_to(root)) for p in render_paths]
    record["acceptance_scope"] = "Ollama vision opinion; no human or independent geometry certification"
    (root / f"visual-round-{round_number}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8",
    )
    return result
