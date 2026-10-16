"""Research-first, bounded multi-agent CAD prototype.

Uses trusted CAD operations, never executes code suggested by an LLM.
This workflow is opt-in, keeps failures as evidence, and is not a visual acceptance gate.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
from io import BytesIO
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, Field, model_validator

from .cad_engine import CAD_SYSTEM_PROMPT, CadBlueprint, CadPart, build_cad
from .config import REASONING_MODEL, VISION_MODEL
from .ollama import OllamaProxyClient
from .research import research_web_references


class VisualFinding(BaseModel):
    matches_subject: bool
    confidence: float = Field(ge=0, le=1)
    viewpoint: str = Field(max_length=80)
    visible_shape: str = Field(min_length=10, max_length=1200)
    identifiable_features: list[str] = Field(min_length=1, max_length=12)


class ComponentJob(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,30}$")
    purpose: str = Field(min_length=10, max_length=400)
    center_mm: tuple[float, float, float]
    size_mm: tuple[float, float, float]
    # Dependent components are generated only after their prerequisites.
    depends_on: list[str] = Field(default_factory=list, max_length=5)


class AssemblyRelation(BaseModel):
    a: str
    b: str
    relationship: str = Field(pattern=r"^(attached|separate)$")
    tolerance_mm: float = Field(default=10, ge=0, le=200)


class DesignPlan(BaseModel):
    title: str = Field(min_length=3, max_length=140)
    identity: str = Field(min_length=8, max_length=300)
    silhouette: str = Field(min_length=20, max_length=1200)
    dimensions_mm: tuple[float, float, float]
    components: list[ComponentJob] = Field(min_length=1, max_length=6)
    relationships: list[AssemblyRelation] = Field(default_factory=list, max_length=18)

    @model_validator(mode="after")
    def validate_graph(self):
        ids = [c.id for c in self.components]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate component identifiers")
        lookup = set(ids)
        for component in self.components:
            if not all(math.isfinite(v) and v > 0 for v in component.size_mm):
                raise ValueError("Component dimensions must be positive")
            if not all(math.isfinite(v) for v in component.center_mm):
                raise ValueError("Non-finite component center")
            if component.id in component.depends_on or not set(component.depends_on) <= lookup:
                raise ValueError("Unknown or self-dependent component")
        for relation in self.relationships:
            if relation.a == relation.b or relation.a not in lookup or relation.b not in lookup:
                raise ValueError("Assembly relation refers to missing or same component")
        levels = topological_levels(self.components)
        if sum(map(len, levels)) != len(self.components):
            raise ValueError("Cyclic component dependencies")
        if not all(math.isfinite(d) and 0 < d <= 20000 for d in self.dimensions_mm):
            raise ValueError("Invalid design envelope")
        return self


def topological_levels(components: list[ComponentJob]) -> list[list[str]]:
    pending = {c.id: set(c.depends_on) for c in components}
    levels = []
    completed: set[str] = set()
    while pending:
        ready = [name for name, deps in pending.items() if deps <= completed]
        if not ready:
            raise ValueError("Cyclic or missing component dependency")
        levels.append(ready)
        completed.update(ready)
        for name in ready:
            del pending[name]
    return levels


def part_aabb(part: CadPart) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Conservative world AABB including local loft offsets and Euler rotation."""
    if part.kind == "loft":
        points = []
        for sec in part.loft_sections:
            if part.loft_axis == "x":
                points.extend([
                    (sec.at_mm, sec.center_u_mm + u * sec.width_mm / 2,
                     sec.center_v_mm + v * sec.depth_mm / 2)
                    for u in (-1, 1) for v in (-1, 1)
                ])
            elif part.loft_axis == "y":
                points.extend([
                    (sec.center_u_mm + u * sec.width_mm / 2, sec.at_mm,
                     sec.center_v_mm + v * sec.depth_mm / 2)
                    for u in (-1, 1) for v in (-1, 1)
                ])
            else:
                points.extend([
                    (sec.center_u_mm + u * sec.width_mm / 2,
                     sec.center_v_mm + v * sec.depth_mm / 2, sec.at_mm)
                    for u in (-1, 1) for v in (-1, 1)
                ])
        lower = [min(p[i] for p in points) for i in range(3)]
        upper = [max(p[i] for p in points) for i in range(3)]
        center = [(lower[i] + upper[i]) / 2 for i in range(3)]
        half = [(upper[i] - lower[i]) / 2 for i in range(3)]
    else:
        center = [0.0, 0.0, 0.0]
        if part.kind == "box":
            half = [v / 2 for v in part.dimensions_mm]
        elif part.kind == "sphere":
            half = [part.radius_mm] * 3
        else:
            half = [part.radius_mm, part.radius_mm, part.height_mm / 2]
    ax, ay, az = (math.radians(a) for a in part.rotation_deg)
    cx, cy, cz = math.cos(ax), math.cos(ay), math.cos(az)
    sx, sy, sz = math.sin(ax), math.sin(ay), math.sin(az)
    rotation = (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )
    extent = [sum(abs(rotation[i][j]) * half[j] for j in range(3)) for i in range(3)]
    world_center = [
        part.position_mm[i] + sum(rotation[i][j] * center[j] for j in range(3))
        for i in range(3)
    ]
    return (
        tuple(world_center[i] - extent[i] for i in range(3)),
        tuple(world_center[i] + extent[i] for i in range(3)),
    )

def aabb_gap(a, b) -> float:
    return math.sqrt(sum(max(a[0][i] - b[1][i], b[0][i] - a[1][i], 0) ** 2 for i in range(3)))


def assembly_report(plan: DesignPlan, groups: dict[str, list[CadPart]]) -> dict:
    """Conservative first-pass spatial screening, NOT exact surface/visual QA."""
    errors = []
    measurements = []
    for relation in plan.relationships:
        if relation.relationship != "attached":
            continue
        gap = min(
            (aabb_gap(part_aabb(a), part_aabb(b))
             for a in groups[relation.a] for b in groups[relation.b]),
            default=float("inf"),
        )
        measurements.append({"a": relation.a, "b": relation.b, "gap_mm": round(gap, 2)})
        if gap > relation.tolerance_mm:
            errors.append(
                f"{relation.a} and {relation.b} have {gap:.1f} mm AABB gap "
                f"(allowed {relation.tolerance_mm:.1f} mm)"
            )
    return {"pass": not errors, "failures": errors, "measurements": measurements,
            "scope": "conservative assembly AABB screening only; visual QA still required"}


def _reference_image(path: Path) -> str:
    """Keep multimodal calls bounded, even when Wikimedia supplies large images."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((768, 768))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=76)
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def gather_research(prompt: str, root: Path, client: OllamaProxyClient) -> dict:
    """Research first; only vision-confirmed images become subject evidence."""
    try:
        pack = await research_web_references(prompt, root / "reference-candidates", max_images=2)
    except Exception as exc:
        raise ValueError(f"Reference research unavailable: {type(exc).__name__}") from exc
    validated = []
    for ref in pack["references"][:4]:
        try:
            response = await client.chat_json(
                model=VISION_MODEL,
                system=("Judge the actual image pixels against the exact subject. "
                        "Do not treat a matching caption as visual proof. "
                        "Describe visible construction, not speculative hidden geometry."),
                prompt=f"Target: {prompt}. Is the pictured whole object actually the target?",
                images=[_reference_image(root / "reference-candidates" / ref["stored_name"])],
                schema=VisualFinding.model_json_schema(),
                temperature=0,
            )
            finding = VisualFinding.model_validate(response.data)
        except (OSError, ValueError, RuntimeError):
            continue
        if finding.matches_subject and finding.confidence >= 0.65:
            validated.append({
                "source_url": ref.get("source_url"), "image_url": ref.get("image_url"),
                "stored_name": ref["stored_name"], "viewpoint": finding.viewpoint,
                "visible_shape": finding.visible_shape,
                "features": finding.identifiable_features,
            })
        if len(validated) >= 2:
            break
    report = {
        "query": prompt, "reference_candidates": pack["candidate_count"],
        "verified_visual_references": validated,
        "text_sources": [
            {"title": p.get("title"), "url": p.get("url"),
             "extract": str(p.get("extract") or "")[:1000]}
            for p in pack["pages"][:3]
        ],
    }
    (root / "research.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not validated:
        raise ValueError("No visually confirmed subject reference; CAD agents did not start")
    return report


async def generate_agent_cad(prompt: str, model: str = REASONING_MODEL,
                             *, client: OllamaProxyClient | None = None,
                             root: Path, max_parallel: int = 3) -> tuple[CadBlueprint, dict, Path]:
    """Research -> design -> dependency-aware parallel specialists -> assembly review."""
    client = client or OllamaProxyClient()
    root.mkdir(parents=True, exist_ok=True)
    evidence = await gather_research(prompt, root, client)
    design_prompt = json.dumps({"request": prompt, "verified_references": evidence}, ensure_ascii=False)
    response = await client.chat_json(
        model=model,
        system=("You are the chief CAD designer. Research has been verified from image pixels. "
                "First reason about subject identity and silhouette, then split construction "
                "into up to SIX generic component jobs. Specify shared global millimetre "
                "coordinates, realistic dimensions, dependency edges and expected attachments. "
                "Never create disconnected shape islands or invent references. "
                "Do not output geometry, code, or object-specific templates."),
        prompt=design_prompt, schema=DesignPlan.model_json_schema(), temperature=0.1,
    )
    plan = DesignPlan.model_validate(response.data)
    (root / "design-plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    tasks = {task.id: task for task in plan.components}
    groups: dict[str, list[CadPart]] = {}
    semaphore = asyncio.Semaphore(max(1, min(max_parallel, 3)))

    async def build_group(task: ComponentJob, correction: str = "") -> list[CadPart]:
        async with semaphore:
            fixed = {
                name: [p.model_dump() for p in parts]
                for name, parts in groups.items()
            }
            reply = await client.chat_json(
                model=model,
                system=(CAD_SYSTEM_PROMPT + "\nYou are ONE component specialist. "
                        "Respect the chief designer's shared world coordinates. "
                        "Produce only your assigned component, max EIGHT parts using the allowed CAD shapes. "
                        "Do not modify other specialists' work. "
                        "Use name prefix 'COMPONENTID_' for every part."),
                prompt=json.dumps({
                    "full_request": prompt, "overall_silhouette": plan.silhouette,
                    "overall_dimensions_mm": plan.dimensions_mm,
                    "component": task.model_dump(), "other_completed_parts": fixed,
                    "repair_feedback": correction, "verified_reference_descriptions":
                        evidence["verified_visual_references"],
                }),
                schema=CadBlueprint.model_json_schema(), temperature=0.1,
            )
            blueprint = CadBlueprint.model_validate(reply.data)
            if len(blueprint.parts) > 8 or not blueprint.parts:
                raise ValueError(f"{task.id}: invalid subjob part count")
            for part in blueprint.parts:
                if not part.name.startswith(f"{task.id}_"):
                    raise ValueError(f"{task.id}: specialist emitted foreign part: {part.name}")
                for idx in range(3):
                    if abs(part.position_mm[idx] - task.center_mm[idx]) > task.size_mm[idx]:
                        raise ValueError(f"{task.id}: part position outside assigned envelope")
            return blueprint.parts

    for level in topological_levels(plan.components):
        # No group is allowed to read an uncompleted dependency; independent jobs run in parallel.
        results = await asyncio.gather(*(build_group(tasks[name]) for name in level))
        for name, parts in zip(level, results, strict=True):
            groups[name] = parts

    verdict = assembly_report(plan, groups)
    # Bound cost and scope: repair at most two failed attachments; preserve good groups.
    repairs = 0
    for failure in verdict["measurements"][:]:
        if failure["gap_mm"] <= next(
            r.tolerance_mm for r in plan.relationships
            if r.a == failure["a"] and r.b == failure["b"] and r.relationship == "attached"
        ):
            continue
        target = failure["b"]
        if repairs >= 2:
            break
        original = groups[target]
        try:
            groups[target] = await build_group(
                tasks[target],
                correction=(f"Fix detached attachment to {failure['a']}. "
                            f"Measured gap={failure['gap_mm']} mm. "
                            f"Previous attempted parts: {[p.model_dump() for p in original]}. "
                            "Keep world proportions and other components unchanged."),
            )
        except (ValueError, RuntimeError):
            groups[target] = original
        repairs += 1
    verdict = assembly_report(plan, groups)
    parts = [part for task in plan.components for part in groups[task.id]]
    if len(parts) > 32:
        raise ValueError("CAD worker supports at most 32 primitives; reduce component complexity")
    blueprint = CadBlueprint(title=plan.title, parts=parts)
    result = await asyncio.to_thread(build_cad, blueprint, root)
    result["assembly_validation"] = verdict
    result["research_verified"] = True
    result["visual_acceptance_verified"] = False
    result["agent_workflow"] = {
        "mode": "research-first", "component_jobs": len(plan.components),
        "parallelism_limit": min(max_parallel, 3), "assembly_passed": verdict["pass"],
    }
    # Fail closed for downstream promotion without destroying evidence.
    result["ok"] = bool(result["ok"] and verdict["pass"])
    (root / "agent-report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    manifest_file = root / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["agent_assessment"] = {
        "assembly": verdict,
        "research_verified": True,
        "visual_acceptance_verified": False,
        "quality_status": "needs_visual_review" if verdict["pass"] else "assembly_failed",
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return blueprint, result, root
