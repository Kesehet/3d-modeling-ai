# 3D Modeling AI — 17-Level Capability Roadmap

This file is the working definition of "done" for the project. A level is only considered complete when its acceptance tests pass on the deployed VPS, not merely when code exists.

## Current focus

**Immediate target:** make Levels 0–4 reliable, establish the Level 12 visual correction loop, and pull forward useful pieces of Levels 10–11 (exports and geometry QA).

**Benchmark subject:** Pikachu remains the first visual regression target because it exposes proportion, silhouette, appendage, face-placement, asymmetry, multi-angle, and iteration problems quickly. It is a benchmark, not the intended final limitation of the system.

---

## Level 0 — Reliable infrastructure

**Goal:** the end-to-end service can execute Blender work reliably.

### Capabilities
- [x] Public dashboard is reachable.
- [x] Public job creation is available during prototype phase.
- [x] Private Blender worker communicates through MCP.
- [x] Headless Blender runs inside Docker on the Hostinger VPS.
- [x] Job workspaces persist artifacts.
- [x] Public gallery and file downloads exist.
- [x] Deployment smoke test creates a Blender file and PNG.
- [x] Traditional tabbed dashboard: Overview / New Job / Gallery / Files / Jobs.
- [x] VPS CPU limits match the actual 2-vCPU host.
- [ ] Add a deploy regression that always validates the current benchmark, not only the cube.
- [ ] Add automatic rollback to the previous healthy image when deployment smoke tests fail.

### Acceptance
A new deployment must create a job, execute Blender through MCP, save a .blend, render a valid PNG, and serve the artifact publicly.

---

## Level 1 — Primitive recognizable object

**Goal:** create recognizable subjects from deterministic basic geometry.

### Capabilities
- [x] Spheres / cones / cylinders / curves / simple polygon meshes.
- [x] Basic materials and colors.
- [x] Camera creation and fixed QA views.
- [x] Multi-angle rendering.
- [x] Deterministic Pikachu benchmark builder.
- [ ] Keep benchmark runtime under 90 seconds on the current VPS.
- [ ] Add at least two non-character regression subjects.

### Acceptance
At least three known benchmark prompts produce recognizable, repeatable results and downloadable source files.

---

## Level 2 — Accurate proportions and silhouette

**Goal:** move from "recognizable" to materially closer geometry.

### Capabilities
- [x] Make the benchmark builder parameter-driven rather than fully hard-coded.
- [ ] Tune head/body ratio from visual critique.
- [ ] Tune ear length/taper/angle.
- [ ] Tune eye spacing/scale.
- [ ] Tune cheek position/scale.
- [ ] Tune arm/foot proportions.
- [ ] Tune tail size/position.
- [ ] Persist parameter sets per iteration.
- [ ] Add silhouette-focused render mode / masks for vision comparison.

### Acceptance
A refinement iteration must be able to change visible proportions without rewriting the entire Blender script.

---

## Level 3 — Reference-based modeling

**Goal:** model from evidence instead of model memory alone.

### Capabilities
- [x] User reference-image upload.
- [x] Multi-image vision endpoint.
- [x] Automatic Wikimedia/Wikipedia research provider with source/license metadata.
- [x] Download free-license research images into the job reference pack.
- [x] Save a research manifest for reproducibility.
- [ ] Add a second general web/image-search provider.
- [ ] Rank references by usefulness: front, side, rear, detail, dimensions.
- [ ] Detect duplicate/near-duplicate references.
- [ ] Let users approve/remove references before modeling.
- [ ] Extract dimensional facts from trustworthy textual sources where available.

### Acceptance
Given only a prompt, the system creates a traceable reference pack and includes those references in visual QA.

---

## Level 4 — Complex appendages and asymmetry

**Goal:** reliably create non-blob features.

### Capabilities
- [x] Curved mouth geometry.
- [x] Polygon + solidify + bevel lightning-tail technique.
- [x] Independent left/right body parts.
- [ ] Improve tail topology and attachment.
- [ ] Bent/tapered appendage helper.
- [ ] Pose-aware asymmetry.
- [ ] General curve-to-mesh appendage helper.
- [ ] Benchmark horns, wings, handles, cables, and branches.

### Acceptance
The subject retains its defining appendages in front, side, rear and isometric views without obvious floating/intersection failures.

---

## Level 5 — Organic mesh editing

**Goal:** stop relying on primitive stacking for organic subjects.

### Capabilities
- [ ] Editable mesh builder helpers.
- [ ] Extrude / inset / loop-cut operations.
- [ ] Subdivision workflow.
- [ ] Proportional editing/deformation.
- [ ] Remesh / voxel-remesh option.
- [ ] Sculpt/base-mesh path.
- [ ] Seam reduction between major body forms.
- [ ] Mesh-density budget.

### Acceptance
At least one organic benchmark is built as a mostly continuous mesh with visibly smoother transitions than the primitive version.

---

## Level 6 — Pose and stance reasoning

**Goal:** generate believable non-neutral poses.

### Capabilities
- [ ] Armature/rig creation or import.
- [ ] Joint and limb constraints.
- [ ] Center-of-mass / support-foot check.
- [ ] Pose presets.
- [ ] Prompt-driven pose planner.
- [ ] Tail/appendage pose response.
- [ ] Pose QA from multiple views.

### Acceptance
Generate three distinct poses that are readable, balanced, and do not visibly self-intersect.

---

## Level 7 — Surface detail and expression

**Goal:** make subjects expressive and specific.

### Capabilities
- [ ] Eyes/eyelids/brows as controllable systems.
- [ ] Mouth/expression presets.
- [ ] Fingers/toes/claws.
- [ ] Markings and secondary forms.
- [ ] Wrinkles/creases where relevant.
- [ ] Detail-level planner tied to intended use.

### Acceptance
Vision QA can identify and request a detail change and the next iteration visibly applies it.

---

## Level 8 — Materials and presentation

**Goal:** produce useful preview/final renders.

### Capabilities
- [x] Fast Workbench checkpoint renderer.
- [x] Material-color previews.
- [ ] Stable Eevee final-preview path on headless VPS.
- [ ] Cycles option where hardware/time allows.
- [ ] Material library: plastic, matte, metal, glass, wood, cloth.
- [ ] Lighting presets.
- [ ] Turntable render/video.
- [ ] Transparent-background product previews.

### Acceptance
Checkpoint renders are fast enough for iteration while final previews are presentation quality.

---

## Level 9 — Multi-object scenes

**Goal:** reason about multiple objects and contacts.

### Capabilities
- [ ] Object hierarchy and semantic names.
- [ ] Relative placement constraints.
- [ ] Contact/collision checks.
- [ ] Scale consistency.
- [ ] Character + prop benchmark.
- [ ] Simple environment benchmark.

### Acceptance
A prompted multi-object scene has plausible scale, placement and visible contact relationships.

---

## Level 10 — Printable model readiness

**Goal:** produce fabrication-oriented outputs.

### Capabilities
- [x] BLEND output.
- [x] Add GLB export attempt.
- [x] Add OBJ export attempt.
- [x] Add STL export attempt.
- [ ] 3MF export.
- [ ] Join/union print parts where required.
- [ ] Apply transforms and real-world mm scale.
- [ ] Flat/stable base generation.
- [ ] Print orientation recommendation.
- [ ] Slicer integration smoke test.

### Acceptance
A benchmark model exports to STL/3MF, opens in a slicer, has correct intended size, and passes mandatory geometry checks.

---

## Level 11 — Automatic mesh repair and print QA

**Goal:** detect and repair geometric failures.

### Capabilities
- [x] Generate first scene/mesh QA JSON report.
- [x] Record mesh-object count, loose vertices, non-manifold edges and scene bounds.
- [ ] Self-intersection detection.
- [ ] Connected-component analysis.
- [ ] Minimum-wall-thickness analysis.
- [ ] Normal orientation check/fix.
- [ ] Duplicate/degenerate geometry cleanup.
- [ ] Overhang/support estimate.
- [ ] Automated repair pass.
- [ ] Mandatory QA gate before "print ready".

### Acceptance
Known broken test meshes are detected correctly and at least common manifold/normal/loose-geometry failures are automatically repaired.

---

## Level 12 — Vision critique and self-correction loop

**Goal:** render, inspect, change, and re-render autonomously.

### Capabilities
- [x] Structured Qwen3-VL critique schema.
- [x] Persist vision reports.
- [x] Keep uploaded/researched references in critique cycles.
- [x] Parameter schema for benchmark corrections.
- [x] Reasoning-model translation from critique to bounded geometry parameters.
- [x] Multi-iteration benchmark refinement endpoint.
- [x] Stop early when no high-severity issues and very few medium issues remain.
- [ ] Compare latest iteration specifically against previous iteration.
- [ ] Detect regressions and rollback.
- [ ] Detect stalled loops.
- [ ] Strategy switch when repeated edits do not improve.
- [ ] Generalize correction operations beyond Pikachu parameters.

### Acceptance
Iteration N+1 must make a measurable/visible correction requested by vision QA, with complete history of critique → parameters → renders.

---

## Level 13 — Autonomous research-driven modeling

**Goal:** make research a standard stage of every relevant job.

### Capabilities
- [x] Initial automatic research provider.
- [x] Research manifest stored per job.
- [ ] Research stage in generic job pipeline.
- [ ] Source quality ranking.
- [ ] Text fact extraction for dimensions/materials/features.
- [ ] Reference-view classification.
- [ ] Research refresh when evidence is inadequate.
- [ ] Provenance shown in dashboard.

### Acceptance
A prompt with no uploaded references triggers a useful research pack, modeling plan, and traceable evidence without manual intervention.

---

## Level 14 — Multi-image reconstruction

**Goal:** reconstruct a user-supplied real object from several photographs.

### Capabilities
- [x] Multiple image upload.
- [ ] View classification.
- [ ] Relative camera/view inference.
- [ ] Shape correspondence across images.
- [ ] Base-mesh generation from images.
- [ ] Multi-view consistency refinement.
- [ ] Scale inference from a known dimension/reference object.

### Acceptance
Four or more photos of a benchmark object produce a clearly matching 3D approximation.

---

## Level 15 — Existing model editing

**Goal:** modify uploaded 3D assets without rebuilding everything.

### Capabilities
- [ ] Upload/import BLEND.
- [ ] Import OBJ/STL/GLB.
- [ ] Scene/object inspection.
- [ ] Localized modifications.
- [ ] Preserve unrelated geometry/materials.
- [ ] Revision snapshots and rollback.
- [ ] Re-export in requested formats.

### Acceptance
Given an existing model and a localized instruction, the requested area changes while unrelated geometry remains materially unchanged.

---

## Level 16 — Mechanical / precision modeling

**Goal:** support functional dimension-driven design.

### Capabilities
- [ ] Parametric dimension constraints.
- [ ] Holes/bosses/threads.
- [ ] Hinges and moving clearances.
- [ ] Snap fits.
- [ ] Tolerance presets by printing process.
- [ ] FreeCAD/OpenSCAD integration.
- [ ] Interference analysis.
- [ ] Mechanical benchmark: functional opening/latching object.

### Acceptance
A dimensioned functional benchmark assembles/moves as intended and respects specified tolerances.

---

## Level 17 — Production autonomous pipeline

**Goal:** one prompt/reference set produces a usable deliverable with minimal intervention.

### Pipeline
- [ ] Understand request.
- [ ] Research references and facts.
- [ ] Choose modeling strategy.
- [ ] Build blockout.
- [ ] Render checkpoints.
- [ ] Run visual critique.
- [ ] Apply corrections.
- [ ] Repeat with stopping rules.
- [ ] Run geometry/print QA.
- [ ] Repair mandatory failures.
- [ ] Produce final preview renders.
- [ ] Export BLEND / GLB / OBJ / STL / 3MF as appropriate.
- [ ] Package provenance, QA report and iteration history.
- [ ] Present all outputs in dashboard.

### Acceptance
A request such as "make a 12 cm printable stylized character figurine from references" completes research → modeling → iterative QA → print QA → exports without developer intervention.

---

# Immediate implementation sprint

## Sprint A — Stabilize benchmark
- [ ] Confirm the corrected Workbench Pikachu builder completes on VPS.
- [ ] Keep nine-view checkpoint render under 90 seconds.
- [ ] Confirm BLEND + GLB/OBJ/STL attempts + QA report appear in Files.
- [ ] Add live regression after deploy once runtime is stable.

## Sprint B — Research + refinement
- [x] Add Wikimedia/Wikipedia research provider.
- [x] Persist research references and source/license metadata.
- [x] Add bounded Pikachu tuning schema.
- [x] Add one-to-three-pass visual refinement endpoint.
- [ ] Add dashboard "Research + Improve" control.
- [ ] Show critique and tuning history in dashboard.

## Sprint C — Generalization
- [ ] Extract reusable high-level Blender geometry library.
- [ ] Build a second character benchmark.
- [ ] Build a hard-surface product benchmark.
- [ ] Replace subject-specific tuning with general edit operations.

---

# Existing engineering backlog

# Tasks

## Phase 0 — Bootstrap and VPS foundation

- [x] Define two-service architecture: public orchestrator + private Blender worker.
- [x] Pin and vendor `djeada/blender-mcp-server` at commit `428f60cdb819c55c69d67eef681f0318e464e0e9`.
- [x] Add Blender + MCP Docker image.
- [x] Add local Compose stack.
- [x] Add Hostinger Compose stack using the existing Traefik/VPS pattern.
- [x] Add authenticated public API skeleton.
- [x] Add per-job filesystem workspace.
- [x] Add internal MCP client -> Blender MCP -> headless Blender tool-call path.
- [x] Add an end-to-end smoke test that creates a cube, saves a `.blend`, and renders a PNG through MCP.
- [x] Add CI and Hostinger deployment workflow.
- [x] Add `HOSTINGER_API_KEY`, `THREED_API_TOKEN`, and `OLLAMA_PROXY_API_KEY` repository secrets.
- [x] Hardcode Ollama proxy base URL to `https://mediapitch.in/ollama-proxy`.
- [ ] Run the first VPS smoke test and verify the render artifact.

## Phase 1 — Job ingestion

- [x] Add reference-image upload endpoint with size/type limits.
- [x] Store image metadata and SHA-256 hashes.
- [ ] Add job cancellation and deletion.
- [x] Add artifact listing/download endpoints.
- [ ] Add structured job/event history.
- [ ] Add concurrency lock so one worker cannot mutate the same scene twice.

## Phase 2 — Ollama orchestration

- [x] Add Ollama proxy client using Bearer auth and `/api/chat` with `/api/generate` fallback.
- [x] Add planner schema and endpoint for modeling stages.
- [x] Add vision-model call supporting multiple reference/render images.
- [x] Default vision model to Qwen3-VL through the Ollama proxy.
- [x] Add structured visual critique schema: object, issue, severity, suggested change.
- [ ] Add iteration budget and stopping rules.
- [ ] Add checkpoint rollback when a visual score/regression gets worse.
- [x] Persist planner/vision model responses and usage metadata for debugging.

## Phase 3 — Blender modeling tools

- [ ] Build a safe high-level Blender script library for primitives, booleans, bevels, modifiers, curves, text, materials, cameras, lights, and transforms.
- [ ] Prefer high-level deterministic tools over unconstrained generated Python.
- [ ] Add persistent `.blend` checkpoint naming.
- [ ] Add automatic scene inspection: objects, dimensions, modifiers, mesh stats.
- [ ] Add multi-view render helper: front, left, right, top, isometric.
- [ ] Standardize preview rendering with Workbench/Eevee.
- [ ] Add final Cycles render option when hardware allows.

## Phase 4 — Visual self-correction loop

- [ ] Keep original reference images attached to every critique cycle.
- [ ] Render checkpoints after blockout, proportions, secondary forms, details, materials, and final QA.
- [ ] Ask vision model to compare references vs current renders.
- [ ] Convert critique into explicit geometry changes.
- [ ] Re-render after each accepted edit batch.
- [ ] Detect stalled loops and switch strategy instead of repeating the same edit.
- [ ] Add human pause/approve/redirect controls.

## Phase 5 — 3D-printing QA

- [ ] Add STL and 3MF export.
- [ ] Validate manifold geometry.
- [ ] Validate normals and loose geometry.
- [ ] Detect self-intersections.
- [ ] Measure dimensions and scale in mm.
- [ ] Estimate minimum wall thickness.
- [ ] Add configurable printability rules by material/nozzle.
- [ ] Add simple overhang/support analysis.
- [ ] Refuse "complete" status if mandatory geometry checks fail.

## Phase 6 — Organic/image-to-3D base mesh

- [ ] Evaluate Hunyuan3D as optional base-mesh generator.
- [ ] Evaluate Rodin or another hosted generator only if licensing/cost is acceptable.
- [ ] Import generated base mesh into Blender.
- [ ] Retopology/remesh cleanup pass.
- [ ] Run the same vision/refinement/printability pipeline after import.
- [ ] Let planner choose procedural vs base-mesh workflow.

## Phase 7 — Product UI

- [ ] Prompt + reference image form.
- [ ] Intended-use selector: 3D printing / rendering / game asset.
- [ ] Target dimensions.
- [ ] Live stage/progress timeline.
- [ ] Render gallery for each iteration.
- [ ] Web GLB viewer.
- [ ] Human feedback box during generation.
- [ ] Downloads for BLEND/GLB/OBJ/STL/3MF.

## Phase 8 — Hardening and scale

- [ ] One disposable Blender worker container per job.
- [ ] CPU/memory/time quotas per job.
- [x] Disable worker outbound Internet access by default.
- [ ] Queue with Redis or Postgres.
- [ ] Store large artifacts in object storage.
- [ ] Automatic cleanup/retention policy.
- [ ] Metrics: generation time, iterations, failures, model-token use.
- [ ] Regression suite with known prompts/reference images.
- [ ] Review upstream MCP updates before changing the pinned commit.

