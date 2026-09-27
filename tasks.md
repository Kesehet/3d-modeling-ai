# 3D Modeling AI — 17-Level Capability Roadmap

This file is the working definition of "done" for the project. A level is only considered complete when its acceptance tests pass on the deployed VPS, not merely when code exists.

## Current focus

**Immediate target:** P0 production recovery only. Do not spend autonomous cycles on roadmap breadth until the generic pipeline can repeatedly turn verified references into recognizable geometry for unseen subjects.

**Primary production gate:** Toyota Prius. It must pass reference acquisition -> modeling -> multi-angle QA -> refinement without subject-specific geometry code.

**Unseen holdout set immediately after Prius:** Bugatti Veyron, teapot, desk fan, bicycle, one household appliance, one furniture item, and one permitted stylized character. A pass means recognizable geometry from multiple views, not merely a successful job status.

**Benchmark subject:** Pikachu remains a regression target because it exposes proportion, silhouette, appendage, face-placement, asymmetry, multi-angle, and iteration problems quickly. It is a benchmark, not the intended final limitation of the system.

### Production-ready fast path

Work these in order. The hourly production-fix automation should choose the first failing item, gather live evidence, fix it, deploy, and re-test before moving lower.

- [x] Hard-gate modeling on usable verified references.
- [x] Preserve and expose exact reference images and every generated iteration for audit.
- [x] Reject unrelated reference candidates and keep automatic reference packs visually generation-coherent.
- [x] Use an AI modeling director for initial strategy and subsequent revise/rebuild decisions.
- [ ] Make Toyota Prius produce a recognizable accepted model on the deployed VPS.
- [ ] Add pairwise visual comparison: candidate iteration vs current accepted best, using the same reference pack.
- [ ] Reject/rollback any candidate that is materially worse than the accepted best.
- [ ] Detect semantic stalls even when SceneSpec JSON changes superficially.
- [ ] Force a strategy switch after repeated non-improving attempts instead of spending more rounds on the same approach.
- [ ] Add an autonomous-run budget ledger: model calls, render attempts, elapsed time, accepted improvements, rejected candidates, and stop reason.
- [ ] Recover stale `running` jobs automatically and guarantee that a dead worker cannot leave a job permanently running.
- [ ] Add a per-job mutation lock so two workers/refinement requests cannot alter the same scene concurrently.
- [ ] Pass the unseen holdout set without adding prompt/object-specific geometry recipes.
- [ ] Run a mixed production batch twice consecutively with no manual intervention and no false-success statuses.

---

## P0 — STOP-THE-LINE: quality regression recovery

**Status: ACTIVE.** Do not promote additional capability-level work until the generic generation/refinement path stops making models worse.

### Observed regression
- Generic `v2` can be materially worse than `v1` while still ending in `generic_refinement_complete`.
- A real holdout prompt, **"A toyota prius"**, produced an unrecognizable wedge/slab with a few wheel-like cylinders while the job still reported `generic_refinement_complete`.
- The Pikachu regression can collapse into a few blob primitives and antenna-like rods instead of retaining defining character parts.
- The desk-lamp regression can collapse into disconnected stacked primitives instead of a round base + connected angled neck + dome shade.
- The recent recovery work has become **benchmark-overfit**: generic generation now contains subject-specific geometry assemblies for the fixed benchmark families (characters, lamps, sneakers, chairs and quadruped robots). Passing those five prompts is therefore not sufficient evidence that the autonomous pipeline generalizes.
- Current live regression mainly proves that known benchmark prompts can be patched; it does not prove that unseen subjects remain recognizable.

### Generalization recovery — AI-directed architecture
- [x] Make a multimodal **modeling director** choose the initial strategy instead of Python object-family rules.
- [x] After each render, let the modeling director choose: **accept / revise procedural / rebuild mesh**.
- [x] Feed the exact prompt, reference images, current renders and AI-generated subject inventory directly to the director.
- [x] Allow the AI to rewrite the full SceneSpec when the current approach is bad; do not force surgical edits or semantic object-count preservation.
- [x] Remove generic-path geometry recipes for characters/Pikachu, lamps, sneakers, chairs and quadruped robots.
- [x] Remove prompt-specific spatial guidance from the generic path; retain only universal coordinate/execution constraints.
- [x] Keep Python responsible for orchestration and safety only: schemas, request-size control, retries, Blender execution, files and job state.
- [x] Compress **vision-request copies** of images while preserving the original saved reference files.
- [x] Replace the retired Qwen vision route with current frontier multimodal routing: Kimi K3 -> Mistral Large 3 -> Gemma 4 31B -> reasoning fallback.
- [x] Normalize common model JSON wrapper variations for the adaptive mesh schema instead of requiring one model-specific response shape.
- [ ] Pass the real deployed Toyota Prius holdout with the AI director making all geometry/strategy decisions.
- [ ] Add additional unseen holdouts: teapot, desk fan, bicycle and one household appliance.
- [ ] Do not add subject-specific geometry code to make any holdout pass.

### P0 recovery tasks
- [x] Freeze unrelated roadmap feature work until the generic path generalizes.
- [x] Keep references visible per job so failures can be audited against what the model actually saw.
- [x] Keep multi-angle renders for every iteration.
- [x] Add adaptive continuous-mesh reconstruction as a strategy available to the AI director.
- [x] Preserve previous artifacts/versions for inspection; do not silently overwrite evidence from earlier passes.
- [x] Make malformed model responses retry through the configured model chain rather than killing the job on the first model.
- [ ] Make the deployed live-quality suite pass with the AI-directed architecture.

### P0 acceptance
A refinement may only become the accepted/latest model when it preserves required parts and does not regress visibly or structurally from the previous accepted version. The Pikachu and articulated desk-lamp benchmarks must remain recognizable from front, side, rear and isometric views.

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
- [x] Add a deploy regression that validates a complex quadruped robot through v1 + visual critique + v2, while retaining the cube only as a hidden infrastructure heartbeat.
- [x] Add automatic rollback to the previous healthy image when deployment smoke/quality tests fail.

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
- [x] Keep benchmark runtime under 90 seconds on the current VPS (live benchmark ~10.3 seconds including 9 renders + exports).
- [ ] Add at least two non-character regression subjects. Complex quadruped robot live benchmark added as first quality regression.

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
- [x] Persist parameter sets per iteration.
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
- [x] Rank whole-object references ahead of incidental part/detail results and add deterministic view-oriented searches.
- [x] Keep automatic reference packs visually coherent around one canonical identity/generation.
- [ ] Classify usable references by view: front, side, rear, 3/4, detail, dimensions.
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
- [x] Safe rod-between-two-points helper for straight limbs, handles, struts, antennas and connectors.
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
- [x] Semantic object names and structured multi-object SceneSpec.
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
- [x] Experimental automated repair pass using join + voxel remesh + cleanup; still requires stronger QA before print-ready status is trusted.
- [ ] Mandatory QA gate before "print ready".

### Acceptance
Known broken test meshes are detected correctly and at least common manifold/normal/loose-geometry failures are automatically repaired.

---

## Level 12 — Vision critique and self-correction loop

**Goal:** render, inspect, change, and re-render autonomously.

### Capabilities
- [x] Structured multimodal visual critique pipeline (Gemma 4 primary with model fallbacks and schema normalization).
- [x] Persist vision reports.
- [x] Keep uploaded/researched references in critique cycles.
- [x] Parameter schema for benchmark corrections.
- [x] Reasoning-model translation from critique to bounded geometry parameters.
- [x] Multi-iteration benchmark refinement endpoint.
- [x] Stop early when no high-severity issues and very few medium issues remain.
- [ ] Compare each candidate iteration directly against the current accepted best using matched views and the same references.
- [ ] Detect regressions and rollback/reject the candidate before it becomes latest.
- [x] Detect a basic stalled loop when the revised generic SceneSpec is unchanged.
- [ ] Detect semantic stalls when specs differ but rendered geometry/quality does not materially improve.
- [ ] Strategy switch when repeated edits do not improve.
- [x] Generalize visual correction beyond Pikachu: multimodal critique can drive a full revised safe SceneSpec and re-render a generic v2.

### Acceptance
Iteration N+1 must make a measurable/visible correction requested by vision QA, with complete history of critique → parameters → renders.

---

## Level 13 — Autonomous research-driven modeling

**Goal:** make research a standard stage of every relevant job.

### Capabilities
- [x] Initial automatic research provider.
- [x] Research manifest stored per job.
- [x] Research stage in generic job pipeline.
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
- [x] Parse job request into a validated generic SceneSpec for primitive blockouts.
- [ ] Research references and facts.
- [ ] Choose modeling strategy (generic safe primitive strategy implemented; automatic strategy selection still pending).
- [ ] Build blockout.
- [x] Render nine-view checkpoints for generic and benchmark builds.
- [x] Run multimodal visual critique through the current vision fallback chain; retired Qwen routes are no longer production defaults.
- [x] Apply safe corrections through bounded benchmark tuning or revised SceneSpec.
- [ ] Repeat with stopping rules (severity and unchanged-spec stops implemented; regression rollback still pending).
- [ ] Run geometry/print QA.
- [ ] Repair mandatory failures.
- [ ] Produce final preview renders.
- [ ] Export BLEND / GLB / OBJ / STL / 3MF as appropriate.
- [ ] Package provenance, QA report and iteration history.
- [x] Present outputs in a gallery-first UI with per-job render/detail/download views.

### Acceptance
A request such as "make a 12 cm printable stylized character figurine from references" completes research → modeling → iterative QA → print QA → exports without developer intervention.

---

# Immediate implementation sprint

## Sprint A — Stabilize benchmark
- [x] Confirm the corrected Workbench Pikachu builder completes on VPS.
- [x] Keep nine-view checkpoint render under 90 seconds (live ~10.3 seconds including exports).
- [x] Confirm BLEND + GLB/OBJ/STL + QA report are generated on the live VPS.
- [x] Add live Pikachu regression and verify all nine PNGs through the public app.

### Live QA baseline

The first deployed benchmark produced 23 mesh objects, 12 non-manifold edges and 0 loose vertices. The new experimental print-repair pass attempts to fuse those visible parts through voxel remeshing, then recalculates connected components and manifold status. Wall thickness, self-intersections, overhangs and slicer validation remain mandatory future gates.

## Sprint B — Research + refinement
- [x] Add Wikimedia/Wikipedia research provider.
- [x] Persist research references and source/license metadata.
- [x] Add bounded Pikachu tuning schema.
- [x] Add one-to-three-pass visual refinement endpoint.
- [x] Add dashboard "Research + Improve" control.
- [x] Show iteration/tuning/status history in a dedicated dashboard tab.

## Sprint C — Generalization
- [x] Add first reusable safe declarative primitive scene builder (sphere/cube/cylinder/cone/torus, materials, transforms, cameras, exports).
- [ ] Build a second character benchmark.
- [ ] Build and live-verify a hard-surface product benchmark using the generic SceneSpec engine. A complex hard-surface quadruped robot quality regression is now wired into deploy validation.
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
- [ ] Add job cancellation. [x] Add permanent job deletion for non-running jobs.
- [x] Add artifact listing/download endpoints.
- [x] Add structured job/event history.
- [ ] Add concurrency lock so one worker cannot mutate the same scene twice.
- [ ] Reconcile stale `running` jobs on startup/heartbeat when no active worker lease exists.

## Phase 2 — Ollama orchestration

- [x] Add Ollama proxy client using Bearer auth and `/api/chat` with `/api/generate` fallback.
- [x] Add planner schema and endpoint for modeling stages.
- [x] Add vision-model call supporting multiple reference/render images.
- [x] Default vision model to Qwen3-VL through the Ollama proxy.
- [x] Add structured visual critique schema: object, issue, severity, suggested change.
- [x] Add bounded auto-improve budgets and explicit stop reasons.
- [ ] Add per-run token/model/render budget telemetry so autonomous loops can prove useful progress per cost.
- [ ] Add checkpoint rollback/rejection when pairwise visual comparison says a candidate is worse.
- [x] Persist planner/vision model responses and usage metadata for debugging.

## Phase 3 — Blender modeling tools

- [ ] Build a safe high-level Blender script library for primitives, booleans, bevels, modifiers, curves, text, materials, cameras, lights, and transforms. **Primitive/camera/material/export subset is now implemented through the safe SceneSpec engine.**
- [x] Prefer high-level deterministic tools over unconstrained generated Python for generic blockouts via a validated SceneSpec schema.
- [x] Add persistent `.blend` checkpoint naming for benchmark iterations.
- [x] Add first automatic scene inspection: object count, dimensions, non-manifold edges, loose vertices; deeper modifier/component inspection remains future work.
- [x] Standardize nine-view rendering for generic and benchmark builders.
- [x] Standardize fast checkpoint rendering with Workbench.
- [ ] Add final Cycles render option when hardware allows.

## Phase 4 — Visual self-correction loop

- [x] Keep original reference images attached to generic and Pikachu critique cycles.
- [ ] Render checkpoints after blockout, proportions, secondary forms, details, materials, and final QA.
- [x] Ask vision model to compare references vs current renders for generic and Pikachu refinement.
- [x] Convert critique into explicit geometry changes through revised bounded parameters or a revised validated SceneSpec.
- [x] Re-render after each accepted generic/Pikachu edit batch.
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

- [x] Prompt form; reference upload API exists (dashboard upload UI still pending).
- [ ] Intended-use selector: 3D printing / rendering / game asset.
- [ ] Target dimensions.
- [ ] Live stage/progress timeline.
- [x] Render gallery includes iteration render sets; dedicated Iterations tab exposes version history.
- [ ] Web GLB viewer.
- [ ] Human feedback box during generation.
- [ ] Downloads for BLEND/GLB/OBJ/STL/3MF (BLEND/GLB/OBJ/STL working; 3MF pending).

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

