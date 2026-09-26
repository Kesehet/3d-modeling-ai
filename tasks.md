# Tasks

## Phase 0 — Bootstrap and VPS foundation

- [x] Define two-service architecture: public orchestrator + private Blender worker.
- [x] Pin `djeada/blender-mcp-server` to commit `428f60cdb819c55c69d67eef681f0318e464e0e9`.
- [x] Add Blender + MCP Docker image.
- [x] Add local Compose stack.
- [x] Add Hostinger Compose stack using the existing Traefik/VPS pattern.
- [x] Add authenticated public API skeleton.
- [x] Add per-job filesystem workspace.
- [x] Add internal MCP client -> Blender MCP -> headless Blender tool-call path.
- [x] Add an end-to-end smoke test that creates a cube, saves a `.blend`, and renders a PNG through MCP.
- [x] Add CI and Hostinger deployment workflow.
- [ ] Add `HOSTINGER_API_KEY` and `THREED_API_TOKEN` to this repository's GitHub Actions secrets.
- [ ] Set production `OLLAMA_PROXY_BASE_URL` and `OLLAMA_PROXY_API_KEY` secrets.
- [ ] Run the first VPS smoke test and verify the render artifact.

## Phase 1 — Job ingestion

- [ ] Add reference-image upload endpoint with size/type limits.
- [ ] Store image metadata and hashes.
- [ ] Add job cancellation and deletion.
- [ ] Add artifact download endpoints.
- [ ] Add structured job/event history.
- [ ] Add concurrency lock so one worker cannot mutate the same scene twice.

## Phase 2 — Ollama orchestration

- [ ] Add Ollama proxy client.
- [ ] Add planner prompt/schema for modeling stages.
- [ ] Add vision-model call supporting multiple reference/render images.
- [ ] Default vision model to Qwen3-VL through the Ollama proxy.
- [ ] Add structured visual critique schema: object, issue, severity, suggested change.
- [ ] Add iteration budget and stopping rules.
- [ ] Add checkpoint rollback when a visual score/regression gets worse.
- [ ] Persist every model prompt/response for debugging.

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
- [ ] Disable worker outbound Internet access by default.
- [ ] Queue with Redis or Postgres.
- [ ] Store large artifacts in object storage.
- [ ] Automatic cleanup/retention policy.
- [ ] Metrics: generation time, iterations, failures, model-token use.
- [ ] Regression suite with known prompts/reference images.
- [ ] Review upstream MCP updates before changing the pinned commit.
