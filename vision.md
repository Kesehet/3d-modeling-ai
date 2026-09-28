# Vision: Iterative Visual 3D Agent

## Engineering handoff — September 28, 2026 (India)

**Production readiness: NOT YET VERIFIED.** HTTP success and file exports do not establish usable model
quality. Preserve this distinction in product status and release reports.

Repository: `Kesehet/3d-modeling-ai`. Service: `https://3d-modeling-ai.srv1058562.hstgr.cloud`.
The owner asked for API/code work and explicitly asked to avoid browser automation. Preserve the generic,
AI-directed observe/edit/render/compare/keep-or-revert approach; no subject-name geometry templates.

### Release baseline and concrete evidence

- Baseline main: `d30d4b1bdd899d19f1a5331ca0738ad6dff65e77` (PR #55). Hostinger run `36363288553`
  and CI run `36363288554` succeeded. Sync main before continuing; several other fixes landed after PR #46.
- PR #46 fixed immutable version reservations, version-bound visual analysis, regression rollback, bounded
  component research queries, and explicit model-visible JSON schemas. Later main fixes added worker
  concurrency limits, compact quality history, persistent stall budgets, representation switching and
  feature decomposition into coordinated modeling passes. Preserve those changes.
- Latest baseline job: `a5e5d049-1c11-405c-8af6-2d2c14421ce9`, prompt `A Toyota Prius`, v1.
  It is a rejected first body feature, not a finished car. Its QA reports **64 non-manifold edges with
  zero cutters**, dimensions `[1.7833, 4.74, 1.3146]`. Actual front-left/left pixels show a sloping body
  with damaged seams and end surfaces. The cage builder wound longitudinal caps opposite to the adjacent
  surface; normal repair happened only after subdivision/bevel. This is a deterministic geometry defect.
- Other current failed/stalled jobs: `c4a8b04d-9060-4db7-88bd-c120ef953d8c` (Prius),
  `1ca5d6cf-286a-4eee-94cc-d15318b1c1cb` (Veyron), and
  `91db76ab-2348-4b50-8b5e-37906a6529e9` (Polo). Preserve their evidence.
- Older jobs mentioned in previous conversations, including `c32d4cc8-0cfa-475d-aa23-c4f2254d8a05`
  and table `301c0dce-d7e6-4820-8fc9-67b6c9d2d76c`, are absent from the latest dashboard snapshot.
  Do not assume an old process/session or job remains available.

### Recovery increment in this branch

Check this branch's PR/CI/deployment status before assuming these changes are live:

- Correct cage cap winding for both axes and normalize the closed shell immediately after Mirror,
  before subdivision/bevel; modifier failures now surface rather than silently exporting damaged geometry.
- Shared 640-pixel orthographic cameras fit projected geometry bounds per view, update transforms before
  bounds, and use neutral backgrounds. All four generic geometry/assembly paths use the same framing.
- Add/move/remove primitive attachments and cutters; change subdivision/bevel/shading without replacing
  topology; reject no-op edits before rendering. Preserve existing proportional and station-region edits.
  Save full action parameters, show attachments/surface state, explain local-cylinder Z and world axes.
- Local primary-feature QA keeps the strict reference score/confidence gates but does not require missing,
  separately planned parts to exist. Final whole-object QA is bound to the active version, runs when the
  required feature plan completes, persists failures for retry, and cannot be satisfied by a local verdict.
- Child components inherit verified parent images with explicit parent-context provenance and component focus.
- Malformed visual comparisons with no explanation retry/fallback instead of becoming silent negative verdicts.
- Blender CI executes closed cages, a boolean edit, a tapered mirrored/subdivided body, adaptive loft,
  primitive components, assembly, all nine views and GLB/OBJ/STL exports. It checks manifold edges and useful
  image framing. Local Python verification: **149 tests passed**, Ruff passed, all four Blender scripts compile.
- PR #57 CI run `36382423810` also replayed the exact live shell as a test-only fixture:
  **64 non-manifold edges before the fix, 0 after**, with zero loose vertices. All seven real Blender
  cases and exports passed. This establishes the geometry repair, not subject-level visual completion.

### Next work and actual release gate

1. Pass real Blender integration and inspect its renders; fix any topology/camera failures before deployment.
2. After deployment, create fresh jobs for at least three different subject classes. Inspect reference pixels,
   actual rendered pixels, version history and QA. Do not substitute cube exports for model-quality acceptance.
3. Follow multiple kept/rejected edits: versions must increase, baseline bytes must stay unchanged, rejected
   candidates must not appear as accepted, and later feature passes must preserve accepted components.
4. Diagnose remaining AI axis/shape errors from explicit coordinate/bounds context and model reasoning.
   Never lower QA thresholds or hardcode subject geometry to turn failures green.
5. Public-launch operational P0s remain: protect dashboard mutations (currently bypass API-token auth),
   durable queued/resumable generation (currently long HTTP/background tasks), per-job exclusion and owner
   pause/retry controls. Worker process concurrency is bounded, but that does not solve per-job orchestration.
6. Make readiness/release identity observable. Keep viewer, downloads and status on the accepted version and
   distinguish working draft, visual acceptance, and print readiness. Manifoldness is only one print check;
   wall thickness, self-intersections, units and slicer validation still need work.

### Access and continuation

- `GET /dashboard/api`: jobs, plans, quality, accepted artifacts and history. Keep diagnostic output compact.
- `GET /dashboard/renders/{job_id}/{filename}`: download actual model renders without browser automation.
- Authenticated `/v1` API supports jobs, references, research, generation, improvement, status, history and
  artifacts. `THREED_API_TOKEN` and `OLLAMA_PROXY_API_KEY` are production secrets; never print or commit them.
- Owner-gated Live Control issue #31 and `.github/workflows/live-control.yml` run authenticated diagnostics.
  Read supported commands and preserve the owner gate. Avoid interrupting running generation with deployment.
- Main pushes auto-deploy to Hostinger. Merge only the tested PR head and verify deployment plus live behavior.
- Use focused changes in `app/main.py`; avoid a sweeping refactor while repairing the pipeline. Key modules:
  `app/artifacts.py`, `app/cage_edits.py`, `app/rendering.py`; regression tests include
  `tests/test_production_regressions.py` and `tests/blender_integration.py`.

## Goal

Build a general 3D modeling agent that turns text and reference images into a progressively improving Blender model.

The system must not depend on object-family hardcoding such as "cars use these coordinates" or "chairs use this topology." The LLM and vision models decide what the subject is, what matters visually, and what should change. Deterministic code provides safe editing primitives, geometry validation, versioning, rendering, and rollback.

The core loop is:

**observe -> diagnose -> choose one small edit -> apply -> render -> compare -> keep/revert -> repeat**

The model should improve through many bounded decisions rather than trying to emit an entire correct mesh in one response.

## Principles

1. **LLM owns intent; Blender owns geometry.**
   - The model describes the visual correction it wants.
   - Code translates that correction into bounded, deterministic geometry edits.
   - The model should not routinely emit long lists of raw XYZ coordinates.

2. **Vision is the critic.**
   - Every important edit is checked against the actual reference images.
   - Relative improvement and absolute completion are separate decisions.
   - "Better than before" means keep the candidate.
   - "Feature complete" requires strict visual acceptance.

3. **Preserve the best known model.**
   - Never destroy the current accepted version in place.
   - Every edit produces a new candidate version.
   - If the candidate is worse or uncertain, revert by leaving the baseline version active.

4. **Ask small questions.**
   - Do not ask an LLM to generate an entire hard-surface object in one JSON payload when the model can instead decide one useful correction.
   - Examples:
     - widen this control section;
     - lower this section;
     - move this section along the length axis;
     - insert a control section between two existing sections;
     - reshape one profile point;
     - move/resize/rotate one cutter;
     - remove a bad cutter.

5. **No subject-specific geometry rules.**
   - The agent can understand that an object is a vehicle, chair, creature, appliance, shoe, etc.
   - That understanding belongs in the model's reasoning and feature plan.
   - The geometry engine only knows generic operations and safety constraints.

6. **Visual progress is cumulative.**
   - A feature may require many improvements.
   - A candidate can be accepted as a better baseline even when the feature is not finished.
   - Do not count useful partial progress as a failed feature attempt.

## Pipeline

### 1. Reference acquisition and verification

- Obtain multiple useful views when possible.
- Reject clearly unrelated references.
- Record which references are trusted.
- Preserve the original reference images and their analysis.

### 2. Visual understanding

The vision model should establish:

- subject identity;
- orientation;
- overall silhouette;
- major proportions;
- major visible parts;
- repeated parts;
- high-information views;
- visually important landmarks and transitions;
- what must be true for an unfamiliar viewer to recognize the result.

This stage describes the subject faithfully. It should not be constrained by the current Blender representation.

### 3. Feature decomposition

Create an ordered feature plan.

Typical categories are generic, not object-specific:

- primary silhouette/body;
- major secondary masses;
- openings/cutouts;
- repeated components;
- visible attachments;
- surface transitions;
- identity-critical detail.

Each feature has:

- visual acceptance criteria;
- target region;
- protected/owned scope;
- dependencies;
- whether it should be built in place or as an isolated component.

### 4. Initial representation

Create the simplest useful editable representation.

For hard-surface continuous forms this may be a mirrored cage. For other subjects another representation may be appropriate.

The first version is not expected to be correct. Its job is to establish a manipulable baseline.

### 5. Visual diagnosis

Render the active version from multiple stable camera views.

Give the references and current renders to vision and ask:

1. What is the single highest-impact visible error?
2. Which generic edit operation can address it?
3. Which views should improve if the edit works?
4. What geometry should be protected?

The answer should be one bounded edit, not a replacement mesh.

### 6. Deterministic edit

Apply one edit using a small generic editing vocabulary.

Initial cage-edit vocabulary:

- reshape an entire longitudinal station;
- reshape one profile point;
- insert a station by interpolation;
- remove an unnecessary station;
- adjust one boolean cutter;
- remove one bad cutter.

Edits are relative to the current model's dimensions where possible.

### 7. Candidate render

Build a new version and render the same diagnostic views.

Do not overwrite the baseline.

### 8. Visual comparison

Compare:

- trusted references;
- baseline renders;
- candidate renders.

Vision decides whether the candidate is a clear net improvement.

If better:
- candidate becomes the active baseline.

If worse or uncertain:
- baseline remains active.

### 9. Feature acceptance

Separately run strict feature QA.

A feature is complete only when it visibly satisfies its acceptance criteria at high confidence.

Important:

**candidate improvement != feature completion**

An improved candidate can remain active while the feature stays in progress.

### 10. Repeat

Continue small edits until:

- the active feature passes;
- then move to the next feature.

If several edits fail to improve the model, escalate:

- choose a different generic operation;
- revise the local representation;
- or re-plan the representation.

Do not blindly repeat the same failed edit family.

### 11. Final technical cleanup

Only after visual modeling is satisfactory:

- topology cleanup;
- manifold checks;
- normals;
- disconnected parts;
- wall thickness where required;
- export validation;
- STL/OBJ/GLB generation;
- print-oriented repair when the intended use is 3D printing.

## State that must be persisted

Every iterative pass should record:

- baseline version;
- candidate version;
- active feature;
- visual diagnosis;
- chosen edit action;
- action parameters;
- expected visual effect;
- comparison result;
- whether candidate was kept or reverted;
- strict feature QA result;
- model used for the decision.

This history should make it possible to understand why a model changed and avoid repeating failed actions.

## Success criteria for the architecture

The architecture is working when the same code can:

1. start from an imperfect generic representation;
2. look at references and renders;
3. select a small edit;
4. improve the visible result;
5. preserve the improvement;
6. continue improving without object-specific Python rules.

The first benchmark is the Toyota Prius because its current failure is well understood.

The architecture only counts as general when the same loop can then be applied to a materially different subject, such as furniture or a humanoid, without changing the geometry rules.
