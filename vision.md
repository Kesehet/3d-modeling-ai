# Vision: Iterative Visual 3D Agent

## Engineering handoff — September 29, 2026

**No usable Prius has been verified. Production model quality is still blocked.** The owner is asking
for the actual model-making software urgently. Work through APIs/code, never browser automation. Do not
turn a passing build, a GLB export, or a local feature verdict into a claim that the full model is ready.

Repository: `Kesehet/3d-modeling-ai`. Live: `https://3d-modeling-ai.srv1058562.hstgr.cloud`.
**Owner instruction: do not implement security/auth changes.** Security draft PR #64 was closed without merge or deployment. Focus on model-making.

Sync main before continuing; other agents have contributed changes. Never replace their work blindly.

### Verified baseline

- Main `f8acfe19e86e124809349a60c96640213329d160` (PR #62) deployed successfully in Hostinger run
  `36468397074`. PR #62 CI `36468097718` passed Python, compose and real Blender integration.
- PR #46 fixed immutable version reservations, version-bound QA, rollback and bounded search queries.
- PR #57 fixed cage cap winding, normal repair before subdivision/bevel, camera framing and generic
  attachment/cutter edits. The exact live Prius shell had **64 non-manifold edges before / 0 after**
  in real Blender CI. That repair did not make the shape a Prius.
- PR #58/#61 improved generic reference discovery. PR #62 added explicit vision response schemas,
  required decisions/explanations, raw verification logs and owner-gated diagnostic log downloads.
- Live retest after #62 recovered **one** valid table reference for `c0f70b5c-cae9-4d17-84f4-04748711303e`.
  Its earlier zero-reference barrier is resolved; no table model has yet been verified.
- Mug `d0a6eb4e-d115-420e-8144-19d564eb2468` found **five** references, but produced a flat slab.
  The AI described a 1.6-unit upright body and incorrectly put that height in Y station positions;
  actual profile Z spans only 0.2. This is a planning/coordinate error, not a Blender export error.
- Prius `a5e5d049-1c11-405c-8af6-2d2c14421ce9`: baseline v1 retained, v2 rejected. Both are poor.
  Actual v2 pixels show a flat shell without recognizable vehicle structure. Raw comparison replies
  were valid negative judgments inside `{ "properties": { ...instance fields... } }`, which the
  application missed. Diagnostics: workflow `36468455029`, artifact `10990052511`.

### Current recovery change

PR #63 merged as `7b9551cd14c44eb91e88817e9b7e0bac58c1d4ba`. CI `36569560206` passed
170 Python tests plus real Blender geometry/export checks. Deployment `36569799658` is being
verified before the live Prius rebuild; do not infer model success from these checks.

- Recover the observed `properties` response envelope only when it contains instance fields matching
  the requested schema. A copied schema is rejected, never treated as a positive judgment.
- Let the visual director explicitly choose a horizontal mirrored cage or a closed loft along X/Y/Z.
  The old generic `build_mesh` action always became a horizontal cage regardless of the intended form.
- Separate visual reference interpretation from coordinate construction for primary forms: vision writes
  a concrete silhouette/dimension brief; the existing reasoning model writes the geometry. Validate
  cage XYZ bounds against the brief before rendering and feed errors to fallback planners.
- Preserve strict local/final QA, immutable versions and keep/revert. No subject-name geometry templates.

### Next action and release gate

1. Deploy the recovery change and rebuild the existing Prius through `/dashboard/jobs/{id}/generate`
   with auto research enabled and automatic rounds initially zero. Inspect actual rendered pixels and
   the saved dimension brief/spec, then run bounded improvements. Never silently accept a bad candidate.
2. Verify full object assembly, not just the primary shell: secondary parts, child jobs, placement and
   final QA must complete. Local feature acceptance is separate from whole-object acceptance.
3. Retest mug and table to ensure the geometry engine remains generic. Vertical lofts still need careful
   testing for later cavity/cutout features. Do not assume a cylindrical draft is a completed mug.
4. Watch representation switching and feature ownership. Current strict dependencies can strand later
   components behind a failed primary mass. Improve geometry planning rather than lowering QA thresholds.
5. Continue only model-making work. The owner explicitly rejected security work on September 29;
   do not reopen or deploy PR #64. Any job scheduling changes must directly unblock generation.
6. Update this handoff with actual live results, version numbers, PR/run IDs and remaining blockers.

### Access and continuation

- `GET /dashboard/api` contains jobs, quality, feature plans and accepted artifacts. Keep output compact.
- `GET /dashboard/renders/{job_id}/{filename}` retrieves actual pixels without browser automation.
- `/v1` endpoints require `THREED_API_TOKEN`; never retrieve, print or commit production secrets.
- Owner-gated Live Control issue #31 supports inspect/research/generate/improve commands via production
  workflows. Preserve its owner gate. Recent captures include the latest 16 logs ordered by timestamp.
- Main pushes deploy to Hostinger. Verify CI and deploy completion; avoid restarting active generation.
- Key code: `app/main.py`, `app/ollama.py`, `app/cage_edits.py`, `app/rendering.py`, `app/feature_tasks.py`.
  Tests: `tests/test_production_regressions.py`, `tests/test_ollama.py`, `tests/blender_integration.py`.

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
