# Vision: Iterative Visual 3D Agent

## Verified live recovery and remaining release gate — October 11, 2026

- #139 merged as `2238b841c2b5e0d8269a407f4b2b1d0799931a7c`.
  PR CI `38089890302`, main CI `38090156741`, and deployment
  `38090156878` passed. Public health confirmed the exact deployed revision.
  Python: 276 tests; Ruff/compile and real Blender render/export integration pass.
- Parent `eabc4e96-60e0-469e-8582-8a64e2d1cf22` resumed from v21.
  Downrod/canopy v41 passed on the first attempt. Exact pre-export evaluated
  geometry proof verified motor_housing and bottom_hub_cap against v21.
  Frozen blade installation v42 passed taper/feature/preservation QA, but final
  integrity rejected blade-to-housing gaps/intersections. Coordinated repair
  reopened blade-mount-sockets; v44 strict QA rejected misplaced bottom notches.
  No whole-model success is verified. Keep accepted geometry and strict gates.
- New ordinary-prompt autonomous jobs on this revision, with no supplied geometry:
  fan `796ca348-f29e-4d55-8f10-c4f95a3fa607` ("Create a modern ceiling fan with three blades.")
  and table `df9abf62-6c9c-4a14-baa1-27f51dea65c4` ("Create a simple wooden dining table with four legs.").
  Both use automatic research and 30 bounded improvement rounds. Tabletop v1
  passed; both full jobs still require final whole-object/physical QA verification.
- Fresh fan diagnostic `38090843381` captured round 7: canopy/downrod accepted,
  motor housing rejected after three procedural attempts; adaptive planning began.
  This fresh job has not passed the release gate. Table diagnostic `38090993623`
  was requested; its final verdict is not yet verified.
- Local executor connection failed during live monitoring; production generation
  continues independently. GitHub live-control inspection remains available.
  Recovery snapshot workflow `38090664591` captured v44/current round 6.
  Inspect the fresh runs and final active-artifact verdicts before claiming release.
  The recovered rendering baseline is not print-ready (existing non-manifold edges).

## Freeze evaluated protected parts during repair — October 11, 2026

- #138 merged/deployed as `e10a62af67def7893e91acd75563447e73edf2e2`;
  CI and deployment `38089017979` passed; live health confirmed. Scoped proof
  correctly refused v38 housing: base topology/transforms match but evaluated
  runtime post-export hashes differed, while bottom cap matched. Reopening the
  saved artifacts confirmed housing surfaces remained identical. No failed proof
  was overridden; it was captured at the wrong lifecycle stage.
- Exporters can alter temporary evaluated meshes/depsgraph caches. Proof now
  snapshots the geometry being saved BEFORE export, rather than transient state.
  Reconstructed SceneSpecs also cannot guarantee identical evaluated modifier output.
  Scoped repair now imports the actual protected mesh objects/materials/modifiers
  from the accepted baseline blend and restores their original world matrices.
  Only AI-selected repair parts are rebuilt; independent signatures still verify
  the result and visual/physical QA remain strict. No subject-specific geometry.
- Python: 276 tests, Ruff/compile pass. Real Blender regression compares unchanged,
  deliberately changed, and frozen-source restored parts with render/export QA.
  The downloaded real v21 protected housing/cap also pass exact evaluated proof
  when restored and measured before export; no coordinate tolerance was relaxed.
  Resume downrod repair from v21 after deployment; inspect proof and final gate.

## Scoped repair mesh proof — October 11, 2026

- #137 merged/deployed as `b16c2ec283b3ff1f51aa15f94e08fcb606062cb3` (deployment
  `38088299728` passed; health confirmed). Scoped v35 retained motor_housing and
  bottom_hub_cap, but visual preservation QA falsely claimed the housing vanished.
- Real Blender inspection of downloaded v21/v35 proves matching world matrices,
  vertex coordinates and polygon indices for both objects. Housing digest:
  `2aac2e160b97fa73c55a2bb9d4993147d45cab6682f3236ccc815d73a71993b8`.
  Frozen geometry is intact; rejection is not evidence of deletion.
- Follow-up measures evaluated Blender geometry/topology/world matrices/diffuse
  materials for selected protected names from the exact baseline and candidate.
  Review receives provenance-bound scoped proof. Only named objects are proven;
  occlusion/intersection/contact and unlisted parts still require independent QA.
- Python: 276 tests, Ruff/compile pass. Real Blender cases verify unchanged versus
  deliberately scaled protected geometry. Deploy, requeue downrod from v21 and
  repeat final whole-object integrity. No full release success is claimed yet.

## Scoped parent repair — October 11, 2026

- #135 merged as `0a5c586fc89a96d0b8cb920cd1f091dddcf57569`; CI and VPS deploy
  `38087382258` passed. Health confirmed that revision. Recovery v31 installed
  all three blades successfully: strict QA confirms outward widening (~1.97x),
  visible pitch and clean edges; protected parent audit passed.
- Whole-model repair then targeted canopy/downrod contact. v32-v34 were rejected
  because complete replacement SceneSpecs dropped/rewrote accepted housing and
  sockets. Parent v21 was preserved. Downrod now failed after three attempts;
  blade accepted, optional seams blocked. Model is NOT release-ready.
- Scoped repair follow-up requires AI-authored exact editable_object_names from
  current SceneSpec. Code freezes all other objects and their cutters, including
  omitted ones; explicit selected-part removal and new contact geometry remain
  supported. Strict feature/preservation/final QA remain mandatory.
- Regression proves frozen omitted/rewritten parts and cutters survive, active
  edits/removal work, invalid scopes fail, and input specs remain unchanged.
  Python/Ruff/compile validation passes. Deploy and retry downrod from preserved
  v21, then reinstall frozen blade and run final integrity. Do not invent success.

## Measured assembly retry feedback — October 11, 2026

- Main `a42b1d5` (#134) is deployed; live health confirmed it. It grounds strict
  directional taper QA in signed Blender cross-sections. Parent
  `eabc4e96-60e0-469e-8582-8a64e2d1cf22` still has four accepted required features.
- Live v30 was correctly rejected: signed negative-axis installation measured
  root width 0.48, distal width 0.24 (narrows outward); criterion requires the
  opposite. Planner incorrectly described the frozen child's end orientation.
  Preservation passed. No full model or final integrity success is claimed.
- Follow-up supplies provenance-bound measured profiles in persisted rejected
  installation context, including after manual resets. Coordinator reasons from
  measured direction and revisits signed axis/rotation; scaling cannot reverse
  taper. Frozen geometry and strict QA thresholds are preserved.
- Measurements now use evaluated Blender meshes including modifiers/curves rather
  than raw vertices. Full real Blender 4.4.3 integration passed with a tapered
  offset child, unapplied array modifier, three radial copies, negative local
  axis, nonuniform scale and rotation; geometry/renders/GLB/OBJ/STL checks passed.
  Python suite: 271 passed, Ruff/compile passed.
- A recovery command requeued blade installation from preserved parent v21.
  Deploy follow-up and resume bounded auto; inspect measured direction and all
  final physical/visual verdicts. A recovery is not the fresh autonomous release
  gate; also test a different unseen subject before declaring the project done.

## Recovery and Boolean shading — October 7, 2026

- PR #124 merged as `aa6cc2474a67dbd381decbdb1d0f55dc9dca17aa`; all CI and
  deployment checks passed, and production health confirmed the revision.
- Recovery of fresh job `b04739bb-5c08-45c2-9e16-3d812b278bae` from preserved v6
  passed the interface at v10 and created motor child `8f8943bb-df48-4b1c-9e75-2d2016ec9c88`.
  Child features passed but whole-object integrity rejected socket seams/intersections.
  Repeated procedural candidates did not improve; the parent safely stopped before
  installation with housing failed and blades blocked. This remains NOT a finished fan.
- Render inspection shows strong socket shading streaks. The generic builder smooths
  all polygons without splitting normals at Boolean rims. A generic correction marks
  sharp joins after subtraction while retaining smoothing on curved bands. Geometry,
  QA thresholds and AI-selected dimensions are unchanged. Real Blender regression checks
  the resulting corner normals on a flat surface surrounding a cavity, plus watertight
  geometry and exports. Production impact still requires deployment and a new live test.

## Fresh production test and next generic fix — October 7, 2026

- PR #123 merged as `bc644f5479cdd413fe94dd2814f7f0fc8f13eb86`. CI included real
  Blender geometry/exports and passed. Deploy run `37656916573` passed the live
  API -> worker -> MCP -> Blender smoke test. `/health` confirmed that exact revision.
- Fresh autonomous fan job `b04739bb-5c08-45c2-9e16-3d812b278bae` started through
  normal research/generation with 30 auto rounds; no supplied design or geometry.
  One reference passed verification. The primary feature passed at v6 after the
  director switched to adaptive loft. This is NOT a finished fan.
- It stopped at round 8: mounting-interface task failed three strict QA attempts;
  the two component tasks remained blocked. Failed v7-v9 were rejected and v6 retained.
- Verified handoff defect: adaptive_representation_review diagnosed a distinct
  attachment/interface, but refine_generic_scene replaced its decision with generic
  queued-feature text. adaptive_mesh_revision_planned therefore lacked the diagnosis,
  and the editor repeatedly tapered the accepted body instead of adding the interface.
- Next generic fix preserves refine_mesh director decisions through the editor handoff.
  A regression verifies that exact diagnosis reaches the editor. Adaptive retries now
  also use the existing partial-progress policy: retain a relatively improving candidate
  only when protected geometry has not regressed, while keeping the feature unfinished.
  Two regressions verify retention/rejection and prevent false completion. 248 local
  tests, Ruff and compile checks pass. Preserve strict QA thresholds.
- After deploying this fix, retry the failed feature from preserved v6 and resume auto.
  A recovery pass is diagnostic progress; a later fresh no-intervention success remains
  the release gate. Do not count command completion or a model export as full success.

## Engineering handoff — October 7, 2026

- Reviewed main `9879f60` (#122), with #120 radial normalization/tests and #121
  active assembled-artifact QA. No complete fresh autonomous fan success is verified.
- Main CI/deploy failed before steps ran because hosted runners were not acquired.
  CI run `37364684464` was rerun; Python and compose jobs passed, Blender job pending
  when this handoff was written. Do not assume deployment from these results.
- Local baseline: 244 tests and Ruff passed. New generic lifecycle regression found
  that retry tasks bypassed unmet dependencies and in-place finish work could run
  before a reopened frozen component was reinstalled. Retry tasks now wait as pending
  until dependency acceptance. The regression exercises real repair preparation,
  scheduling, component promotion, and original-feature re-evaluation; external
  Blender/LLM responses are simulated. With the fix, 245 local tests passed.
- Production `/health` was healthy. Existing fan
  `dd8b0c31-c6df-4e22-8cb1-76741a1d1beb` remained at `feature_retry_ready_v24`,
  four accepted features and one unresolved finish. This is partial progress.
- Deployment now bakes SOURCE_REVISION into the image and reports it via `/health`
  so a fresh production test can be tied to the actual deployed code.
- Next: pass CI/real Blender checks, deploy and verify revision, then start a NEW
  autonomous fan job with no supplied design/manual geometry. Retain references,
  version-bound renders, feature QA and final whole-object/integrity verdicts.

## Engineering handoff — September 30, 2026

**No usable Prius has been verified. Production model quality is still blocked.** The owner is asking
for the actual model-making software urgently. Work through APIs/code, never browser automation. Do not
turn a passing build, a GLB export, or a local feature verdict into a claim that the full model is ready.

Repository: `Kesehet/3d-modeling-ai`. Live: `https://3d-modeling-ai.srv1058562.hstgr.cloud`.
**Owner instruction: do not implement security/auth changes.** Security draft PR #64 was closed without merge or deployment. Focus on model-making.
**Owner reiterated: improve the general model-maker; do not hard-code objects to manufacture successful results.**

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
170 Python tests plus real Blender geometry/export checks. Deployment `36569799658` succeeded.
The subsequent Prius rebuild produced a poor procedural v3, then failed with HTTP 502: the photo
brief estimated XYZ `[6.5, 8, 3.2]`, while all geometry planners proposed narrower body shapes.
Treating a perspective estimate as an exact measurement incorrectly rejected those candidates.
Latest private diagnostic capture: workflow `36609442466`, artifact `11053016528`.

- Recover the observed `properties` response envelope only when it contains instance fields matching
  the requested schema. A copied schema is rejected, never treated as a positive judgment.
- Let the visual director explicitly choose a horizontal mirrored cage or a closed loft along X/Y/Z.
  The old generic `build_mesh` action always became a horizontal cage regardless of the intended form.
- Separate visual reference interpretation from coordinate construction for primary forms: vision writes
  a concrete silhouette/dimension brief; the existing reasoning model writes the geometry. Validate
  cage XYZ bounds before rendering and feed construction errors to fallback planners.
- Preserve strict local/final QA, immutable versions and keep/revert. No subject-name geometry templates.

PR #65 merged as `e7494552d289ca29ab88fc4198789d0d6a2c7d59`; deployment `36610075715` succeeded.
It makes photo dimensions advisory, requires new primary cage plans to declare their
own intended XYZ bounds, and checks actual coordinates against that declaration. It also restores
the previous active model and its quality record if regeneration fails during a strategy switch.
The live Prius v4 generated successfully but remained a rounded shell, without recognizable vehicle
structure. Relative review preferred v4 over working v1, but comparison against displayed v3 rejected
it and incorrectly discarded progress in the working construction track. Four subsequent automatic
rounds did not finish the model. This is not a production-quality Prius.

PR #66 merged as `fcdf8f3c251c0f9e3294ab9e809eadb266ec20a2`; CI `36643792103` and deployment `36643960960` succeeded:

- Arbitrary declarative polygon mesh parts work in primitive, loft and cage builders, and bounded
  attachment edits. This supports shaped panels and components beyond the original primitive vocabulary.
- The visual model diagnoses a visible edit; the reasoning model translates the diagnosis into geometry.
  The editor receives actual profile points during primary-form work, previously hidden from it.
- Improving working cage geometry can advance without replacing the better displayed model. Quality
  and completion remain attached to the displayed version until a candidate wins promotion.
- Primary-form review includes front, side and rear coverage. Review receives other feature ownership
  so later components are not confused with the active pass. New plans forbid criteria dependent on
  geometry owned by later features.
- A generic `/v1/jobs/{id}/design` API accepts a complete editable cage/parts design and runs the same
  render, compare, feature and final review pipeline. Dashboard API alias exists. No canned subject
  design is included in the generator or used as evidence that autonomous generation works.
- An optional TRELLIS public-image experiment reached a ZeroGPU quota limit; no external reconstruction
  backend was added. Do not rotate hosts/accounts to bypass that limit.

### Live geometry findings and current follow-up

Normal API generation after #66, without supplied designs or canned models:

- Table `c0f70b5c-cae9-4d17-84f4-04748711303e`: v1/v2 export correctly but vertical parts protrude
  through the tabletop. The planner used full leg height as Blender scale, doubling its actual height.
  The schema had no dimension/half-scale contract. v2 failed strict local review and v1 was retained.
- Mug `d0a6eb4e-d115-420e-8144-19d564eb2468`: v2 correctly uses Z-up but the reasoning model wrote square
  ring coordinates while describing them as circular. Actual pixels show a rounded rectangular block.
  Quality remained failed, score 0.1. Do not claim these objects are finished.

Follow-up under development (verify merge and deployment):

- Full local `dimensions` override legacy `scale` for scene parts; existing saved scales retain their meaning.
- Generic `lathe` revolves an AI-authored closed radius/height material profile, including hollow walls.
  Generic `sweep` follows an AI-authored curved centerline with a round section and capped ends.
  These are geometric operations, not object templates. They work in all three builders and attachment edits.
- Procedural generation reviews the active feature before whole-object completion, retains relative improvement
  without falsely accepting the feature, and checks already-present features before rewriting geometry.
- Procedural revision uses visual diagnosis plus the reasoning model for coordinates. Representation switching
  respects the director's choice; failed upright lofts are no longer forced into a horizontal cage.
- Real Blender CI must verify a hollow revolved profile by raycast, a capped curved sweep, and exact full-size
  dimensions after rotation. Unit/lifecycle tests cover progress, regressions and unchanged strict QA thresholds.

### Next action and release gate

1. Deploy the follow-up and rebuild the existing Prius through `/dashboard/jobs/{id}/generate`
   with auto research disabled (two verified reference photos exist) and automatic rounds initially zero. Inspect actual rendered pixels and
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
