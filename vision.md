# Vision: Iterative Visual 3D Agent

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
