# CPU-only CAD branch experiment

Branch: `experiment/cpu-cad-ollama-20261011`. This is **not** a production
deployment or a proven general autonomous model maker.

## Why a separate service?

The existing app uses Blender MCP and multimodal quality gates. We do not
replace those on main. This optional service uses the existing authenticated
MediaPitch Ollama proxy for reasoning and open-source `build123d`/OpenCascade
to construct and export true CAD solids on CPU. No generative 3D GPU model runs
on our server. The source and workflow architecture are informed by the
MIT-licensed text-to-cad project
[here](https://github.com/sagarshedage7/text-to-cad), but no upstream code
is vendored: its desktop-oriented pipeline executes model-generated Python,
which isn't appropriate to run with public prompts on our server.

## Run the isolated service

Use **branch checkout**, not production main. Supply secrets via environment:

```sh
export THREED_API_TOKEN_B64="$(printf '%s' "$THREED_API_TOKEN" | base64 -w0)"
export OLLAMA_PROXY_API_KEY_B64="$(printf '%s' "$OLLAMA_PROXY_API_KEY" | base64 -w0)"
docker compose -f compose.cad-experiment.yaml up --build -d
curl http://127.0.0.1:18084/health
curl -X POST http://127.0.0.1:18084/v1/cad-experiments \
  -H "Authorization: Bearer $THREED_API_TOKEN" -H 'Content-Type: application/json' \
  -d '{"prompt":"A tubular spacer, 50 mm outer diameter, 20 mm inner diameter, 35 mm length"}'
```

A successful POST returns validation evidence and authenticated download paths
for STEP (editable BRep), STL and GLB. No front-end changes have been made.
The port binds to localhost only; use an SSH tunnel for remote access.

## GitHub Actions

`cad-cpu-e2e` uses a **fake LLM response** but real build123d/OpenCascade.
It tests the HTTP API, export/re-import, watertight per-part meshes,
retries, invalid schema rejection, and emits inspectable files.

`cad-container` checks the separate Docker image can build and execute CAD.

`cad-real-ollama` runs only after explicitly dispatching the workflow with
`live_ollama=true`, with repository secret `OLLAMA_PROXY_API_KEY`.
It asks the real configured Ollama model to independently plan and construct a
three-bladed fan; it saves exported artifacts even if visual review is still
required. Until the workflow is on the default branch, a **single tagged push** whose
commit message contains `[cad-live]` can run the live Ollama test on this
experiment branch. Ordinary pushes and pull requests never use Ollama secrets.

## Limitations / promotion criteria

* Supported primitives: box, cylinder, cone, sphere, tube, plus generic polygon
  profile extrusions and multi-section rectangular lofts. No splines, fillets,
  mechanical constraints, gears, threads, booleans from AI, or freeform meshes yet.
* A successful STEP re-import demonstrates geometrical integrity, **not**
  correct proportions, connected mechanical design, fan recognizability,
  collision-free joins, physical feasibility, or printable assembly.
* Each emitted object is an individually named GLB node; GLB is a preview
  mesh, not an editable parametric CAD feature tree.
* Jobs run serially with up to 3 retries and a maximum of 32 bounded parts.
  GPU is not needed, but BRep operations use significant CPU/RAM.
* The quality gate for replacing existing production must include an
  **actual unmocked Ollama run** plus independent visual inspection of
  downloaded GLB/STEP and honest missing-part reporting.
* Workflows do not deploy or alter the production server.

## Unified dashboard and live application

Once this PR is merged, the regular **New Job** dialog will offer CPU CAD
and Blender AI. Selecting CPU CAD creates the same existing job ID, status,
history, GLB live model viewer and downloadable artifacts, but dispatches
modeling to an internal CPU-only build123d/OpenCascade container. STEP is
included in the regular Downloads panel.

The API endpoint is `POST /v1/jobs/{job_id}/cad` (Bearer token); the
dashboard dispatches to `POST /dashboard/jobs/{job_id}/cad`. Both start a
tracked, asynchronous job and return immediately. The dashboard refreshes
the status automatically. Errors are displayed, previous export versions
are retained, and success explicitly means **geometry export passed, visual
acceptance still required**. CPU CAD jobs do not offer the older Blender
auto-improve action.

The deployment adds a CPU-limited `cad` sidecar on the private Docker
network. It uses its **own** volume so the CAD engine doesn't need access to
Blender worker files. The main API receives exports over internal HTTP,
stores them atomically in the standard job workspace, and cleans up the
sidecar's temporary run. The existing Blender worker continues unchanged.


## Reference-aware CPU CAD and reliable state reporting (PR #145)

The existing **New Job** dialog supports up to three PNG/JPEG/WebP reference images.
They are ingested through the same validated `/v1/jobs/{id}/references`
endpoint as Blender jobs before CAD generation starts. The CPU CAD adapter
selects previously uploaded or verified reference files, downsamples them to
JPEG, and passes the bounded base64 images over the private Docker network to
the Ollama-driven CAD sidecar.

`POST /v1/jobs/{id}/cad` and the dashboard support:
- `reference_mode`: `exact_match`, `shape_guidance` (default), or `inspiration`.
- `assembly_mode`: `single` (default) or `independent` for truly separate deliverables.
  **The caller sets this mode.** The LLM cannot silently opt out of connected-part QA.

Every successful geometry export also produces headless CPU render evidence:
`renders/model-vN-front-left.png` and `renders/model-vN-top.png`. When references
are supplied, an independent vision-model call receives the original reference
images FIRST and the rendered evidence LAST. The review records
`reference_match_score`, `passed`, `subject_recognizable`, and specific findings.
A failed review is fed back to the planner for up to three bounded corrective
attempts; unavailable/invalid visual review is **unverified**, not a pass.

`assembly_qa` is measured on actual generated CAD solids and rejects disconnected
AABB part groups with mm gap evidence, except explicitly independent requests.
It never uses keywords such as "wheel" or "blade" to determine placements.
**Limitation:** overlap of bounding boxes is NOT proof of physical contact,
interference-free parts, watertight combined assembly, dimensional fidelity, or
mechanical feasibility. This first-stage QA is conservative rather than final.

When structural or reference visual checks still fail, the job status is
`state=failed`, stage `cad_needs_assembly_repair` or `cad_needs_reference_repair`,
and the last valid STEP/GLB/STL and render evidence remain available. Export
integrity and subject quality never share the same pass flag. A job with no
reference images remains `cad_needs_visual_review`; it cannot claim visually
verified quality. Existing Blender generation is separate and unchanged.

Before claiming a broadly reliable general model maker, add real narrow-phase
contact/penetration checks, photo-to-geometry multi-view registration, and
fresh independent visual reviews across unrelated unseen objects.

### Live model-quality findings (not an acceptance claim)

An unmocked reference-photo sedan test on the experimental branch identified a
serious geometry/recognition gap despite validated STEP. The model generated
loft and profile parts but a vertical cabin panel and disconnected geometry,
so assembly or reference QA correctly rejected it. Both numerical and visual
failure evidence now enter the next bounded attempt; no subject-specific hardcoding
is accepted. A good export is **not** a finished or recognizable object.
