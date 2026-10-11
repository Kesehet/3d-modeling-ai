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

* Supported primitives: box, cylinder, cone, sphere, tube. No splines,
  lofts, fillets, constraints, gears, threads or freeform meshes yet.
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
