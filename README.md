# 3D Modeling AI

AI-assisted 3D modeling service built around Blender, MCP, and an Ollama-compatible AI gateway.

## Goal

Turn text and reference images into Blender scenes through an iterative agent loop:

1. Plan the model.
2. Modify a persistent `.blend` scene through Blender MCP.
3. Render checkpoint views.
4. Send those views plus the original references to a vision model.
5. Convert critique into the next Blender edits.
6. Repeat until the model passes visual and geometric QA.
7. Export `.blend`, `.glb`, `.obj`, and later `.stl/.3mf`.

## Upstream Blender MCP

The first backend is based on:

- Repository: `djeada/blender-mcp-server`
- Version/commit pinned by our Docker build: `428f60cdb819c55c69d67eef681f0318e464e0e9`
- License: MIT
- Upstream supports headless Blender execution, rendering, Python execution, and async jobs.

We pin the exact upstream commit so production is reproducible. The integration layer in this repository is ours and can be changed independently.

## Initial architecture

```text
Browser / API client
        |
        v
3D Orchestrator API
        |
        +----> Ollama Proxy / vision + reasoning models
        |
        v
Blender Worker API
        |
        v
MCP client -> blender-mcp-server (stdio)
        |
        v
Blender headless
        |
        +----> .blend checkpoints
        +----> preview renders
        +----> GLB / OBJ / STL / 3MF
```

The Blender worker is kept behind the Docker network. It is not exposed publicly.

## Development

The geometry vocabulary includes polygon mesh parts (`shape: "mesh"`) with local `vertices`
and `faces` containing vertex indices. Primitive, loft and mirrored-cage builders share this
representation; bounded attachment edits can add and transform it. The AI chooses shapes from
references; the runtime contains no object-name templates.

Clients can submit an editable `HardSurfaceCageSpec` to `POST /v1/jobs/{job_id}/design` using
the normal API token. The job must already have a verified reference. The API reserves a new
version, renders nine views, compares against the active model, and performs feature and final
visual review. A regressing candidate cannot replace the active model. Successful submissions
also export `model-vN-design.json` beside the GLB, OBJ, STL and Blender files.

An improved working cage may remain separate from the displayed best model. The editor continues
from that improved geometry; promotion and completion still require their own visual verdicts.

```bash
cp .env.example .env
docker compose up --build
```

API health:

```bash
curl http://localhost:8080/health
```

Worker health is available only inside the Compose network.

## Hostinger VPS

Production follows the same pattern used by the other Hostinger projects:

- `compose.hostinger.yaml`
- Traefik `traefik-proxy` network
- `hostinger/deploy-on-vps` GitHub Action
- Hostinger VPS VM `1058562`
- main-branch deployment
- health checks and bounded container resources

The GitHub repository needs the same `HOSTINGER_API_KEY` secret used by the other VPS deployment workflows. AI credentials should be added as repository/environment secrets rather than committed.

See `tasks.md` and `docs/architecture.md`.

### Parametric modeling parts

The same declarative part vocabulary is available to scene generation and loft/cage attachments:

- `dimensions: [x, y, z]` specifies **full local size before rotation** and overrides legacy `scale`.
  Existing scale-only designs retain Blender's original primitive dimensions (a unit-scale cube is size 2).
- `shape: "lathe"` with `profile: [[radius, z], ...]` revolves a closed material cross-section around local Z.
  Outer and inner profile walls describe hollow forms; zero-radius points close a profile on the axis.
- `shape: "sweep"` with `path: [[x, y, z], ...]` and `radius` builds a smooth curved tube with capped ends.
- `shape: "mesh"` accepts local vertices and indexed faces for arbitrary polygon parts.

The AI chooses profiles, paths, dimensions and placement from the request and references. These operations
contain no subject-specific geometry. Generated candidates still require visual comparison and strict feature
review; successful export alone does not certify a finished model.

## Experimental research-first Ollama CAD agents (PR #155)

This opt-in workflow builds **research -> chief design -> dependency-aware
component agents -> assembly measurement -> CAD export -> three CPU previews ->
vision critique -> targeted component repair**. It is not enabled on production.
To enable only inside the CAD sidecar, set `CAD_AGENT_WORKFLOW_ENABLED=1`; the
existing one-shot CAD engine continues unchanged when the flag is absent.

- The research agent finds Wikimedia images, then checks actual pixels with the
  vision model. Missing verified references stop generation instead of inventing
  a subject.
- The chief designer emits a bounded, generic 3D plan with shared millimetre
  coordinates and component attachment relationships. No car/fan templates.
- Up to three independently assigned component jobs run concurrently, respecting
  explicit dependencies. Each can use bounded loft cross-sections and solids;
  the overall blueprint is limited to 48 named parts.
- The assembly reviewer measures world-space AABB gaps, preserving working
  subassemblies and passing defect measurements to limited repairs. This is a
  conservative broad-phase check, not exact surface contact or collision QA.
- The engine builds real STEP/STL/GLB geometry and generates three CPU-only
  orthographic GLB previews. The Ollama vision critic compares these actual
  previews to pixel-verified reference photos.
- Up to three rounds in total: a failed review can request changes to at most
  two specific components, not an unbounded rewrite. Better candidates are
  preserved; rejected geometry cannot replace an existing dashboard model.
- `research.json`, `design-plan.json`, `agent-report.json`,
  `visual-round-N.json`, preview PNGs and each candidate's exports persist.
  Dashboard export copies also preserve failed CAD QA evidence.
- Quality status distinguishes export validity, assembly screening and
  **Ollama vision-model acceptance**; none imply independent human certification.

Unmocked test (explicitly opt-in using CI with required Ollama credentials):

```sh
python scripts/cad_agent_live_benchmark.py --prompt Prius --output cad-agent-live-artifacts
```

**Still experimental:** Wikimedia coverage may be insufficient for very specific
models; the preview renderer is an approximate software rasterizer, not Blender
or a true z-buffer; the CAD vocabulary doesn't cover arbitrary freeform meshes,
fillets, mechanical constraints or vehicle-grade surface design. Live
prompt-to-model results must be reviewed before merging or deploying.

The explicitly tagged experiment runs a **real Ollama** test from GitHub Actions and
uploads its source references, CAD exports, renders, and review reports for inspection.
