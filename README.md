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
