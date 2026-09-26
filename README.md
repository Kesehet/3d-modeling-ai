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
