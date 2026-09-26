# Architecture

## Why MCP is inside the worker

The selected Blender MCP server uses stdio as its MCP transport. We therefore do not expose the MCP process directly to the Internet.

The internal Blender worker launches the MCP server as a subprocess, talks to it using the MCP client protocol, and exposes a very small private HTTP surface to the orchestrator.

This gives us:

- a normal HTTP API for the product;
- real MCP tool calls for Blender execution;
- no public arbitrary-Python endpoint;
- a clean place to add tool allowlists and per-job policy;
- the option to swap Blender MCP implementations later.

## Containers

### api

Public-facing FastAPI application.

Responsibilities:

- authenticate requests;
- create job workspaces;
- persist job state;
- call the Ollama proxy;
- decide modeling stages;
- compare visual critiques;
- call the Blender worker;
- expose final artifacts.

### worker

Private FastAPI process plus:

- Blender;
- MCP Python client;
- pinned `djeada/blender-mcp-server`;
- headless Blender execution.

Only the API container can reach it through `worker-net`.

## Workspace

```text
/jobs/<job-id>/
  request.json
  status.json
  references/
  scene/
  renders/
  exports/
  logs/
```

Every meaningful checkpoint should save a new `.blend` file. Never overwrite the only known-good scene.

## Planned orchestration loop

```text
prompt + refs
  -> planning model
  -> Blender MCP edit
  -> save checkpoint
  -> render front/side/top/isometric
  -> vision model compares refs vs renders
  -> structured critique
  -> planner converts critique to Blender actions
  -> repeat
  -> geometric QA
  -> final exports
```

Vision checkpoints should happen after meaningful modeling stages rather than after every Blender tool call.

## Security model

Blender Python execution is powerful. The worker therefore:

- runs as an unprivileged container user;
- drops Linux capabilities;
- is not attached to the public Traefik network;
- uses an internal Docker network with no direct Internet access in production;
- exposes only an allowlisted set of headless MCP tools;
- stores generated data in a dedicated jobs volume.

This is still not a perfect sandbox. Model-generated Python must be treated as untrusted code. A later hardening phase should create one short-lived worker container per job.
