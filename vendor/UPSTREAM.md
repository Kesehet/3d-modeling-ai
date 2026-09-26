# Blender MCP upstream

This repository vendors the runtime source of:

- upstream: https://github.com/djeada/blender-mcp-server
- pinned upstream commit: `428f60cdb819c55c69d67eef681f0318e464e0e9`
- upstream package version at that commit: `0.2.0`
- upstream license: MIT

Vendored files include the package metadata, MIT license, README, and `src/blender_mcp_server/` runtime source needed by production.

The Docker image installs this local vendored copy. It does not clone upstream during a VPS build.

Why vendor instead of following `main`?

Agentic Blender execution is sensitive to tool schema and transport changes. Vendoring gives us reproducible production builds and a local codebase we can modify as the product evolves.

Upgrade procedure:

1. Review upstream changes.
2. Select an explicit upstream commit.
3. Re-vendor the runtime files while preserving the MIT license.
4. Run CI and the MCP Blender smoke test.
5. Update this document and `tasks.md`.

Optional upstream script examples were intentionally not vendored; our production modeling scripts will live in our own controlled library.

Secondary backend to evaluate later:

- https://github.com/bpy-dev/blender-mcp
- its saved-file/CLI work is useful, but it remains an experimental alternative rather than the production default.
