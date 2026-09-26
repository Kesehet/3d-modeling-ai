# Blender MCP upstream

Primary upstream:

- https://github.com/djeada/blender-mcp-server
- pinned commit: `428f60cdb819c55c69d67eef681f0318e464e0e9`
- upstream version at that commit: 0.2.0
- license: MIT

The production Dockerfile clones this exact revision and installs it into the runtime.

Why not track `main`?

Agentic Blender execution is sensitive to tool schema and transport changes. Pinning gives us reproducible production builds and lets us review upstream upgrades deliberately.

Secondary backend to evaluate later:

- https://github.com/bpy-dev/blender-mcp
- useful saved-file/CLI work, but currently treated as an experimental alternative rather than the production default.
