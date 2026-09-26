from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

from fastapi import FastAPI, HTTPException
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field

from .config import BLENDER_BIN

app = FastAPI(title="3D Modeling AI Blender Worker", version="0.1.0")

ALLOWED_TOOLS = {
    "blender_python_exec",
    "blender_python_exec_async",
    "blender_job_status",
    "blender_job_cancel",
    "blender_job_list",
    "blender_render_still",
    "blender_render_animation",
}


class ToolCall(BaseModel):
    tool: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


def _server_env() -> dict[str, str]:
    result = os.environ.copy()
    result["BLENDER_BIN"] = BLENDER_BIN
    result["BLENDER_MCP_HEADLESS"] = "1"
    return result


@app.get("/health")
async def health() -> dict:
    blender = shutil.which(BLENDER_BIN) or (BLENDER_BIN if os.path.exists(BLENDER_BIN) else None)
    mcp_server = shutil.which("blender-mcp-server")
    if not blender or not mcp_server:
        return {"ok": False, "blender": blender, "mcp_server": mcp_server}

    try:
        proc = subprocess.run(
            [BLENDER_BIN, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        version = proc.stdout.splitlines()[0] if proc.stdout else "unknown"
    except Exception as exc:
        return {"ok": False, "blender": blender, "mcp_server": mcp_server, "error": str(exc)}

    return {"ok": True, "blender": version, "mcp_server": mcp_server}


@app.post("/v1/mcp/call")
async def call_mcp(payload: ToolCall) -> dict:
    if payload.tool not in ALLOWED_TOOLS:
        raise HTTPException(status_code=403, detail="Tool is not enabled on the headless worker")

    arguments = dict(payload.arguments)
    if payload.tool in {
        "blender_python_exec",
        "blender_python_exec_async",
        "blender_render_still",
        "blender_render_animation",
    }:
        arguments["transport"] = "headless"

    params = StdioServerParameters(
        command="blender-mcp-server",
        args=[],
        env=_server_env(),
    )

    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(payload.tool, arguments=arguments)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MCP tool call failed: {exc}") from exc

    content = []
    for item in result.content:
        text = getattr(item, "text", None)
        content.append(text if text is not None else str(item))

    return {
        "tool": payload.tool,
        "is_error": bool(getattr(result, "isError", False)),
        "content": content,
    }
