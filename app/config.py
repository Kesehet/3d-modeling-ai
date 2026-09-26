from __future__ import annotations

import os
from pathlib import Path


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


JOBS_ROOT = Path(env("JOBS_ROOT", "/var/lib/3d-modeling-ai/jobs")).resolve()
WORKER_URL = env("WORKER_URL", "http://worker:8090").rstrip("/")
API_TOKEN = env("THREED_API_TOKEN", "")
OLLAMA_PROXY_BASE_URL = env("OLLAMA_PROXY_BASE_URL", "").rstrip("/")
OLLAMA_PROXY_API_KEY = env("OLLAMA_PROXY_API_KEY", "")
REASONING_MODEL = env("REASONING_MODEL", "gpt-oss:120b")
VISION_MODEL = env("VISION_MODEL", "qwen3-vl:235b-cloud")
BLENDER_BIN = env("BLENDER_BIN", "/usr/bin/blender")
