from __future__ import annotations

import base64
import binascii
import os
from pathlib import Path


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def secret_env(name: str) -> str:
    raw = env(name)
    if raw:
        return raw

    encoded = env(f"{name}_B64")
    if not encoded:
        return ""
    try:
        return base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise RuntimeError(f"{name}_B64 is not valid base64 UTF-8") from exc


JOBS_ROOT = Path(env("JOBS_ROOT", "/var/lib/3d-modeling-ai/jobs")).resolve()
WORKER_URL = env("WORKER_URL", "http://worker:8090").rstrip("/")
API_TOKEN = secret_env("THREED_API_TOKEN")

# This is intentionally fixed to our existing MediaPitch Ollama proxy.
OLLAMA_PROXY_BASE_URL = "https://mediapitch.in/ollama-proxy"
OLLAMA_PROXY_API_KEY = secret_env("OLLAMA_PROXY_API_KEY")
REASONING_MODEL = env("REASONING_MODEL", "gpt-oss:120b")
VISION_MODEL = env("VISION_MODEL", "qwen3-vl:235b-cloud")
VISION_MODEL_FALLBACKS = tuple(
    model.strip()
    for model in env(
        "VISION_MODEL_FALLBACKS",
        "gemma4:31b-cloud,gemma4:cloud",
    ).split(",")
    if model.strip()
)
VISION_MODELS = tuple(dict.fromkeys((VISION_MODEL, *VISION_MODEL_FALLBACKS)))
BLENDER_BIN = env("BLENDER_BIN", "/home/headless/blender/blender")
