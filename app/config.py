from __future__ import annotations

import base64
import binascii
import os
from pathlib import Path


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = env(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


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
SOURCE_REVISION = env("THREED_SOURCE_REVISION", "unknown")
API_TOKEN = secret_env("THREED_API_TOKEN")

# This is intentionally fixed to our existing MediaPitch Ollama proxy.
OLLAMA_PROXY_BASE_URL = "https://mediapitch.in/ollama-proxy"
OLLAMA_PROXY_API_KEY = secret_env("OLLAMA_PROXY_API_KEY")
OLLAMA_PROXY_TIMEOUT_SECONDS = float_env(
    "OLLAMA_PROXY_TIMEOUT_SECONDS",
    180.0,
    minimum=30.0,
)
AUTO_IMPROVE_ROUND_TIMEOUT_SECONDS = float_env(
    "AUTO_IMPROVE_ROUND_TIMEOUT_SECONDS",
    240.0,
    minimum=60.0,
)
REASONING_MODEL = env("REASONING_MODEL", "gpt-oss:120b")
VISION_MODEL = env("VISION_MODEL", "gemma4:31b-cloud")
VISION_MODEL_FALLBACKS = tuple(
    model.strip()
    for model in env(
        "VISION_MODEL_FALLBACKS",
        "gemma4:cloud",
    ).split(",")
    if model.strip()
)
VISION_MODELS = tuple(dict.fromkeys((VISION_MODEL, *VISION_MODEL_FALLBACKS)))
BLENDER_BIN = env("BLENDER_BIN", "/home/headless/blender/blender")
