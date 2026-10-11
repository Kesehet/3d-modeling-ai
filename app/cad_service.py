"""Standalone experimental CPU CAD API; production app remains unchanged."""
from fastapi import FastAPI

from .cad_experiment import router

app = FastAPI(title="3D AI CPU CAD Experiment", version="0.1.0")
app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "engine": "build123d", "requires_gpu": False,
            "mode": "experimental", "quality_certified": False}
