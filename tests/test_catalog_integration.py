"""A single object name is a valid request; catalog results remain unverified."""
from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app import cad_jobs, main, model_catalog
from app.cad_experiment import CadRequest


def _job(tmp_path, monkeypatch, prompt="Prius"):
    monkeypatch.setattr(main, "JOBS_ROOT", tmp_path)
    job_id = str(uuid.uuid4())
    root = tmp_path / job_id
    for name in main.ARTIFACT_CATEGORIES:
        (root / name).mkdir(parents=True)
    (root / "request.json").write_text(json.dumps({"prompt": prompt}))
    main._write_status(root, state="created", stage="waiting_for_input")
    return job_id, root


def test_five_character_prompt_is_accepted_by_cad_input_contract():
    assert CadRequest(prompt="Prius").prompt == "Prius"
    with pytest.raises(ValidationError):
        CadRequest(prompt="")


def test_asset_search_uses_actual_job_prompt_and_preserves_unverified_state(tmp_path, monkeypatch):
    ident, _ = _job(tmp_path, monkeypatch)
    seen = []

    async def fake_search(prompt):
        seen.append(prompt)
        return {"status": "candidates_need_visual_review", "candidates": [
            {"title": "Toyota Prius", "quality_status": "unverified_candidate"}
        ]}

    monkeypatch.setattr(model_catalog, "search_licensed_models", fake_search)
    result = asyncio.run(cad_jobs.find_reusable_model_candidates(ident))
    assert seen == ["Prius"]
    assert result["job_id"] == ident
    assert result["candidates"][0]["quality_status"] == "unverified_candidate"


def test_asset_provider_failure_is_not_reported_as_empty_catalog(tmp_path, monkeypatch):
    ident, _ = _job(tmp_path, monkeypatch)

    async def broken(_):
        raise ValueError("bad response")

    monkeypatch.setattr(model_catalog, "search_licensed_models", broken)
    with pytest.raises(HTTPException) as error:
        asyncio.run(cad_jobs.find_reusable_model_candidates(ident))
    assert error.value.status_code == 502


def test_short_single_word_job_starts_cpu_generation(tmp_path, monkeypatch):
    ident, _ = _job(tmp_path, monkeypatch)
    invocations = []

    async def fake_run(*args, **kwargs):
        invocations.append((args, kwargs))

    monkeypatch.setattr(cad_jobs, "run_cad_job", fake_run)

    async def run():
        result = await cad_jobs.start_cad_generation(ident, cad_jobs.CadGenerateRequest())
        await asyncio.sleep(0)
        return result

    result = asyncio.run(run())
    assert result["started"] is True
    assert result["engine"] == "cpu_cad"
    assert invocations and invocations[0][0][0] == ident


def test_whitespace_only_prompt_stays_invalid(tmp_path, monkeypatch):
    ident, _ = _job(tmp_path, monkeypatch, prompt="   ")
    with pytest.raises(HTTPException) as error:
        asyncio.run(cad_jobs.start_cad_generation(ident, cad_jobs.CadGenerateRequest()))
    assert error.value.status_code == 422
