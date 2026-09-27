import asyncio
import json

import pytest
from fastapi import HTTPException

import app.main as main


def _make_job(tmp_path, monkeypatch, job_id="abc123"):
    jobs_root = tmp_path / "jobs"
    root = jobs_root / job_id
    root.mkdir(parents=True)
    (root / "request.json").write_text(
        json.dumps({"prompt": "Toyota Prius", "intended_use": "rendering"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "JOBS_ROOT", jobs_root)
    return root


def test_reference_gate_blocks_modeling_when_research_still_has_zero_refs(tmp_path, monkeypatch):
    root = _make_job(tmp_path, monkeypatch)
    calls = []

    async def fake_research(job_id, request):
        calls.append(request.query)
        return {"rejected": [{"stored_name": "bad.jpg"}], "research_errors": []}

    monkeypatch.setattr(main, "research_job", fake_research)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(main._ensure_reference_pack("abc123", max_images=3, attempts=2))

    assert exc.value.status_code == 424
    assert len(calls) == 2
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["stage"] == "waiting_for_references"
    assert status["reference_gate"]["state"] == "blocked"
    assert status["reference_count"] == 0


def test_reference_gate_passes_only_after_usable_reference_is_saved(tmp_path, monkeypatch):
    root = _make_job(tmp_path, monkeypatch)
    calls = []

    async def fake_research(job_id, request):
        calls.append(request.query)
        (root / "references.json").write_text(
            json.dumps(
                [
                    {
                        "stored_name": "ref-user.jpg",
                        "original_name": "prius.jpg",
                        "uploaded_at": "2026-09-27T00:00:00+00:00",
                    }
                ]
            ),
            encoding="utf-8",
        )
        return {"rejected": [], "research_errors": []}

    monkeypatch.setattr(main, "research_job", fake_research)

    refs = asyncio.run(main._ensure_reference_pack("abc123", max_images=3, attempts=2))

    assert len(calls) == 1
    assert len(refs) == 1
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["stage"] == "references_ready"
    assert status["reference_gate"]["state"] == "ready"
    assert status["reference_count"] == 1
