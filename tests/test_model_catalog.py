"""Realistic licensed-catalog behavior, with the provider boundary mocked."""
import asyncio

import httpx

from app.model_catalog import rank_catalog_result, search_licensed_models, search_subject


def candidate(name="2012 Toyota Prius", *, uid="a" * 32, license_id="by", faces=19000, **extras):
    return {
        "uid": uid, "name": name, "isDownloadable": True,
        "license": {"uid": license_id}, "faceCount": faces,
        "user": {"username": "Author"}, **extras,
    }


def test_prompt_language_cleanup_retains_identity():
    assert search_subject("Make a 3D model of Toyota Prius 2012") == "Toyota Prius 2012"
    assert search_subject("Prius") == "Prius"
    assert search_subject("Please create a Toyota Prius") == "Toyota Prius"


def test_unsafe_licenses_noai_and_wrong_identity_are_excluded():
    assert rank_catalog_result("Toyota Prius", candidate()) is not None
    assert rank_catalog_result("Toyota Prius", candidate(license_id="by-nc")) is None
    assert rank_catalog_result("Toyota Prius", candidate(license_id="unknown")) is None
    assert rank_catalog_result("Toyota Prius", candidate(isNoAi=True)) is None
    assert rank_catalog_result("Toyota Prius", candidate(tags=[{"name": "NoAI"}])) is None
    assert rank_catalog_result("Toyota Prius", candidate(name="Honda Civic")) is None
    assert rank_catalog_result("Toyota Prius", candidate(faces=0)) is None
    assert rank_catalog_result("Toyota Prius", candidate(isDownloadable=False)) is None
    assert rank_catalog_result("Toyota Prius 2026", candidate()) is None


def test_metadata_search_keeps_provenance_and_never_claims_imported_mesh():
    def respond(request):
        assert request.url.path == "/v3/models"
        assert request.url.params["q"] == "Toyota Prius"
        assert request.url.params["downloadable"] == "true"
        return httpx.Response(200, json={"results": [
            candidate("Toyota Prius", uid="b" * 32, faces=50000),
            candidate("Toyota Prius", uid="c" * 32, license_id="by-nc"),
            candidate("Toyota Prius 2012", uid="d" * 32),
            candidate("Honda Civic", uid="e" * 32),
        ]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await search_licensed_models("Make a 3D model of Toyota Prius", client=client)

    report = asyncio.run(run())
    assert report["status"] == "candidates_need_visual_review"
    assert len(report["candidates"]) == 2
    assert report["candidates"][0]["title"] == "Toyota Prius"
    assert report["candidates"][0]["requires_user_oauth_to_download"]
    assert report["download_ready"] is False
    assert report["candidates"][0]["source_url"].startswith("https://sketchfab.com/models/")


def test_empty_results_cannot_pretend_model_success():
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"results": []})
        )) as client:
            return await search_licensed_models("Prius", client=client)

    output = asyncio.run(run())
    assert output["status"] == "no_licensed_match"
    assert output["candidates"] == []


def test_catalog_outage_does_not_pretend_no_matches():
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(503)
        )) as client:
            return await search_licensed_models("Prius", client=client)

    try:
        asyncio.run(run())
    except httpx.HTTPStatusError:
        pass
    else:
        raise AssertionError("HTTP errors must be observable")
