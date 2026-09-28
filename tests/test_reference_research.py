import asyncio
import json
from io import BytesIO

import httpx
from PIL import Image

from app.dashboard import _reference_rows
from app.main import (
    ReferencePackDecision,
    ReferenceSearchPlan,
    _is_usable_reference_record,
    _metadata_supports_reference_identity,
    _metadata_title_supports_reference_identity,
    _normalize_reference_coherence_payload,
    _normalize_reference_pack_payload,
    _prune_unverified_auto_references,
)
from app.research import (
    _candidate_has_identity_metadata_signal,
    _candidate_relevance_score,
    _commons_media_search_query,
    _query_terms,
    research_web_references,
)


def test_reference_search_skips_document_thumbnails_before_download(tmp_path, monkeypatch):
    photo = BytesIO()
    Image.new("RGB", (400, 300), "brown").save(photo, format="JPEG")
    downloads = []
    commons_queries = []

    def respond(request):
        if request.url.host == "en.wikipedia.org":
            return httpx.Response(200, json={"query": {"pages": []}})
        if request.url.host == "commons.wikimedia.org":
            commons_queries.append(str(request.url.params.get("gsrsearch") or ""))
            assert request.url.params.get("mediasearch") == "true"
            return httpx.Response(200, json={"query": {"pages": [
                {"title": "File:Subject book.pdf", "imageinfo": [{
                    "mime": "application/pdf", "thumburl": "https://images.test/book.jpg"}]},
                {"title": "File:Subject photo.jpg", "imageinfo": [{
                    "mime": "image/jpeg", "thumburl": "https://images.test/photo.jpg"}]},
            ]}})
        downloads.append(str(request.url))
        return httpx.Response(200, content=photo.getvalue(), headers={"content-type": "image/jpeg"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr("app.research.httpx.AsyncClient", lambda **_: client)
    result = asyncio.run(research_web_references("Subject", tmp_path))
    assert commons_queries == ["Subject filetype:bitmap"]
    assert downloads == ["https://images.test/photo.jpg"]
    assert result["candidate_count"] == 1
    assert result["references"][0]["title"] == "File:Subject photo.jpg"


def test_view_words_do_not_pollute_reference_identity_terms():
    assert _query_terms("Toyota Prius front three quarter view reference photo") == [
        "toyota",
        "prius",
    ]


def test_commons_media_search_is_bitmap_only():
    assert _commons_media_search_query("dining table") == "dining table filetype:bitmap"
    assert _commons_media_search_query("dining table", exact=True) == (
        '"dining table" filetype:bitmap'
    )


def test_metadata_gate_rejects_incidental_full_text_spillover():
    unrelated = {
        "provider": "wikimedia_commons",
        "title": "File:Portrait of Ada Example.jpg",
        "source_title": "Portrait of Ada Example",
        "description": "The subject is standing beside a rectangular wooden dining table.",
    }
    useful = {
        "provider": "wikimedia_commons",
        "title": "File:Wooden dining table in a room.jpg",
        "source_title": "Wooden dining table",
        "description": "Furniture photograph.",
    }
    generic_filename_but_exact_description = {
        "provider": "wikimedia_commons",
        "title": "File:DSC 1234.jpg",
        "source_title": "DSC 1234",
        "description": "A dining table photographed from the side.",
    }

    assert _candidate_has_identity_metadata_signal("dining table side view", unrelated) is False
    assert _candidate_has_identity_metadata_signal("dining table side view", useful) is True
    assert _candidate_has_identity_metadata_signal(
        "dining table side view",
        generic_filename_but_exact_description,
    ) is True


def test_exact_subject_metadata_outranks_unrelated_wikimedia_result():
    exact = {
        "provider": "wikimedia_commons",
        "title": "File:Toyota Prius 2023 front.jpg",
        "description": "Toyota Prius passenger car",
        "search_rank": 2,
    }
    unrelated = {
        "provider": "wikimedia_commons",
        "title": "File:Toyota Corolla sedan.jpg",
        "description": "Toyota Corolla",
        "search_rank": 1,
    }

    assert _candidate_relevance_score("Toyota Prius", exact) > _candidate_relevance_score(
        "Toyota Prius",
        unrelated,
    )


def test_automatic_reference_is_not_usable_until_visually_verified():
    record = {
        "stored_name": "candidate-01-example.jpg",
        "provider": "wikimedia_commons",
        "match_score": 0.95,
    }
    assert _is_usable_reference_record(record) is False

    record.update(
        {
            "verified": True,
            "exact_identity_match": True,
            "useful_for_geometry": True,
        }
    )
    assert _is_usable_reference_record(record) is True

    record["match_score"] = 0.69
    assert _is_usable_reference_record(record) is False


def test_user_uploaded_reference_remains_authoritative():
    record = {
        "stored_name": "ref-user.jpg",
        "uploaded_at": "2026-09-27T00:00:00+00:00",
    }
    assert _is_usable_reference_record(record) is True


def test_prune_removes_bad_auto_reference_file_but_keeps_upload(tmp_path):
    refs = tmp_path / "references"
    refs.mkdir()
    bad = refs / "web-01-bad.jpg"
    upload = refs / "ref-good.jpg"
    Image.new("RGB", (400, 300)).save(bad)
    Image.new("RGB", (400, 300)).save(upload)

    index = [
        {
            "stored_name": bad.name,
            "provider": "wikimedia_commons",
            "verified": False,
        },
        {
            "stored_name": upload.name,
            "uploaded_at": "2026-09-27T00:00:00+00:00",
        },
    ]

    kept = _prune_unverified_auto_references(tmp_path, index)

    assert bad.exists() is False
    assert upload.exists() is True
    assert [item["stored_name"] for item in kept] == [upload.name]


def test_dashboard_hides_unverified_search_candidates(tmp_path):
    refs = tmp_path / "references"
    refs.mkdir()
    hidden = refs / "candidate-01-bad.jpg"
    visible = refs / "candidate-02-good.jpg"
    Image.new("RGB", (400, 300)).save(hidden)
    Image.new("RGB", (400, 300)).save(visible)

    (tmp_path / "references.json").write_text(
        json.dumps(
            [
                {
                    "stored_name": hidden.name,
                    "provider": "wikimedia_commons",
                    "verified": False,
                },
                {
                    "stored_name": visible.name,
                    "provider": "wikimedia_commons",
                    "verified": True,
                    "exact_identity_match": True,
                    "useful_for_geometry": True,
                    "match_score": 0.92,
                },
            ]
        ),
        encoding="utf-8",
    )

    rows = _reference_rows(tmp_path)

    assert [row["name"] for row in rows] == [visible.name]
    assert rows[0]["match_score"] == 0.92



def test_reference_verifier_normalizes_common_model_output_variants():
    records = [
        {"stored_name": "candidate-a.jpg"},
        {"stored_name": "candidate-b.jpg"},
    ]
    payload = {
        "results": [
            {
                "filename": "candidate-a.jpg",
                "accepted": "yes",
                "score": 92,
                "identity_match": "true",
                "geometry_useful": True,
                "explanation": "Clear view of the requested car.",
            },
            {
                # Deliberately omit filename to exercise positional recovery.
                "relevant": False,
                "confidence": 0.2,
                "reason": "Different model.",
            },
        ]
    }

    normalized = _normalize_reference_pack_payload(payload, records)
    pack = ReferencePackDecision.model_validate(normalized)

    assert len(pack.decisions) == 2
    assert pack.decisions[0].stored_name == "candidate-a.jpg"
    assert pack.decisions[0].accept is True
    assert pack.decisions[0].match_score == 0.92
    assert pack.decisions[0].exact_identity_match is True
    assert pack.decisions[0].useful_for_geometry is True
    assert pack.decisions[1].stored_name == "candidate-b.jpg"
    assert pack.decisions[1].accept is False


def test_metadata_identity_supports_named_model_without_invented_generation():
    plan = ReferenceSearchPlan(
        primary_query="Volkswagen Polo",
        subject_description="A Volkswagen Polo car",
        identity_constraints=[],
    )
    matching = {
        "title": "File:Volkswagen Polo front three-quarter.jpg",
        "description": "Volkswagen Polo hatchback",
    }
    sibling = {
        "title": "File:Volkswagen Golf front.jpg",
        "description": "Volkswagen Golf hatchback",
    }

    assert _metadata_supports_reference_identity(plan, matching) is True
    assert _metadata_supports_reference_identity(plan, sibling) is False



def test_reference_verifier_recovers_missing_accept_flag():
    records = [{"stored_name": "candidate-a.jpg"}]
    payload = {
        "decisions": [
            {
                "stored_name": "candidate-a.jpg",
                "match_score": 0.91,
                "exact_identity_match": True,
                "useful_for_geometry": True,
                "reason": "Exact requested subject and useful three-quarter view.",
            }
        ]
    }

    normalized = _normalize_reference_pack_payload(payload, records)
    pack = ReferencePackDecision.model_validate(normalized)

    assert pack.decisions[0].accept is True



def test_whole_subject_title_outranks_incidental_part_mention():
    whole = {
        "provider": "wikimedia_commons",
        "title": "File:Toyota Prius XW30 front three-quarter.jpg",
        "description": "Toyota Prius passenger car",
        "search_rank": 8,
    }
    part = {
        "provider": "wikimedia_commons",
        "title": "File:Prius 12v Battery Location.jpg",
        "description": "Shows the 12 volt battery in the back of a Toyota Prius.",
        "search_rank": 1,
    }

    assert _candidate_relevance_score("Toyota Prius", whole) > _candidate_relevance_score(
        "Toyota Prius",
        part,
    )


def test_dashboard_and_wheel_are_penalized_unless_requested():
    dashboard = {
        "provider": "wikimedia_commons",
        "title": "File:Toyota Prius Dashboard.jpg",
        "description": "Toyota Prius dashboard",
        "search_rank": 1,
    }
    exterior = {
        "provider": "wikimedia_commons",
        "title": "File:Toyota Prius side view.jpg",
        "description": "Toyota Prius",
        "search_rank": 4,
    }
    wheel = {
        "provider": "wikimedia_commons",
        "title": "File:Toyota Prius Wheel.jpg",
        "description": "Wheel on a Toyota Prius",
        "search_rank": 1,
    }

    assert _candidate_relevance_score("Toyota Prius", exterior) > _candidate_relevance_score(
        "Toyota Prius",
        dashboard,
    )
    assert _candidate_relevance_score("Toyota Prius", exterior) > _candidate_relevance_score(
        "Toyota Prius",
        wheel,
    )
    assert _candidate_relevance_score("Toyota Prius wheel", wheel) > _candidate_relevance_score(
        "Toyota Prius wheel",
        exterior,
    )



def test_reference_verifier_infers_score_from_strict_boolean_decision():
    records = [{"stored_name": "candidate-a.jpg"}]
    payload = {
        "decisions": [
            {
                "stored_name": "candidate-a.jpg",
                "accept": True,
                "exact_identity_match": True,
                "useful_for_geometry": True,
            }
        ]
    }

    normalized = _normalize_reference_pack_payload(payload, records)
    pack = ReferencePackDecision.model_validate(normalized)
    decision = pack.decisions[0]

    assert decision.accept is True
    assert decision.match_score == 0.85
    assert decision.score_inferred is True


def test_reference_verifier_infers_accept_when_strict_booleans_are_present():
    records = [{"stored_name": "candidate-a.jpg"}]
    payload = {
        "decisions": [
            {
                "stored_name": "candidate-a.jpg",
                "exact_identity_match": True,
                "useful_for_geometry": True,
            }
        ]
    }

    normalized = _normalize_reference_pack_payload(payload, records)
    pack = ReferencePackDecision.model_validate(normalized)
    decision = pack.decisions[0]

    assert decision.accept is True
    assert decision.match_score == 0.85
    assert decision.score_inferred is True


def test_title_identity_support_rejects_incidental_caption_match():
    plan = ReferenceSearchPlan(
        primary_query="Toyota Prius",
        subject_description="Toyota Prius",
        identity_constraints=[],
    )
    full_vehicle = {
        "title": "File:Toyota Prius XW30 China.jpg",
        "source_title": "Toyota Prius XW30 China",
        "description": "Toyota Prius",
    }
    battery = {
        "title": "File:Prius 12v Battery Location.jpg",
        "source_title": "Prius 12v Battery Location",
        "description": "Battery in the back of a Toyota Prius",
    }

    assert _metadata_title_supports_reference_identity(plan, full_vehicle) is True
    assert _metadata_title_supports_reference_identity(plan, battery) is False



def test_reference_coherence_normalizer_keeps_anchor():
    records = [
        {"stored_name": "current-prius.jpg"},
        {"stored_name": "old-prius.jpg"},
    ]
    payload = {
        "anchor": "current-prius.jpg",
        "keep": ["current-prius.jpg"],
        "canonical_identity": "Toyota Prius fifth generation",
        "reason": "The second image is a visibly older body generation.",
    }

    normalized = _normalize_reference_coherence_payload(payload, records)

    assert normalized["anchor_stored_name"] == "current-prius.jpg"
    assert normalized["keep_stored_names"] == ["current-prius.jpg"]
    assert normalized["search_hint"] == "Toyota Prius fifth generation"


def test_reference_coherence_normalizer_never_drops_first_anchor():
    records = [
        {"stored_name": "anchor.jpg"},
        {"stored_name": "other.jpg"},
    ]
    payload = {"keep": ["other.jpg"]}

    normalized = _normalize_reference_coherence_payload(payload, records)

    assert normalized["anchor_stored_name"] == "anchor.jpg"
    assert normalized["keep_stored_names"][0] == "anchor.jpg"
    assert "other.jpg" in normalized["keep_stored_names"]
