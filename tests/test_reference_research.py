import json

from PIL import Image

from app.dashboard import _reference_rows
from app.main import (
    ReferencePackDecision,
    ReferenceSearchPlan,
    _is_usable_reference_record,
    _metadata_supports_reference_identity,
    _normalize_reference_pack_payload,
    _prune_unverified_auto_references,
)
from app.research import _candidate_relevance_score


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
