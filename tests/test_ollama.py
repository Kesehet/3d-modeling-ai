import pytest

from app.ollama import OllamaProxyError, decode_structured_json


def test_decode_structured_json_plain() -> None:
    assert decode_structured_json('{"ok": true}') == {"ok": True}


def test_decode_structured_json_fenced() -> None:
    assert decode_structured_json('```json\n{"ok": true}\n```') == {"ok": True}


def test_decode_structured_json_top_level_array() -> None:
    assert decode_structured_json('[{"stored_name":"a.jpg","accept":true}]') == [
        {"stored_name": "a.jpg", "accept": True}
    ]


def test_decode_structured_json_fenced_array() -> None:
    assert decode_structured_json('```json\n[{"ok": true}]\n```') == [{"ok": True}]


def test_decode_structured_json_embedded() -> None:
    assert decode_structured_json('answer: {"ok": true} done') == {"ok": True}


def test_decode_structured_json_embedded_array() -> None:
    assert decode_structured_json('results: [{"ok": true}] done') == [{"ok": True}]


def test_decode_structured_json_rejects_non_json() -> None:
    with pytest.raises(OllamaProxyError):
        decode_structured_json("not json")


def test_live_comparison_properties_envelope_preserves_negative_judgment():
    from app.main import RefinementComparison

    raw = {"properties": {
        "candidate_is_better": False,
        "summary": "Both models remain generic blobs; the candidate is not an improvement.",
        "improvements": [], "regressions": [],
    }}
    data = decode_structured_json(raw, schema=RefinementComparison.model_json_schema())
    comparison = RefinementComparison.model_validate(data)
    assert comparison.candidate_is_better is False
    assert comparison.summary == raw["properties"]["summary"]


def test_properties_recovery_never_treats_a_copied_schema_as_a_verdict():
    schema = {"properties": {"candidate_is_better": {"type": "boolean"}}}
    with pytest.raises(OllamaProxyError, match="schema definitions"):
        decode_structured_json(schema, schema=schema)


def test_properties_field_without_output_schema_is_preserved():
    raw = {"properties": {"name": "a legitimate object"}}
    assert decode_structured_json(raw) == raw
