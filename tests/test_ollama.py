import asyncio

import httpx
import pytest

from app.ollama import OllamaProxyClient, OllamaProxyError, decode_structured_json


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



def test_chat_json_falls_back_to_generate_on_forbidden_chat(monkeypatch):
    client = OllamaProxyClient(api_key="test", base_url="https://proxy.test")
    calls = []

    async def post(path, payload):
        calls.append(path)
        request = httpx.Request("POST", "https://proxy.test" + path)
        if path == "/api/chat":
            return httpx.Response(403, request=request, text="Forbidden")
        return httpx.Response(
            200,
            request=request,
            json={"response": '{"ok": true}', "prompt_eval_count": 1, "eval_count": 2},
        )

    monkeypatch.setattr(client, "_post", post)
    result = asyncio.run(client.chat_json(
        model="test-model",
        system="Return JSON.",
        prompt="Test.",
        schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    ))

    assert calls == ["/api/chat", "/api/generate"]
    assert result.endpoint == "/api/generate"
    assert result.data == {"ok": True}


def test_chat_json_reports_both_route_errors(monkeypatch):
    client = OllamaProxyClient(api_key="test", base_url="https://proxy.test")

    async def post(path, payload):
        request = httpx.Request("POST", "https://proxy.test" + path)
        status = 403 if path == "/api/chat" else 429
        text = "Forbidden" if path == "/api/chat" else "Rate limited"
        return httpx.Response(status, request=request, text=text)

    monkeypatch.setattr(client, "_post", post)

    with pytest.raises(OllamaProxyError) as exc:
        asyncio.run(client.chat_json(
            model="test-model",
            system="Return JSON.",
            prompt="Test.",
        ))

    message = str(exc.value)
    assert "/api/chat returned HTTP 403" in message
    assert "/api/generate returned HTTP 429" in message
