import pytest

from app.ollama import OllamaProxyError, decode_structured_json


def test_decode_structured_json_plain() -> None:
    assert decode_structured_json('{"ok": true}') == {"ok": True}


def test_decode_structured_json_fenced() -> None:
    assert decode_structured_json('```json\n{"ok": true}\n```') == {"ok": True}


def test_decode_structured_json_embedded() -> None:
    assert decode_structured_json('answer: {"ok": true} done') == {"ok": True}


def test_decode_structured_json_rejects_non_json() -> None:
    with pytest.raises(OllamaProxyError):
        decode_structured_json("not json")
