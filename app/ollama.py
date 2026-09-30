from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from .config import OLLAMA_PROXY_API_KEY, OLLAMA_PROXY_BASE_URL


class OllamaProxyError(RuntimeError):
    pass


JSONValue = dict[str, Any] | list[Any]


@dataclass(frozen=True)
class OllamaJSONResult:
    data: JSONValue
    endpoint: str
    usage: dict[str, Any]


def _unwrap_schema_instance(data: JSONValue, schema: dict[str, Any] | None) -> JSONValue:
    """Recover a gateway's properties envelope without inventing judgment values."""
    if not schema or not isinstance(data, dict) or set(data) != {"properties"}:
        return data
    instance = data["properties"]
    expected = set(schema.get("properties", {}))
    if not isinstance(instance, dict) or not expected.intersection(instance):
        return data
    # A copied JSON schema is not an evaluation. Keep it on the error/fallback path.
    values = [instance[key] for key in expected.intersection(instance)]
    if all(isinstance(value, dict) and set(value).intersection({"type", "$ref", "anyOf", "enum"}) for value in values):
        raise OllamaProxyError("Ollama returned schema definitions instead of judgment values.")
    return instance


def decode_structured_json(content: str | JSONValue, *, schema: dict[str, Any] | None = None) -> JSONValue:
    if isinstance(content, (dict, list)):
        return _unwrap_schema_instance(content, schema)

    text = content.strip()
    if not text:
        raise OllamaProxyError("Ollama returned an empty response.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, (dict, list)):
        return _unwrap_schema_instance(parsed, schema)

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, (dict, list)):
            return _unwrap_schema_instance(parsed, schema)

    candidates_with_offsets: list[tuple[int, str]] = []
    first_object = text.find("{")
    last_object = text.rfind("}")
    if first_object >= 0 and last_object > first_object:
        candidates_with_offsets.append(
            (first_object, text[first_object : last_object + 1])
        )
    first_array = text.find("[")
    last_array = text.rfind("]")
    if first_array >= 0 and last_array > first_array:
        candidates_with_offsets.append(
            (first_array, text[first_array : last_array + 1])
        )
    candidates = [
        candidate
        for _, candidate in sorted(candidates_with_offsets, key=lambda item: item[0])
    ]

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, (dict, list)):
            return _unwrap_schema_instance(parsed, schema)

    preview = " ".join(text.split())[:240]
    if last_error is not None:
        raise OllamaProxyError(
            f"Ollama returned text, but not valid structured JSON. Preview: {preview!r}"
        ) from last_error
    raise OllamaProxyError(
        f"Ollama returned text, but not a JSON object or array. Preview: {preview!r}"
    )


class OllamaProxyClient:
    def __init__(
        self,
        *,
        api_key: str = OLLAMA_PROXY_API_KEY,
        base_url: str = OLLAMA_PROXY_BASE_URL,
        timeout_seconds: float = 600.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise OllamaProxyError("OLLAMA_PROXY_API_KEY is not configured.")
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        timeout = httpx.Timeout(self.timeout_seconds, connect=15.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            return await client.post(f"{self.base_url}{path}", json=payload, headers=self._headers())

    @staticmethod
    def _response_error_message(response: httpx.Response, path: str) -> str:
        detail = response.text.strip().replace("\n", " ")
        if len(detail) > 1000:
            detail = detail[:1000] + "..."
        return (
            f"Ollama proxy {path} returned HTTP {response.status_code}: "
            f"{detail or response.reason_phrase}"
        )

    @classmethod
    def _raise_for_response(cls, response: httpx.Response, path: str) -> None:
        if response.is_success:
            return
        raise OllamaProxyError(cls._response_error_message(response, path))

    async def chat_json(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        images: list[str] | None = None,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.1,
        num_predict: int | None = None,
    ) -> OllamaJSONResult:
        if schema:
            # Hosted gateways may treat `format` as JSON mode and ignore the
            # schema itself. The model must see the contract in every endpoint.
            system += (
                "\nOutput contract: return one JSON value matching this schema. "
                "Include every judgment field explicitly, even fields with defaults. "
                "Put the instance fields directly at the root, without a properties envelope. "
                "Do not return the schema itself.\n"
                + json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
            )
        user_message: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            user_message["images"] = images

        options: dict[str, Any] = {"temperature": temperature}
        if num_predict is not None:
            options["num_predict"] = int(num_predict)

        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                user_message,
            ],
            "format": schema or "json",
            "options": options,
        }

        response = await self._post("/api/chat", payload)
        chat_decode_error: OllamaProxyError | None = None
        chat_transport_error: str | None = None
        recoverable_route_statuses = {403, 404, 405, 429, 500, 502, 503, 504}
        if response.is_success:
            body = response.json()
            content = (body.get("message") or {}).get("content") or ""
            try:
                data = decode_structured_json(content, schema=schema)
            except OllamaProxyError as exc:
                chat_decode_error = exc
            else:
                return OllamaJSONResult(
                    data=data,
                    endpoint="/api/chat",
                    usage={
                        "prompt_eval_count": int(body.get("prompt_eval_count") or 0),
                        "eval_count": int(body.get("eval_count") or 0),
                    },
                )
        elif response.status_code in recoverable_route_statuses:
            chat_transport_error = self._response_error_message(response, "/api/chat")
        else:
            self._raise_for_response(response, "/api/chat")

        generate_payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "prompt": f"[SYSTEM]\n{system}\n\n[USER]\n{prompt}\n\nReturn valid JSON only.",
            "format": schema or "json",
            "options": options,
        }
        if images:
            generate_payload["images"] = images

        response = await self._post("/api/generate", generate_payload)
        if not response.is_success:
            generate_transport_error = self._response_error_message(response, "/api/generate")
            if chat_transport_error:
                raise OllamaProxyError(
                    f"{chat_transport_error}; fallback route also failed: {generate_transport_error}"
                )
            self._raise_for_response(response, "/api/generate")
        body = response.json()
        try:
            data = decode_structured_json(body.get("response") or "", schema=schema)
        except OllamaProxyError as generate_error:
            if schema:
                # Some hosted multimodal models accept JSON mode but do not
                # reliably obey a full JSON-schema response format. Retry the
                # same visual task in plain JSON mode before rejecting the
                # model. Validation/normalization still happens at the caller.
                relaxed_payload = {
                    **payload,
                    "format": "json",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                system
                                + "\nReturn one valid JSON object only. "
                                "Do not use Markdown fences or commentary."
                            ),
                        },
                        user_message,
                    ],
                }
                relaxed_response = await self._post("/api/chat", relaxed_payload)
                if relaxed_response.status_code not in {404, 405}:
                    self._raise_for_response(relaxed_response, "/api/chat")
                    relaxed_body = relaxed_response.json()
                    relaxed_content = (relaxed_body.get("message") or {}).get("content") or ""
                    try:
                        relaxed_data = decode_structured_json(relaxed_content, schema=schema)
                    except OllamaProxyError:
                        pass
                    else:
                        return OllamaJSONResult(
                            data=relaxed_data,
                            endpoint="/api/chat?format=json",
                            usage={
                                "prompt_eval_count": int(relaxed_body.get("prompt_eval_count") or 0),
                                "eval_count": int(relaxed_body.get("eval_count") or 0),
                            },
                        )

                relaxed_generate_payload = {
                    **generate_payload,
                    "format": "json",
                    "prompt": (
                        f"[SYSTEM]\n{system}\n\n[USER]\n{prompt}\n\n"
                        "Return one valid JSON object only. Do not use Markdown fences or commentary."
                    ),
                }
                relaxed_response = await self._post("/api/generate", relaxed_generate_payload)
                self._raise_for_response(relaxed_response, "/api/generate")
                relaxed_body = relaxed_response.json()
                try:
                    relaxed_data = decode_structured_json(relaxed_body.get("response") or "", schema=schema)
                except OllamaProxyError as relaxed_error:
                    if chat_decode_error:
                        raise OllamaProxyError(
                            "Ollama returned malformed structured JSON from schema-mode /api/chat "
                            f"({chat_decode_error}), schema-mode /api/generate ({generate_error}), "
                            f"and relaxed JSON mode ({relaxed_error})."
                        ) from relaxed_error
                    raise
                return OllamaJSONResult(
                    data=relaxed_data,
                    endpoint="/api/generate?format=json",
                    usage={
                        "prompt_eval_count": int(relaxed_body.get("prompt_eval_count") or 0),
                        "eval_count": int(relaxed_body.get("eval_count") or 0),
                    },
                )

            if chat_decode_error:
                raise OllamaProxyError(
                    "Ollama returned malformed structured JSON from both /api/chat "
                    f"({chat_decode_error}) and /api/generate ({generate_error})."
                ) from generate_error
            raise
        return OllamaJSONResult(
            data=data,
            endpoint="/api/generate",
            usage={
                "prompt_eval_count": int(body.get("prompt_eval_count") or 0),
                "eval_count": int(body.get("eval_count") or 0),
            },
        )
