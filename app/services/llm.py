"""Groq chat completions over the free OpenAI-compatible HTTP API."""

from __future__ import annotations

import re
from typing import Any, Protocol

import httpx

from app.errors import AppError, LLMError
from app.redact import redact

_THINK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_MAX_KEY_LENGTH = 500


def resolve_api_key(header: str | None, body: str | None, env_key: str) -> str:
    """Pick the first key the caller actually provided.

    The header wins, then the JSON field, then `GROQ_API_KEY`. The chosen key
    is returned to the caller of this function and is not logged.
    """
    for candidate in (header, body, env_key):
        if candidate and candidate.strip():
            key = candidate.strip()
            if len(key) > _MAX_KEY_LENGTH:
                raise AppError("That API key does not look valid.", 400, "invalid_api_key")
            return key
    raise AppError(
        "Provide a Groq API key in the request, or set GROQ_API_KEY in the server environment.",
        400,
        "missing_api_key",
    )


def extract_answer(message: dict[str, Any]) -> str:
    """Return the user-facing answer, without a reasoning trace."""
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
        text = "\n".join(parts)
    else:
        text = ""
    return _THINK.sub("", text).strip()


class LLMClient(Protocol):
    """Anything that can turn a grounded prompt into an answer."""

    def complete(self, *, api_key: str, system: str, user: str) -> str:
        """Return the answer text. Implementations must not log `api_key`."""


class GroqLLM:
    """Small client for `POST /chat/completions` on Groq.

    `client` is injectable so tests can use `httpx.MockTransport` and never
    open a socket.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        timeout: float,
        temperature: float,
        max_completion_tokens: int,
        reasoning_effort: str,
        include_reasoning: bool,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort.strip()
        self.include_reasoning = include_reasoning
        self._client = client

    def __repr__(self) -> str:
        return f"GroqLLM(model={self.model!r})"

    def complete(self, *, api_key: str, system: str, user: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        payload.update(self._reasoning_fields())
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = self._post(headers, payload)
        except httpx.TimeoutException as exc:
            raise LLMError("The Groq request timed out. Try again in a moment.") from exc
        except httpx.HTTPError as exc:
            raise LLMError("Could not reach the Groq API.") from exc

        if response.status_code >= 400:
            raise LLMError(_failure_message(response))

        try:
            body = response.json()
            message = body["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("Groq returned an unexpected response.") from exc
        if not isinstance(message, dict):
            raise LLMError("Groq returned an unexpected response.")

        answer = extract_answer(message)
        if not answer:
            raise LLMError("Groq returned an empty answer. Try again, or choose another free model.")
        return answer

    def _reasoning_fields(self) -> dict[str, Any]:
        # These fields are specific to Groq's gpt-oss models. Other model ids
        # reject them, so they are omitted unless the configured model is one.
        if "gpt-oss" not in self.model.lower():
            return {}
        fields: dict[str, Any] = {"include_reasoning": self.include_reasoning}
        if self.reasoning_effort:
            fields["reasoning_effort"] = self.reasoning_effort
        return fields

    def _post(self, headers: dict[str, str], payload: dict[str, Any]) -> httpx.Response:
        url = f"{self.base_url}/chat/completions"
        if self._client is not None:
            return self._client.post(url, headers=headers, json=payload)
        timeout = httpx.Timeout(self.timeout, connect=10.0)
        with httpx.Client(timeout=timeout) as client:
            return client.post(url, headers=headers, json=payload)


def _failure_message(response: httpx.Response) -> str:
    if response.status_code == 401:
        return "Groq rejected the API key."
    if response.status_code == 429:
        return "Groq rate limit reached on the free tier. Wait a moment and try again."
    detail = ""
    try:
        body = response.json()
        message = body.get("error", {}).get("message")
        if isinstance(message, str):
            detail = redact(message).strip()[:400]
    except ValueError:
        detail = ""
    if detail:
        return f"Groq request failed ({response.status_code}). {detail}"
    return f"Groq request failed ({response.status_code})."
