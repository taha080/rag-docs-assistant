"""Groq client behavior with a mocked HTTP transport."""

import json

import httpx
import pytest

from app.errors import LLMError
from app.services.llm import GroqLLM, extract_answer, resolve_api_key


def _llm(handler) -> GroqLLM:
    return GroqLLM(
        model="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1",
        timeout=5,
        temperature=0.5,
        max_completion_tokens=256,
        reasoning_effort="low",
        include_reasoning=False,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_complete_reads_the_answer_and_hides_reasoning() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer gsk_testkey"
        body = json.loads(request.content.decode())
        assert body["model"] == "openai/gpt-oss-20b"
        assert body["include_reasoning"] is False
        assert body["reasoning_effort"] == "low"
        assert "gsk_" not in json.dumps(body)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "<think>hidden chain</think>\n\nHarvest is in October.",
                            "reasoning": "should not be returned",
                        }
                    }
                ]
            },
        )

    answer = _llm(handler).complete(api_key="gsk_testkey", system="Be brief.", user="When?")
    assert answer == "Harvest is in October."


def test_empty_content_is_not_replaced_with_reasoning() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": None, "reasoning": "gsk_secret thoughts"}}]},
        )

    with pytest.raises(LLMError) as caught:
        _llm(handler).complete(api_key="gsk_testkey", system="s", user="u")
    assert "gsk_secret" not in caught.value.message
    assert "thoughts" not in caught.value.message


def test_auth_and_rate_limit_messages_drop_the_response_body() -> None:
    def unauthorized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid gsk_supersecret"}})

    with pytest.raises(LLMError) as caught:
        _llm(unauthorized).complete(api_key="gsk_supersecret", system="s", user="u")
    assert caught.value.message == "Groq rejected the API key."
    assert "gsk_supersecret" not in caught.value.message

    def limited(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down gsk_supersecret"}})

    with pytest.raises(LLMError) as caught:
        _llm(limited).complete(api_key="gsk_supersecret", system="s", user="u")
    assert "rate limit" in caught.value.message.lower()
    assert "gsk_supersecret" not in caught.value.message


def test_model_error_is_redacted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "The model does not exist. key gsk_supersecret"}},
        )

    with pytest.raises(LLMError) as caught:
        _llm(handler).complete(api_key="gsk_supersecret", system="s", user="u")
    assert "does not exist" in caught.value.message
    assert "gsk_supersecret" not in caught.value.message
    assert "gsk_[redacted]" in caught.value.message


def test_timeout_is_a_friendly_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    with pytest.raises(LLMError) as caught:
        _llm(handler).complete(api_key="gsk_testkey", system="s", user="u")
    assert "timed out" in caught.value.message.lower()


def test_other_models_do_not_send_gpt_oss_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert "include_reasoning" not in body
        assert "reasoning_effort" not in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    llm = GroqLLM(
        model="qwen/qwen3.8-27b",
        base_url="https://api.groq.com/openai/v1",
        timeout=5,
        temperature=0.5,
        max_completion_tokens=128,
        reasoning_effort="low",
        include_reasoning=False,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert llm.complete(api_key="gsk_testkey", system="s", user="u") == "ok"


def test_resolve_api_key_priority_and_missing() -> None:
    assert resolve_api_key(" header ", "body", "env") == "header"
    assert resolve_api_key("  ", " body ", "env") == "body"
    assert resolve_api_key(None, None, " env ") == "env"
    with pytest.raises(Exception) as caught:
        resolve_api_key(None, "  ", "")
    assert caught.value.code == "missing_api_key"


def test_extract_answer_joins_text_parts() -> None:
    text = extract_answer(
        {"content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "there"}]}
    )
    assert text == "Hello\nthere"
