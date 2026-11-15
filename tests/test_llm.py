"""Tests for the LLM client. The network is replaced by httpx.MockTransport, so no Ollama is needed."""
import json

import httpx
import pytest

from agents.llm import DEFAULT_MODEL, MAX_REPLY_TOKENS, NullClient, OllamaClient, get_llm_client
from agents.schemas import IntentReading

VALID_INTENT = {"intent": "complain", "urgency": "low", "needs_support": False, "reason": "annoyed at the phone"}


def client_with(handler):
    calls = []

    def record(request):
        calls.append(request)
        return handler(request, len(calls))

    return OllamaClient(transport=httpx.MockTransport(record)), calls


def chat_reply(content):
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


def test_null_client_always_returns_none():
    assert NullClient().ask_json("system", "user", IntentReading) is None


def test_successful_structured_reply_is_validated():
    client, calls = client_with(lambda request, n: chat_reply(json.dumps(VALID_INTENT)))
    result = client.ask_json("system", "user", IntentReading)
    assert result == IntentReading(**VALID_INTENT)
    sent = json.loads(calls[0].content)
    assert sent["model"] == DEFAULT_MODEL
    assert sent["stream"] is False
    assert sent["format"] == IntentReading.model_json_schema()


def test_request_turns_thinking_off_and_keeps_output_limit():
    """Regression: with gemma4 thinking on, hidden reasoning used up the token limit and the JSON was cut off."""
    client, calls = client_with(lambda request, n: chat_reply(json.dumps(VALID_INTENT)))
    client.ask_json("system", "user", IntentReading)
    sent = json.loads(calls[0].content)
    assert sent["think"] is False
    assert sent["options"]["num_predict"] == MAX_REPLY_TOKENS == 400
    assert sent["options"]["temperature"] == 0


def test_malformed_reply_is_retried_once_then_accepted():
    replies = ["{not json", json.dumps(VALID_INTENT)]
    client, calls = client_with(lambda request, n: chat_reply(replies[n - 1]))
    assert client.ask_json("system", "user", IntentReading) == IntentReading(**VALID_INTENT)
    assert len(calls) == 2


def test_malformed_reply_twice_returns_none():
    client, calls = client_with(lambda request, n: chat_reply('{"intent": "not-a-real-intent"}'))
    assert client.ask_json("system", "user", IntentReading) is None
    assert len(calls) == 2
    assert "invalid JSON" in client.last_error


def test_unreachable_ollama_returns_none_without_retry():
    def refuse(request, n):
        raise httpx.ConnectError("connection refused", request=request)

    client, calls = client_with(refuse)
    assert client.ask_json("system", "user", IntentReading) is None
    assert len(calls) == 1
    assert "ConnectError" in client.last_error
    assert client.is_available() is False


def test_timeout_returns_none_without_retry():
    def too_slow(request, n):
        raise httpx.ReadTimeout("timed out", request=request)

    client, calls = client_with(too_slow)
    assert client.ask_json("system", "user", IntentReading) is None
    assert len(calls) == 1


def test_server_error_returns_none():
    client, _ = client_with(lambda request, n: httpx.Response(500, json={"error": "out of memory"}))
    assert client.ask_json("system", "user", IntentReading) is None


def test_is_available_checks_the_model_is_installed():
    installed, _ = client_with(lambda request, n: httpx.Response(200, json={"models": [{"name": DEFAULT_MODEL}]}))
    missing, _ = client_with(lambda request, n: httpx.Response(200, json={"models": [{"name": "llama3:latest"}]}))
    assert installed.is_available() is True
    assert missing.is_available() is False
    assert "not installed" in missing.reason


def test_get_llm_client_respects_disable_switch(monkeypatch):
    monkeypatch.setenv("EMOCONTEXT_LLM", "none")
    client = get_llm_client()
    assert isinstance(client, NullClient)
    assert "disabled" in client.reason


def test_get_llm_client_keeps_ollama_client_when_ollama_is_down(monkeypatch):
    """P4: the client is kept (not swapped for NullClient) so Ollama can start later without an API restart;
    calls fail fast and return None, which makes each agent use its fallback."""
    monkeypatch.setenv("EMOCONTEXT_LLM", "ollama")
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")   # nothing listens on port 9
    client = get_llm_client()
    assert isinstance(client, OllamaClient)
    assert "not reachable" in client.reason
    assert client.is_available(timeout=1.0) is False
    assert client.ask_json("system", "user", IntentReading) is None


def test_is_available_respects_a_short_timeout():
    def hang(request, n):
        raise httpx.ConnectTimeout("no answer", request=request)

    client, calls = client_with(hang)
    assert client.is_available(timeout=1.0) is False
    assert "ConnectTimeout" in client.reason
    assert calls[0].extensions["timeout"]["connect"] == 1.0


@pytest.mark.llm
def test_live_ollama_round_trip():
    """Needs a running Ollama with the configured model. Skipped otherwise: run with `pytest -m llm`."""
    client = OllamaClient()
    if not client.is_available():
        pytest.skip(client.reason)
    result = client.ask_json("Classify the intent of the post.", "<post>\nmy flight got cancelled again\n</post>",
                             IntentReading)
    assert isinstance(result, IntentReading), client.last_error
