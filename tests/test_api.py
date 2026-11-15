"""API tests with FastAPI's TestClient. Fast tests inject a fake predictor and NullClient;
the model-marked test starts the real lifespan with the trained model (LLM disabled)."""
import logging

import pytest
from conftest import FakePredictor, build_prediction
from fastapi.testclient import TestClient

from agents.graph import build_graph
from agents.llm import NullClient
from api.main import MAX_TEXT_CHARS, Services, create_app

SARCASTIC = build_prediction({"anger": 0.8, "sadness": 0.15, "joy": 0.05}, before_sarcasm="joy", p_sarcastic=0.99)
POST = "I just love it when my phone dies at 2%"


def fake_services(predictor=None):
    predictor = predictor or FakePredictor(SARCASTIC)
    llm = NullClient("LLM disabled for tests")
    return Services(predictor=predictor, graph=build_graph(predictor, llm), llm=llm,
                    emotion_baseline=lambda text: "joy", sentiment_baseline=lambda text: "positive")


@pytest.fixture
def client():
    with TestClient(create_app(fake_services())) as test_client:   # `with` runs the lifespan
        yield test_client


@pytest.fixture
def client_without_model():
    services = Services(llm=NullClient("LLM disabled for tests"), model_error="trained model not found")
    with TestClient(create_app(services)) as test_client:
        yield test_client


class ExplodingPredictor:
    def predict(self, text):
        raise RuntimeError(f"internal failure while reading: {text}")


# ---------- health ----------

def test_health_reports_model_and_llm(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["llm"] == {"provider": "none", "model": None, "available": False, "reason": "LLM disabled for tests"}


def test_health_is_degraded_without_model(client_without_model):
    response = client_without_model.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_error"] == "trained model not found"


# ---------- successful requests ----------

def test_analyze_returns_ml_only_result(client):
    response = client.post("/analyze", json={"text": POST})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"emotion", "sentiment", "errors"}
    assert body["emotion"]["label"] == "anger" and body["emotion"]["sarcasm_rule_applied"] is True
    assert body["sentiment"]["label"] == "negative"
    assert body["emotion"]["baseline_label"] == "joy" and body["sentiment"]["baseline_label"] == "positive"
    assert all(isinstance(value, float) for value in body["emotion"]["scores"].values())


def test_agents_analyze_runs_all_six_agents(client):
    response = client.post("/agents/analyze", json={"text": POST})
    assert response.status_code == 200
    body = response.json()
    assert [step["agent"] for step in body["trace"]] == ["emotion", "sentiment", "context", "intent", "response", "critic"]
    assert body["final"]["emotion"] == "anger" and body["final"]["emotion_source"] == "model"
    assert body["context"]["source"] == "fallback"   # NullClient: deterministic fallback
    assert body["revision_count"] == 0 and body["errors"] == []


def test_text_is_stripped_before_analysis(client):
    assert client.post("/analyze", json={"text": "   " + POST + "\n"}).status_code == 200


# ---------- invalid input ----------

@pytest.mark.parametrize("payload", [
    {"text": ""}, {"text": "   \n\t "}, {"text": "x" * (MAX_TEXT_CHARS + 1)}, {}, {"text": None}, {"post": POST},
])
@pytest.mark.parametrize("path", ["/analyze", "/agents/analyze"])
def test_invalid_input_is_rejected_with_422(client, path, payload):
    assert client.post(path, json=payload).status_code == 422


def test_text_at_the_length_limit_is_accepted(client):
    assert client.post("/analyze", json={"text": "x" * MAX_TEXT_CHARS}).status_code == 200


# ---------- missing model and unexpected errors ----------

@pytest.mark.parametrize("path", ["/analyze", "/agents/analyze"])
def test_missing_model_returns_503(client_without_model, path):
    response = client_without_model.post(path, json={"text": POST})
    assert response.status_code == 503
    assert response.json() == {"detail": "trained model not found"}


@pytest.mark.parametrize("path", ["/analyze", "/agents/analyze"])
def test_unexpected_error_returns_500_without_leaking_or_logging_text(path, caplog):
    secret = "my private post 12345"
    with TestClient(create_app(fake_services(ExplodingPredictor()))) as test_client:
        with caplog.at_level(logging.INFO):
            response = test_client.post(path, json={"text": secret})
    assert response.status_code == 500
    assert secret not in response.text
    assert "RuntimeError" in caplog.text           # the error type is logged...
    assert secret not in caplog.text               # ...but never the user's text


def test_successful_request_does_not_log_text(client, caplog):
    secret = "another private post 67890"
    with caplog.at_level(logging.DEBUG):
        client.post("/agents/analyze", json={"text": secret})
    assert secret not in caplog.text


# ---------- real model ----------

@pytest.mark.model
def test_real_lifespan_loads_model_and_matches_predictor(monkeypatch):
    from model import MODEL_DIR, Predictor
    if not (MODEL_DIR / "model.safetensors").exists():
        pytest.skip("trained model not present")
    monkeypatch.setenv("EMOCONTEXT_LLM", "none")
    expected = Predictor().predict(POST)
    with TestClient(create_app()) as test_client:
        assert test_client.get("/health").json()["model_loaded"] is True
        ml_only = test_client.post("/analyze", json={"text": POST}).json()
        full = test_client.post("/agents/analyze", json={"text": POST}).json()
    assert ml_only["emotion"]["label"] == full["final"]["emotion"] == expected["emotion"]
    assert ml_only["sentiment"]["label"] == full["final"]["sentiment"] == expected["sentiment"]


# ---------- revision history (P2, additive fields) ----------

FIELDS_BEFORE_P2 = {"final", "emotion", "sentiment", "context", "intent", "response", "critic", "revision_count", "trace", "errors"}


def test_agents_analyze_keeps_every_existing_field_and_adds_histories(client):
    body = client.post("/agents/analyze", json={"text": POST}).json()
    assert FIELDS_BEFORE_P2 <= set(body)
    assert set(body) - FIELDS_BEFORE_P2 == {"response_history", "critic_history"}
    assert body["response_history"] == [body["response"]] and body["critic_history"] == [body["critic"]]


def test_analyze_endpoint_is_unchanged(client):
    assert set(client.post("/analyze", json={"text": POST}).json()) == {"emotion", "sentiment", "errors"}


def test_api_serialises_both_passes_of_a_revision():
    from conftest import GOOD_CONTEXT, GOOD_INTENT, GOOD_RESPONSE, FakeClient
    revised = {"explanation": "The post reads as anger.", "suggested_reply": "That sounds frustrating."}
    predictor = FakePredictor(build_prediction({"anger": 0.6, "sadness": 0.4}, p_sarcastic=0.7))
    llm = FakeClient({"ContextReading": [dict(GOOD_CONTEXT, is_sarcastic=False)], "IntentReading": [GOOD_INTENT],
                      "ResponseText": [GOOD_RESPONSE, revised], "CriticReview": [{"approved": False, "issues": ["too casual"]}]})
    with TestClient(create_app(Services(predictor=predictor, graph=build_graph(predictor, llm), llm=llm))) as api:
        body = api.post("/agents/analyze", json={"text": POST}).json()
    assert [d["explanation"] for d in body["response_history"]] == [GOOD_RESPONSE["explanation"], revised["explanation"]]
    assert [(c["approved"], c["issues"]) for c in body["critic_history"]] == [(False, ["too casual"]), (True, [])]
    assert body["response"] == body["response_history"][-1] and body["critic"] == body["critic_history"][-1]
    assert body["revision_count"] == 1 and body["final"]["emotion"] == "anger"


# ---------- live Ollama status (P4) ----------

class SwitchableOllama:
    """A fake Ollama server that can be switched up, down, made to hang, or lose the model."""
    def __init__(self, state="up"):
        self.state, self.tag_calls = state, 0

    def __call__(self, request):
        import json as _json
        import httpx as _httpx
        if self.state == "down":
            raise _httpx.ConnectError("connection refused", request=request)
        if self.state == "hang":
            raise _httpx.ConnectTimeout("no answer", request=request)
        if request.url.path == "/api/tags":
            self.tag_calls += 1
            names = [] if self.state == "no-model" else ["gemma4:latest"]
            return _httpx.Response(200, json={"models": [{"name": n} for n in names]})
        reply = {"ContextReading": GOOD_CONTEXT_JSON, "IntentReading": GOOD_INTENT_JSON}.get(
            _json.loads(request.content)["format"].get("title"), GOOD_RESPONSE_JSON)
        return _httpx.Response(200, json={"message": {"content": _json.dumps(reply)}})


from conftest import GOOD_CONTEXT as GOOD_CONTEXT_JSON, GOOD_INTENT as GOOD_INTENT_JSON, GOOD_RESPONSE as GOOD_RESPONSE_JSON  # noqa: E402


def live_app(fake):
    import httpx as _httpx
    from agents.llm import OllamaClient
    llm = OllamaClient(transport=_httpx.MockTransport(fake))
    predictor = FakePredictor(SARCASTIC)
    return create_app(Services(predictor=predictor, graph=build_graph(predictor, llm), llm=llm))


def test_health_follows_ollama_starting_and_stopping_without_restart():
    fake = SwitchableOllama("down")
    with TestClient(live_app(fake)) as api:
        down = api.get("/health").json()["llm"]
        fake.state = "up"
        up = api.get("/health").json()["llm"]
        fake.state = "down"
        down_again = api.get("/health").json()["llm"]
    assert (down["available"], up["available"], down_again["available"]) == (False, True, False)
    assert down["provider"] == up["provider"] == "ollama" and up["model"] == "gemma4:latest"
    assert "not reachable" in down["reason"] and up["reason"] == ""


def test_health_reports_missing_model_and_hanging_server():
    fake = SwitchableOllama("no-model")
    with TestClient(live_app(fake)) as api:
        missing = api.get("/health").json()["llm"]
        fake.state = "hang"
        hanging = api.get("/health").json()["llm"]
    assert missing["available"] is False and "not installed" in missing["reason"]
    assert hanging["available"] is False and "ConnectTimeout" in hanging["reason"]


def test_analysis_recovers_when_ollama_comes_back_without_restart():
    import time
    fake = SwitchableOllama("down")
    with TestClient(live_app(fake)) as api:
        started = time.perf_counter()
        offline = api.post("/agents/analyze", json={"text": POST}).json()
        offline_seconds = time.perf_counter() - started
        fake.state = "up"
        online = api.post("/agents/analyze", json={"text": POST}).json()
    # Ollama down: every LLM agent falls back fast, failures are recorded, nothing crashes
    assert [s["agent"] for s in offline["trace"]] == ["emotion", "sentiment", "context", "intent", "response", "critic"]
    assert {offline[k]["source"] for k in ("context", "intent", "response")} == {"fallback"}
    assert len(offline["errors"]) == 3 and all("ConnectError" in e for e in offline["errors"])
    assert offline["final"]["emotion"] == "anger" and offline_seconds < 2.0
    # Ollama back: the same running API now uses it
    assert {online[k]["source"] for k in ("context", "intent", "response")} == {"llm"}
    assert online["errors"] == [] and online["final"]["emotion"] == "anger"


def test_disable_switch_still_uses_null_client_and_never_calls_ollama(monkeypatch):
    from agents.llm import get_llm_client
    monkeypatch.setenv("EMOCONTEXT_LLM", "none")
    llm = get_llm_client()
    predictor = FakePredictor(SARCASTIC)
    with TestClient(create_app(Services(predictor=predictor, graph=build_graph(predictor, llm), llm=llm))) as api:
        health = api.get("/health").json()["llm"]
        result = api.post("/agents/analyze", json={"text": POST}).json()
    assert health == {"provider": "none", "model": None, "available": False, "reason": "LLM disabled by EMOCONTEXT_LLM=none"}
    assert result["errors"] == [] and result["context"]["source"] == "fallback"
