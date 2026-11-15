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
