"""End-to-end graph tests with a fake predictor (fast) and the real model (pytest -m model)."""
import json

import pytest
from conftest import GOOD_CONTEXT, GOOD_INTENT, GOOD_RESPONSE, FakeClient, FakePredictor, build_prediction

from agents.graph import analyse, build_graph, state_to_dict
from agents.llm import NullClient

AGENT_ORDER = ["emotion", "sentiment", "context", "intent", "response", "critic"]
SARCASTIC = build_prediction({"anger": 0.8, "sadness": 0.15, "joy": 0.05}, before_sarcasm="joy", p_sarcastic=0.99)
POST = "I just love it when my phone dies at 2%"


def test_full_graph_without_ollama_runs_all_six_agents():
    state = analyse(build_graph(FakePredictor(SARCASTIC), NullClient()), POST)
    assert [step.agent for step in state["trace"]] == AGENT_ORDER
    assert not any(step.used_llm for step in state["trace"])
    assert state["final"].emotion == SARCASTIC["emotion"] == "anger"
    assert state["final"].emotion_source == "model"
    assert state["context"].source == state["intent"].source == state["response"].source == "fallback"
    assert state["errors"] == []
    json.dumps(state_to_dict(state), ensure_ascii=False)


def test_full_graph_with_fake_llm_uses_three_calls_for_a_clear_case():
    client = FakeClient({"ContextReading": [GOOD_CONTEXT], "IntentReading": [GOOD_INTENT],
                         "ResponseText": [GOOD_RESPONSE], "CriticReview": [{"approved": True, "issues": []}]})
    state = analyse(build_graph(FakePredictor(SARCASTIC), client), POST)
    assert [step.agent for step in state["trace"]] == AGENT_ORDER
    assert len(client.calls) == 3   # no review case, so the Critic decides by rules alone
    assert state["final"].explanation == GOOD_RESPONSE["explanation"]
    assert state["final"].emotion == "anger"


def test_critic_rejection_causes_exactly_one_revision():
    uncertain = build_prediction({"anger": 0.6, "sadness": 0.4}, p_sarcastic=0.7)   # review case
    client = FakeClient({"ContextReading": [dict(GOOD_CONTEXT, is_sarcastic=False)], "IntentReading": [GOOD_INTENT],
                         "ResponseText": [GOOD_RESPONSE], "CriticReview": [{"approved": False, "issues": ["too casual"]}]})
    state = analyse(build_graph(FakePredictor(uncertain), client), POST)
    assert [step.agent for step in state["trace"]] == AGENT_ORDER + ["response", "critic"]
    assert state["revision_count"] == 1
    assert client.count("ResponseText") == 2
    assert client.count("CriticReview") == 1   # the second Critic pass uses rules only
    assert len(client.calls) == 5              # the documented maximum
    assert state["final"].emotion == "anger"


def test_llm_failures_mid_run_still_finish_with_fallbacks():
    client = FakeClient({"ContextReading": [GOOD_CONTEXT], "IntentReading": [None], "ResponseText": [None]})
    state = analyse(build_graph(FakePredictor(SARCASTIC), client), POST)
    assert state["intent"].source == "fallback" and state["response"].source == "fallback"
    assert len(state["errors"]) == 2
    assert "final" in state


@pytest.mark.model
def test_real_model_final_emotion_matches_original_prediction():
    from baseline import load_keyword_model, load_sentiment_baseline
    from model import MODEL_DIR, Predictor
    if not (MODEL_DIR / "model.safetensors").exists():
        pytest.skip("trained model not present")
    predictor = Predictor()
    graph = build_graph(predictor, NullClient(), load_keyword_model(), load_sentiment_baseline())
    for text in [POST, "I got the job!!!", "my hands won't stop shaking before this interview",
                 "The meeting is at 3pm on Tuesday.", "😂😂😂", "Great, another Monday 🙄"]:
        expected = predictor.predict(text)
        state = analyse(graph, text)
        assert state["final"].emotion == expected["emotion"]
        assert state["final"].sentiment == expected["sentiment"]
        json.dumps(state_to_dict(state), ensure_ascii=False)
