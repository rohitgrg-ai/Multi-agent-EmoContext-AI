"""Unit tests for the Emotion and Sentiment agents, using a fake predictor (no model loaded).

The fake output copies the real `Predictor.predict` types: numpy float32 scores and
(char, emotion) tuples, because those are exactly what broke JSON serialisation (B2).
"""
import json
import typing

import numpy as np
import pytest

from agents import emotion_agent, sentiment_agent
from agents.schemas import EmotionLabel
from model import EMOTIONS


def fake_prediction(scores, emotion=None, before_sarcasm=None, from_text=None,
                    p_sarcastic=0.1, text_confidence=0.8, emojis=()):
    final = np.array([scores.get(name, 0.0) for name in EMOTIONS], dtype=np.float32)
    top = EMOTIONS[int(np.argmax(final))]
    return {
        "emotion": emotion or top,
        "emotion_from_text": from_text or top,
        "emotion_before_sarcasm": before_sarcasm or top,
        "emotion_scores": dict(zip(EMOTIONS, final)),          # numpy float32 values, like the real one
        "text_confidence": np.float32(text_confidence),
        "sentiment": None,                                      # not read by the agents
        "sentiment_scores": None,
        "sarcasm_probability": p_sarcastic,
        "sarcasm_flagged": p_sarcastic >= 0.90,
        "emojis_used": list(emojis),                           # tuples, like the real one
    }


class FakePredictor:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, text):
        return self.prediction


def test_emotion_labels_in_schema_match_model():
    assert list(typing.get_args(EmotionLabel)) == EMOTIONS


# ---------- Emotion Agent ----------

def test_emotion_result_is_json_serialisable_with_plain_floats():
    prediction = fake_prediction({"joy": 0.7, "love": 0.2, "neutral": 0.1}, emojis=[("😂", "joy")])
    result = emotion_agent.to_emotion_result(prediction)
    dumped = result.model_dump()
    json.dumps(dumped)   # raises if a numpy value or tuple slipped through
    assert all(type(value) is float for value in dumped["scores"].values())
    assert dumped["emojis_used"] == [{"char": "😂", "emotion": "joy"}]


def test_emotion_result_top3_and_margin():
    result = emotion_agent.to_emotion_result(fake_prediction({"anger": 0.5, "sadness": 0.3, "fear": 0.15, "joy": 0.05}))
    assert result.label == "anger"
    assert result.top3 == ["anger", "sadness", "fear"]
    assert result.margin == pytest.approx(0.2, abs=1e-6)


def test_emotion_result_reports_when_sarcasm_rule_changed_the_label():
    prediction = fake_prediction({"anger": 0.8, "sadness": 0.2}, before_sarcasm="joy", p_sarcastic=0.97)
    result = emotion_agent.to_emotion_result(prediction)
    assert result.sarcasm_rule_applied and result.sarcasm_flagged
    assert result.emotion_before_sarcasm == "joy"


def test_emotion_agent_run_writes_result_trace_and_baseline():
    predictor = FakePredictor(fake_prediction({"fear": 0.9, "sadness": 0.1}))
    update = emotion_agent.run({"text": "so scared"}, predictor, emotion_baseline=lambda text: "sadness")
    assert update["emotion"].label == "fear"
    assert update["emotion"].baseline_label == "sadness"
    assert update["errors"] == []
    [step] = update["trace"]
    assert step.agent == "emotion" and step.used_llm is False


def test_emotion_agent_survives_a_failing_baseline():
    def broken_baseline(text):
        raise RuntimeError("vectorizer missing")

    update = emotion_agent.run({"text": "hi"}, FakePredictor(fake_prediction({"neutral": 1.0})), broken_baseline)
    assert update["emotion"].label == "neutral"
    assert update["emotion"].baseline_label is None
    assert "vectorizer missing" in update["errors"][0]


# ---------- Sentiment Agent ----------

def test_sentiment_follows_emotion_and_sums_scores():
    emotion = emotion_agent.to_emotion_result(fake_prediction({"joy": 0.6, "love": 0.2, "anger": 0.2}))
    result = sentiment_agent.to_sentiment_result(emotion)
    assert result.label == "positive"
    assert result.scores["positive"] == pytest.approx(0.8, abs=1e-6)
    assert result.scores["negative"] == pytest.approx(0.2, abs=1e-6)
    assert result.consistent_with_scores is True


def test_sentiment_flags_label_that_disagrees_with_summed_scores():
    # Joy is the single top emotion, but the negative emotions together outweigh it.
    emotion = emotion_agent.to_emotion_result(
        fake_prediction({"joy": 0.35, "anger": 0.25, "sadness": 0.2, "fear": 0.2}))
    result = sentiment_agent.to_sentiment_result(emotion)
    assert result.label == "positive"
    assert result.consistent_with_scores is False


def test_sentiment_agent_run_writes_result_trace_and_baseline():
    emotion = emotion_agent.to_emotion_result(fake_prediction({"sadness": 1.0}))
    update = sentiment_agent.run({"text": "miss you", "emotion": emotion}, sentiment_baseline=lambda text: "negative")
    assert update["sentiment"].label == "negative"
    assert update["sentiment"].baseline_label == "negative"
    assert update["trace"][0].agent == "sentiment"
    json.dumps(update["sentiment"].model_dump())
