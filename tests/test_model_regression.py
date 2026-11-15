"""Regression tests against the real trained model. Run with: pytest -m model

They check that the agents reproduce `Predictor.predict` exactly, and that the README's
example posts still give the documented answers.
"""
import json

import pytest

from agents import emotion_agent, sentiment_agent

pytestmark = pytest.mark.model

# From the README results table: (post, emotion, sentiment, sarcasm probability shown as a percentage).
README_EXAMPLES = [
    ("I just love it when my phone dies at 2%", "anger", "negative", 0.99),
    ("I got the job!!!", "joy", "positive", 0.02),
    ("my hands won't stop shaking before this interview", "fear", "negative", 0.14),
    ("The meeting is at 3pm on Tuesday.", "neutral", "neutral", 0.01),
]


@pytest.fixture(scope="module")
def predictor():
    from model import MODEL_DIR, Predictor
    if not (MODEL_DIR / "model.safetensors").exists():
        pytest.skip("trained model not present in models/emotion_sarcasm")
    return Predictor()


@pytest.mark.parametrize("text, emotion, sentiment, sarcasm", README_EXAMPLES)
def test_readme_examples_unchanged(predictor, text, emotion, sentiment, sarcasm):
    prediction = predictor.predict(text)
    assert prediction["emotion"] == emotion
    assert prediction["sentiment"] == sentiment
    assert prediction["sarcasm_probability"] == pytest.approx(sarcasm, abs=0.01)   # README rounds to whole %


@pytest.mark.parametrize("text", [row[0] for row in README_EXAMPLES] + ["Great, another Monday 🙄", "😂😂😂"])
def test_agents_reproduce_predictor_exactly(predictor, text):
    prediction = predictor.predict(text)
    emotion_update = emotion_agent.run({"text": text}, predictor)
    sentiment_update = sentiment_agent.run({"text": text, "emotion": emotion_update["emotion"]})
    emotion, sentiment = emotion_update["emotion"], sentiment_update["sentiment"]

    assert emotion.label == prediction["emotion"]
    assert emotion.scores == {name: float(score) for name, score in prediction["emotion_scores"].items()}
    assert emotion.sarcasm_probability == prediction["sarcasm_probability"]
    assert sentiment.label == prediction["sentiment"]
    assert sentiment.scores == prediction["sentiment_scores"]
    json.dumps({"emotion": emotion.model_dump(), "sentiment": sentiment.model_dump()})
