"""Shared fakes for the agent and graph tests. Nothing here loads the model or calls Ollama."""
import numpy as np
import pytest

from agents import emotion_agent, sentiment_agent
from model import EMOTIONS


def build_prediction(scores, text="", before_sarcasm=None, p_sarcastic=0.1, text_confidence=0.8, emojis=()):
    """A dict shaped exactly like Predictor.predict output (numpy floats, emoji tuples)."""
    final = np.array([scores.get(name, 0.0) for name in EMOTIONS], dtype=np.float32)
    top = EMOTIONS[int(np.argmax(final))]
    return {
        "emotion": top, "emotion_from_text": before_sarcasm or top, "emotion_before_sarcasm": before_sarcasm or top,
        "emotion_scores": dict(zip(EMOTIONS, final)), "text_confidence": np.float32(text_confidence),
        "sentiment": None, "sentiment_scores": None, "sarcasm_probability": p_sarcastic,
        "sarcasm_flagged": p_sarcastic >= 0.90, "emojis_used": list(emojis),
    }


class FakePredictor:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, text):
        return self.prediction


class FakeClient:
    """Stands in for OllamaClient. `replies` maps a schema name to a list of replies used in order;
    the last reply repeats. A reply of None simulates an LLM failure."""
    provider = "fake"
    model = "fake-model"
    reason = ""

    def __init__(self, replies=None):
        self.replies = {name: list(queue) for name, queue in (replies or {}).items()}
        self.calls = []
        self.last_error = None

    def ask_json(self, system, user, schema):
        self.calls.append((schema.__name__, user))
        queue = self.replies.get(schema.__name__)
        if not queue:
            self.last_error = "no fake reply configured"
            return None
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if reply is None:
            self.last_error = "simulated failure"
            return None
        return schema(**reply)

    def count(self, schema_name):
        return sum(1 for name, _ in self.calls if name == schema_name)


GOOD_CONTEXT = {"implied_meaning": "The writer is annoyed that the phone died.", "is_sarcastic": True,
                "cues": ["love it", "2%"], "proposed_emotion": None, "reason": "praise for a bad event"}
GOOD_INTENT = {"intent": "complain", "urgency": "low", "needs_support": False, "reason": "minor annoyance"}
GOOD_RESPONSE = {"explanation": "The post reads as anger: the praise is sarcastic.",
                 "suggested_reply": "Ugh, that is so annoying. Time for a new battery?"}


def build_state(text="I just love it when my phone dies at 2%", **prediction_args):
    """State after the Emotion and Sentiment agents, from a fake prediction."""
    prediction_args.setdefault("scores", {"anger": 0.8, "sadness": 0.15, "joy": 0.05})
    state = {"text": text, "revision_count": 0, "trace": [], "errors": []}
    state.update(emotion_agent.run(state, FakePredictor(build_prediction(text=text, **prediction_args))))
    state.update(sentiment_agent.run(state))
    return state


@pytest.fixture
def state():
    return build_state()
