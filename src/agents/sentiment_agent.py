"""Sentiment Agent: derives sentiment from the emotion, exactly as the original app does. No LLM.

The label comes from `model.EMOTION_SENTIMENT[emotion]`, and the scores are the emotion
scores summed per sentiment, the same calculation as `Predictor.predict`. Because the
label follows the top emotion while the scores are sums, the two can disagree (for
example joy at 0.35 but negative emotions summing to 0.6). That is a known limitation of
the original design, so the agent reports it in `consistent_with_scores` instead of hiding it.
The TF-IDF + SVM sentiment model is run for comparison only.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from agents.schemas import EmotionResult, SentimentResult, TraceStep
from agents.state import AgentState
from model import EMOTION_SENTIMENT


def to_sentiment_result(emotion: EmotionResult, baseline_label: Optional[str] = None) -> SentimentResult:
    scores = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    for name, score in emotion.scores.items():
        scores[EMOTION_SENTIMENT[name]] += score
    label = EMOTION_SENTIMENT[emotion.label]
    return SentimentResult(
        label=label,
        scores=scores,
        consistent_with_scores=scores[label] == max(scores.values()),
        baseline_label=baseline_label,
    )


def run(state: AgentState, sentiment_baseline: Optional[Callable[[str], str]] = None) -> dict:
    """LangGraph node. Reads `emotion` (and `text` for the baseline)."""
    started = time.perf_counter()
    errors = []
    baseline_label = None
    if sentiment_baseline is not None:
        try:
            baseline_label = sentiment_baseline(state["text"])
        except Exception as error:   # comparison only; never let it stop the pipeline
            errors.append(f"sentiment baseline failed: {error}")

    result = to_sentiment_result(state["emotion"], baseline_label)
    summary = f"{result.label} (from emotion {state['emotion'].label})"
    if not result.consistent_with_scores:
        summary += "; note: summed scores favour a different sentiment"
    step = TraceStep(agent="sentiment", used_llm=False,
                     duration_ms=(time.perf_counter() - started) * 1000, summary=summary)
    return {"sentiment": result, "trace": [step], "errors": errors}
