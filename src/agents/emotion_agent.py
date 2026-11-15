"""Emotion Agent: runs the trained RoBERTa pipeline and the TF-IDF baseline. No LLM.

The prediction itself comes from `model.Predictor.predict`, unchanged: RoBERTa scores,
then the emoji vote, then the sarcasm rule. This agent only converts that output into
the typed, JSON-safe `EmotionResult` (numpy floats become plain floats, emoji tuples
become objects) and adds the top 3 emotions and the confidence margin.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from agents.schemas import EmojiVote, EmotionResult, TraceStep
from agents.state import AgentState


def to_emotion_result(prediction: dict, baseline_label: Optional[str] = None) -> EmotionResult:
    scores = {name: float(score) for name, score in prediction["emotion_scores"].items()}
    # sorted() is stable, so ties keep the model's label order, the same tie-break as numpy argmax.
    ranked = sorted(scores, key=scores.get, reverse=True)
    return EmotionResult(
        label=prediction["emotion"],
        scores=scores,
        top3=ranked[:3],
        margin=scores[ranked[0]] - scores[ranked[1]],
        text_confidence=float(prediction["text_confidence"]),
        emotion_from_text=prediction["emotion_from_text"],
        emotion_before_sarcasm=prediction["emotion_before_sarcasm"],
        sarcasm_probability=float(prediction["sarcasm_probability"]),
        sarcasm_flagged=bool(prediction["sarcasm_flagged"]),
        sarcasm_rule_applied=prediction["emotion"] != prediction["emotion_before_sarcasm"],
        emojis_used=[EmojiVote(char=char, emotion=emotion) for char, emotion in prediction["emojis_used"]],
        baseline_label=baseline_label,
    )


def run(state: AgentState, predictor, emotion_baseline: Optional[Callable[[str], str]] = None) -> dict:
    """LangGraph node. `predictor` is a loaded model.Predictor; the baseline is optional."""
    started = time.perf_counter()
    text = state["text"]
    prediction = predictor.predict(text)   # a model failure is a real error, so it is not caught here

    errors = []
    baseline_label = None
    if emotion_baseline is not None:
        try:
            baseline_label = emotion_baseline(text)
        except Exception as error:   # the baseline is only for comparison; never let it stop the pipeline
            errors.append(f"emotion baseline failed: {error}")

    result = to_emotion_result(prediction, baseline_label)
    summary = f"{result.label} (margin {result.margin:.2f}), sarcasm {result.sarcasm_probability:.0%}"
    if result.sarcasm_rule_applied:
        summary += f", sarcasm rule changed {result.emotion_before_sarcasm} to {result.label}"
    step = TraceStep(agent="emotion", used_llm=False,
                     duration_ms=(time.perf_counter() - started) * 1000, summary=summary)
    return {"emotion": result, "trace": [step], "errors": errors}
