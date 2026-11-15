"""Response Agent (LLM): a plain-language explanation and a suggested reply.

On a revision (the Critic rejected the first draft), the Critic's issues are added to the
prompt. Without the LLM it uses template sentences built only from the classifier's
results; the wording follows the original app's result panel.
"""
from __future__ import annotations

import time

from agents import prompts
from agents.schemas import ResponseDraft, ResponseText, TraceStep
from agents.state import AgentState

REPLY_TEMPLATES = {
    "anger": "That sounds really frustrating. Do you want to talk about what happened?",
    "fear": "That sounds stressful. I'm here if you need anything.",
    "sadness": "I'm sorry you're going through this. I'm here if you want to talk.",
    "joy": "That's great to hear, I'm happy for you!",
    "love": "That's lovely to hear!",
    "surprise": "Wow, that's unexpected! What happened?",
    "neutral": "Thanks for sharing.",
}


def fallback_draft(state: AgentState) -> ResponseDraft:
    emotion, sentiment = state["emotion"], state["sentiment"]
    sentences = [f"The model reads this post as {emotion.label} ({emotion.scores[emotion.label]:.0%}), "
                 f"so the sentiment is {sentiment.label}."]
    chance = f"{emotion.sarcasm_probability:.0%}"
    if emotion.sarcasm_rule_applied:
        sentences.append(f"It reads like {emotion.emotion_before_sarcasm}, but it looks sarcastic "
                         f"({chance} likely), so the emotion moved to {emotion.label}.")
    elif emotion.sarcasm_flagged:
        sentences.append(f"This looks sarcastic ({chance} likely).")
    if emotion.emojis_used:
        sentences.append("Emojis recognised: " + ", ".join(f"{v.char} = {v.emotion}" for v in emotion.emojis_used) + ".")
    elif emotion.text_confidence < 0.5:
        sentences.append("The model is unsure about the emotion in these words, so treat it as a weak guess.")
    return ResponseDraft(explanation=" ".join(sentences), suggested_reply=REPLY_TEMPLATES[emotion.label],
                         source="fallback")


def run(state: AgentState, llm) -> dict:
    started = time.perf_counter()
    is_revision = state.get("revision_count", 0) > 0 and "critic" in state
    issues = state["critic"].issues if is_revision else []
    reading = llm.ask_json(prompts.RESPONSE_SYSTEM, prompts.response_prompt(state, issues), ResponseText)
    errors = []
    if reading is None:
        draft = fallback_draft(state)
        if llm.provider != "none":
            errors.append(f"response agent: LLM failed ({llm.last_error}); used fallback")
    else:
        draft = ResponseDraft(**reading.model_dump(), source="llm")

    summary = ("revised draft" if is_revision else "draft") + f" written ({draft.source})"
    step = TraceStep(agent="response", used_llm=draft.source == "llm",
                     duration_ms=(time.perf_counter() - started) * 1000, summary=summary)
    return {"response": draft, "response_history": [draft], "trace": [step], "errors": errors}
