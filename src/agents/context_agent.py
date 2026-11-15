"""Context Agent (LLM): reads the implied meaning, sarcasm and supporting cues.

It may suggest a different emotion, but that suggestion is stored separately as an
alternative. The classifier's emotion stays the final answer (see critic_agent.py).
"""
from __future__ import annotations

import time

from agents import prompts
from agents.schemas import ContextReading, ContextReport, TraceStep
from agents.state import AgentState


def fallback_report(state: AgentState) -> ContextReport:
    """Used when the LLM is unavailable: built only from the classifier's own signals."""
    emotion = state["emotion"]
    if emotion.sarcasm_flagged:
        meaning = "The post looks sarcastic, so it probably means the opposite of its literal words."
    else:
        meaning = f"The post reads as {emotion.label}, taken at face value."
    emoji_cues = list(dict.fromkeys(vote.char for vote in emotion.emojis_used))   # unique, in order
    return ContextReport(implied_meaning=meaning, is_sarcastic=emotion.sarcasm_flagged, cues=emoji_cues,
                         proposed_emotion=None, reason="LLM unavailable; based only on the classifier's "
                         "sarcasm score and emojis.", source="fallback")


def run(state: AgentState, llm) -> dict:
    started = time.perf_counter()
    reading = llm.ask_json(prompts.CONTEXT_SYSTEM, prompts.context_prompt(state), ContextReading)
    errors = []
    if reading is None:
        report = fallback_report(state)
        if llm.provider != "none":
            errors.append(f"context agent: LLM failed ({llm.last_error}); used fallback")
    else:
        report = ContextReport(**reading.model_dump(), source="llm")
        if report.proposed_emotion == state["emotion"].label:
            report.proposed_emotion = None   # agreeing with the classifier is not an alternative

    summary = f"{'sarcastic' if report.is_sarcastic else 'literal'}; {len(report.cues)} cue(s)"
    if report.proposed_emotion:
        summary += f"; suggests {report.proposed_emotion} as an alternative"
    step = TraceStep(agent="context", used_llm=report.source == "llm",
                     duration_ms=(time.perf_counter() - started) * 1000, summary=summary)
    return {"context": report, "trace": [step], "errors": errors}
