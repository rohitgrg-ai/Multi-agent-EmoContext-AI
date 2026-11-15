"""Intent Agent (LLM): what the writer wants, how urgent it is, and whether they may need support."""
from __future__ import annotations

import time

from agents import prompts
from agents.schemas import IntentReading, IntentReport, TraceStep
from agents.state import AgentState


def fallback_report() -> IntentReport:
    # Intent cannot be read from the classifier's labels alone, so the fallback says
    # "unknown" honestly instead of guessing.
    return IntentReport(intent="unknown", urgency="low", needs_support=False,
                        reason="LLM unavailable; intent was not assessed.", source="fallback")


def run(state: AgentState, llm) -> dict:
    started = time.perf_counter()
    reading = llm.ask_json(prompts.INTENT_SYSTEM, prompts.intent_prompt(state), IntentReading)
    errors = []
    if reading is None:
        report = fallback_report()
        if llm.provider != "none":
            errors.append(f"intent agent: LLM failed ({llm.last_error}); used fallback")
    else:
        report = IntentReport(**reading.model_dump(), source="llm")

    summary = f"{report.intent}, urgency {report.urgency}" + (", may need support" if report.needs_support else "")
    step = TraceStep(agent="intent", used_llm=report.source == "llm",
                     duration_ms=(time.perf_counter() - started) * 1000, summary=summary)
    return {"intent": report, "trace": [step], "errors": errors}
