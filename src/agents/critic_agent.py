"""Critic Agent: checks the other agents' work and writes the final answer.

Order of work, cheapest first:
1. Deterministic rule checks (no LLM).
2. If a check on the Response Agent's draft fails and no revision has been used yet,
   send the draft back once with the list of issues. No LLM call is made for this.
3. Only if every rule passes AND the post is a "review case" (see `review_reasons`) is the
   LLM asked for one short faithfulness review. Never on the second pass, so a post costs
   at most 5 LLM calls in total.

Emotion overrides are disabled: the final emotion is always the trained classifier's. The
Context Agent's suggestion is reported as `alternative_emotion`, and `override_eligible`
records whether it would have passed every override condition, so evaluation can measure
whether enabling overrides would help before anyone turns them on.
"""
from __future__ import annotations

import time
from typing import Dict, List

from agents import faithfulness, prompts
from agents.schemas import CriticReport, CriticReview, FinalResult, TraceStep
from agents.state import AgentState
from model import EMOTION_SENTIMENT, SARCASM_THRESHOLD

MAX_REVISIONS = 1
MAX_EXPLANATION_CHARS = 800
MAX_REPLY_CHARS = 400
UNCERTAIN_SARCASM_FLOOR = 0.5      # 0.5 <= P(sarcasm) < SARCASM_THRESHOLD is the uncertain band
LOW_TEXT_CONFIDENCE = 0.5          # same "weak guess" cut-off as the original app

# Checks on the Response Agent's draft. Only these can be fixed by a revision.
# The faithfulness checks (agents/faithfulness.py) compare the explanation's factual claims
# with the real model outputs, so a wrong number is caught without an LLM call.
RESPONSE_CHECKS = ("explanation_present", "reply_present", "explanation_length_ok",
                   "reply_length_ok", "explanation_names_emotion") + faithfulness.CHECK_NAMES


def cues_in_post(cues: List[str], text: str) -> bool:
    lowered = text.lower()
    return all(cue.strip().lower() in lowered for cue in cues if cue.strip())


def faithfulness_problems(state: AgentState) -> Dict[str, List[str]]:
    return faithfulness.check_explanation(state["text"], state["emotion"], state["sentiment"],
                                          state["response"].explanation)


def rule_checks(state: AgentState) -> Dict[str, bool]:
    emotion, sentiment, context, response = state["emotion"], state["sentiment"], state["context"], state["response"]
    checks = {
        "sentiment_matches_emotion": sentiment.label == EMOTION_SENTIMENT[emotion.label],
        "context_cues_in_post": cues_in_post(context.cues, state["text"]),
        "explanation_present": bool(response.explanation.strip()),
        "reply_present": bool(response.suggested_reply.strip()),
        "explanation_length_ok": len(response.explanation) <= MAX_EXPLANATION_CHARS,
        "reply_length_ok": len(response.suggested_reply) <= MAX_REPLY_CHARS,
        "explanation_names_emotion": emotion.label in response.explanation.lower(),
    }
    checks.update({name: not problems for name, problems in faithfulness_problems(state).items()})
    return checks


def ml_is_uncertain(state: AgentState) -> bool:
    emotion = state["emotion"]
    return (UNCERTAIN_SARCASM_FLOOR <= emotion.sarcasm_probability < SARCASM_THRESHOLD
            or emotion.text_confidence < LOW_TEXT_CONFIDENCE)


def review_reasons(state: AgentState) -> List[str]:
    """The cases where an LLM review is worth its cost. Empty list means rules are enough."""
    reasons = []
    if ml_is_uncertain(state):
        reasons.append("classifier uncertain")
    context = state["context"]
    if context.source == "llm" and context.is_sarcastic != state["emotion"].sarcasm_flagged:
        reasons.append("context and classifier disagree on sarcasm")
    if state["intent"].needs_support:
        reasons.append("writer may need support")
    return reasons


def override_eligible(state: AgentState) -> bool:
    """Would the Context Agent's alternative pass every override condition? (Never applied.)"""
    emotion, context = state["emotion"], state["context"]
    alternative = context.proposed_emotion
    # Emotions scored exactly 0 (e.g. after an emoji-only vote) are only in top3 by tie order.
    real_top3 = [name for name in emotion.top3 if emotion.scores[name] > 0]
    return (context.source == "llm" and alternative is not None and alternative != emotion.label
            and alternative in real_top3 and bool(context.cues)
            and cues_in_post(context.cues, state["text"]) and ml_is_uncertain(state))


def issues_from(checks: Dict[str, bool], state: AgentState) -> List[str]:
    messages = {
        "sentiment_matches_emotion": "Sentiment does not follow the classifier's emotion.",
        "context_cues_in_post": "Some Context Agent cues are not found in the post.",
        "explanation_present": "The explanation is empty.",
        "reply_present": "The suggested reply is empty.",
        "explanation_length_ok": f"The explanation is longer than {MAX_EXPLANATION_CHARS} characters.",
        "reply_length_ok": f"The suggested reply is longer than {MAX_REPLY_CHARS} characters.",
        "explanation_names_emotion": f"The explanation must name the emotion '{state['emotion'].label}'.",
    }
    issues = [messages[name] for name, passed in checks.items() if not passed and name in messages]
    for name, problems in faithfulness_problems(state).items():   # specific, actionable messages
        if not checks.get(name, True):
            issues.extend(problems)
    return issues


def final_result(state: AgentState, alternative) -> FinalResult:
    intent, response = state["intent"], state["response"]
    return FinalResult(emotion=state["emotion"].label, sentiment=state["sentiment"].label,
                       emotion_source="model", alternative_emotion=alternative,
                       intent=intent.intent, urgency=intent.urgency, needs_support=intent.needs_support,
                       explanation=response.explanation, suggested_reply=response.suggested_reply)


def run(state: AgentState, llm) -> dict:
    started = time.perf_counter()
    revisions_used = state.get("revision_count", 0)
    can_revise = revisions_used < MAX_REVISIONS
    checks = rule_checks(state)
    issues = issues_from(checks, state)
    response_ok = all(checks[name] for name in RESPONSE_CHECKS)

    llm_reviewed = False
    reasons = review_reasons(state) if response_ok else []
    if reasons and can_revise:   # LLM review only on the first pass, and only for review cases
        review = llm.ask_json(prompts.CRITIC_SYSTEM, prompts.critic_prompt(state), CriticReview)
        if review is not None:
            llm_reviewed = True
            if not review.approved:
                issues += review.issues or ["The reviewer rejected the draft without details."]
                response_ok = False

    send_back = not response_ok and can_revise
    report = CriticReport(approved=response_ok, issues=issues, rule_checks=checks, llm_reviewed=llm_reviewed,
                          override_eligible=override_eligible(state),
                          alternative_emotion=state["context"].proposed_emotion)

    if send_back:
        summary = f"sent back for revision: {len(issues)} issue(s)"
    elif response_ok:
        summary = "approved" + (" after LLM review" if llm_reviewed else " by rules")
    else:
        summary = f"not approved after {revisions_used} revision(s); kept with {len(issues)} issue(s)"
    if reasons:
        summary += f" [review case: {', '.join(reasons)}]"
    step = TraceStep(agent="critic", used_llm=llm_reviewed,
                     duration_ms=(time.perf_counter() - started) * 1000, summary=summary)

    update = {"critic": report, "trace": [step], "errors": []}
    if send_back:
        update["revision_count"] = revisions_used + 1
    else:
        update["final"] = final_result(state, report.alternative_emotion)
    return update
