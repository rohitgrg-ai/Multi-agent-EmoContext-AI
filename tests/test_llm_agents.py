"""Context, Intent and Response agents: LLM success, malformed/failed LLM, and NullClient fallback."""
import json

from conftest import GOOD_CONTEXT, GOOD_INTENT, GOOD_RESPONSE, FakeClient, build_state

from agents import context_agent, intent_agent, response_agent
from agents.llm import NullClient
from agents.schemas import CriticReport


def with_context_and_intent(state, client):
    state.update(context_agent.run(state, client))
    state.update(intent_agent.run(state, client))
    return state


# ---------- Context Agent ----------

def test_context_agent_uses_llm_reading(state):
    update = context_agent.run(state, FakeClient({"ContextReading": [GOOD_CONTEXT]}))
    report = update["context"]
    assert report.source == "llm" and report.is_sarcastic and report.cues == ["love it", "2%"]
    assert update["trace"][0].used_llm is True
    assert update["errors"] == []


def test_context_agent_drops_alternative_that_matches_the_classifier(state):
    agreeing = dict(GOOD_CONTEXT, proposed_emotion="anger")   # classifier also says anger
    assert context_agent.run(state, FakeClient({"ContextReading": [agreeing]}))["context"].proposed_emotion is None


def test_context_agent_keeps_a_real_alternative_separately(state):
    update = context_agent.run(state, FakeClient({"ContextReading": [dict(GOOD_CONTEXT, proposed_emotion="sadness")]}))
    assert update["context"].proposed_emotion == "sadness"
    assert state["emotion"].label == "anger"   # the classifier's result is untouched


def test_context_agent_fallback_with_null_client_records_no_error():
    state = build_state("so fun 🙄🙄", scores={"anger": 0.9, "joy": 0.1}, p_sarcastic=0.95, emojis=[("🙄", "anger")] * 2)
    update = context_agent.run(state, NullClient())
    report = update["context"]
    assert report.source == "fallback" and report.is_sarcastic is True
    assert report.cues == ["🙄"]
    assert update["errors"] == []          # LLM disabled on purpose: not an error
    assert update["trace"][0].used_llm is False


def test_context_agent_failed_llm_falls_back_and_records_error(state):
    update = context_agent.run(state, FakeClient({"ContextReading": [None]}))
    assert update["context"].source == "fallback"
    assert "LLM failed" in update["errors"][0]


# ---------- Intent Agent ----------

def test_intent_agent_uses_llm_reading(state):
    state.update(context_agent.run(state, NullClient()))
    report = intent_agent.run(state, FakeClient({"IntentReading": [GOOD_INTENT]}))["intent"]
    assert (report.intent, report.urgency, report.source) == ("complain", "low", "llm")


def test_intent_agent_fallback_is_unknown_not_a_guess(state):
    state.update(context_agent.run(state, NullClient()))
    report = intent_agent.run(state, NullClient())["intent"]
    assert (report.intent, report.needs_support, report.source) == ("unknown", False, "fallback")


# ---------- Response Agent ----------

def test_response_agent_uses_llm_draft(state):
    with_context_and_intent(state, NullClient())
    draft = response_agent.run(state, FakeClient({"ResponseText": [GOOD_RESPONSE]}))["response"]
    assert draft.source == "llm" and "anger" in draft.explanation


def test_response_fallback_mentions_label_and_sarcasm_shift():
    state = build_state(scores={"anger": 0.9, "joy": 0.1}, before_sarcasm="joy", p_sarcastic=0.99)
    with_context_and_intent(state, NullClient())
    draft = response_agent.run(state, NullClient())["response"]
    assert draft.source == "fallback"
    assert "anger" in draft.explanation and "sarcastic (99% likely)" in draft.explanation
    assert draft.suggested_reply
    json.dumps(draft.model_dump())


def test_response_revision_sends_critic_issues_to_the_llm(state):
    with_context_and_intent(state, NullClient())
    state["revision_count"] = 1
    state["critic"] = CriticReport(approved=False, issues=["Name the emotion anger."], rule_checks={},
                                   llm_reviewed=False, override_eligible=False)
    client = FakeClient({"ResponseText": [GOOD_RESPONSE]})
    update = response_agent.run(state, client)
    assert "Name the emotion anger." in client.calls[0][1]
    assert update["trace"][0].summary.startswith("revised")
