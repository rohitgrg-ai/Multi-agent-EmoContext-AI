"""Critic Agent: rule checks, when the LLM is (not) consulted, the revision cap and override eligibility."""
import pytest
from conftest import GOOD_CONTEXT, GOOD_INTENT, GOOD_RESPONSE, FakeClient, build_state

from agents import critic_agent
from agents.llm import NullClient
from agents.schemas import ContextReport, IntentReport, ResponseDraft

REJECT = {"approved": False, "issues": ["The reply is too casual."]}
APPROVE = {"approved": True, "issues": []}


def ready_state(context=None, intent=None, response=None, **prediction_args):
    """State as the Critic sees it, with every earlier agent's output filled in."""
    state = build_state(**prediction_args)
    state["context"] = ContextReport(**(context or dict(GOOD_CONTEXT, is_sarcastic=False)), source="llm")
    state["intent"] = IntentReport(**(intent or GOOD_INTENT), source="llm")
    state["response"] = ResponseDraft(**(response or GOOD_RESPONSE), source="llm")
    return state


# ---------- rules decide alone when there is no review case ----------

def test_clear_case_is_approved_by_rules_without_calling_the_llm():
    client = FakeClient({"CriticReview": [REJECT]})
    update = critic_agent.run(ready_state(), client)
    assert update["critic"].approved is True
    assert update["critic"].llm_reviewed is False
    assert client.calls == []
    assert update["final"].emotion == "anger"


def test_failed_response_rule_sends_back_once_without_llm():
    client = FakeClient({"CriticReview": [APPROVE]})
    state = ready_state(response={"explanation": "It is a post.", "suggested_reply": "ok"}, p_sarcastic=0.7)
    update = critic_agent.run(state, client)
    assert update["revision_count"] == 1
    assert "final" not in update
    assert not update["critic"].rule_checks["explanation_names_emotion"]
    assert client.calls == []   # rules failed, so no LLM review even though this is a review case


def test_second_failure_is_kept_not_sent_back_again():
    state = ready_state(response={"explanation": "It is a post.", "suggested_reply": "ok"})
    state["revision_count"] = 1
    update = critic_agent.run(state, NullClient())
    assert "revision_count" not in update
    assert update["final"].emotion == "anger"
    assert update["critic"].approved is False


def test_context_cue_not_in_post_is_reported_but_not_revised():
    state = ready_state(context=dict(GOOD_CONTEXT, is_sarcastic=False, cues=["battery died"]))
    update = critic_agent.run(state, NullClient())
    assert update["critic"].rule_checks["context_cues_in_post"] is False
    assert any("cues are not found" in issue for issue in update["critic"].issues)
    assert "final" in update   # a revision of the response cannot fix the context


# ---------- LLM review only for review cases ----------

@pytest.mark.parametrize("prediction_args, intent", [
    ({"p_sarcastic": 0.7}, None),                                   # sarcasm in the uncertain band
    ({"text_confidence": 0.3}, None),                               # weak text confidence
    ({}, dict(GOOD_INTENT, needs_support=True)),                    # writer may need support
])
def test_review_cases_call_the_llm_once(prediction_args, intent):
    client = FakeClient({"CriticReview": [APPROVE]})
    update = critic_agent.run(ready_state(intent=intent, **prediction_args), client)
    assert client.count("CriticReview") == 1
    assert update["critic"].llm_reviewed is True and update["critic"].approved is True


def test_sarcasm_disagreement_is_a_review_case():
    state = ready_state(context=dict(GOOD_CONTEXT, is_sarcastic=True), p_sarcastic=0.1)
    assert "context and classifier disagree on sarcasm" in critic_agent.review_reasons(state)


def test_llm_rejection_sends_back_with_its_issues():
    update = critic_agent.run(ready_state(p_sarcastic=0.7), FakeClient({"CriticReview": [REJECT]}))
    assert update["revision_count"] == 1
    assert "The reply is too casual." in update["critic"].issues


def test_no_llm_review_on_the_second_pass():
    client = FakeClient({"CriticReview": [REJECT]})
    state = ready_state(p_sarcastic=0.7)
    state["revision_count"] = 1
    update = critic_agent.run(state, client)
    assert client.calls == []
    assert update["final"] is not None


def test_unavailable_llm_review_falls_back_to_rules():
    update = critic_agent.run(ready_state(p_sarcastic=0.7), NullClient())
    assert update["critic"].approved is True and update["critic"].llm_reviewed is False


# ---------- emotion override stays disabled ----------

def test_eligible_alternative_is_reported_but_never_replaces_the_model_emotion():
    state = ready_state(context=dict(GOOD_CONTEXT, is_sarcastic=False, proposed_emotion="sadness"), p_sarcastic=0.7)
    update = critic_agent.run(state, FakeClient({"CriticReview": [APPROVE]}))
    assert update["critic"].override_eligible is True
    assert update["final"].alternative_emotion == "sadness"
    assert update["final"].emotion == "anger" and update["final"].emotion_source == "model"


def test_alternative_outside_top3_is_not_eligible():
    state = ready_state(context=dict(GOOD_CONTEXT, is_sarcastic=False, proposed_emotion="love"), p_sarcastic=0.7)
    assert critic_agent.override_eligible(state) is False


def test_zero_score_emotions_do_not_count_as_top3():
    # Emoji-only post: only joy has weight, so anger/fear are in top3 by tie order only.
    state = ready_state(context=dict(GOOD_CONTEXT, is_sarcastic=False, cues=["😂"], proposed_emotion="anger"),
                        scores={"joy": 1.0}, text_confidence=0.3, emojis=[("😂", "joy")])
    state["text"] = "😂😂"
    assert state["emotion"].top3[1] == "anger"
    assert critic_agent.override_eligible(state) is False


def test_alternative_is_not_eligible_when_classifier_is_confident():
    state = ready_state(context=dict(GOOD_CONTEXT, is_sarcastic=False, proposed_emotion="sadness"))
    assert critic_agent.override_eligible(state) is False
