"""Revision history (P2): every Response and Critic pass is kept, in order, without duplicates."""
import json

from conftest import GOOD_CONTEXT, GOOD_INTENT, GOOD_RESPONSE, FakeClient, FakePredictor, build_prediction

from agents.graph import analyse, build_graph, state_to_dict
from agents.llm import NullClient

POST = "I just love it when my phone dies at 2%"
CLEAR = build_prediction({"anger": 0.8, "sadness": 0.15, "joy": 0.05}, before_sarcasm="joy", p_sarcastic=0.99)
UNCERTAIN = build_prediction({"anger": 0.6, "sadness": 0.4}, p_sarcastic=0.7)        # a Critic review case
REVISED = {"explanation": "The post reads as anger; the praise is clearly not meant.",
           "suggested_reply": "That sounds frustrating. Hope it charges fast!"}
NO_EMOTION_NAMED = {"explanation": "This is a post about a phone.", "suggested_reply": "Oh no."}
REJECT = {"approved": False, "issues": ["The reply is too casual."]}


def run(prediction, replies):
    client = FakeClient(replies)
    return analyse(build_graph(FakePredictor(prediction), client), POST), client


def assert_no_duplicates(state):
    """Exactly one history entry per Response / Critic step in the trace."""
    agents = [step.agent for step in state["trace"]]
    assert len(state["response_history"]) == agents.count("response")
    assert len(state["critic_history"]) == agents.count("critic")


def test_one_pass_records_exactly_one_entry_each():
    state, _ = run(CLEAR, {"ContextReading": [GOOD_CONTEXT], "IntentReading": [GOOD_INTENT], "ResponseText": [GOOD_RESPONSE]})
    assert state["revision_count"] == 0
    assert state["response_history"] == [state["response"]]
    assert state["critic_history"] == [state["critic"]]
    assert state["critic_history"][0].approved is True
    assert_no_duplicates(state)


def test_null_client_one_pass():
    state = analyse(build_graph(FakePredictor(CLEAR), NullClient()), POST)
    assert len(state["response_history"]) == len(state["critic_history"]) == 1
    assert state["response_history"][0].source == "fallback"


def test_rejection_then_successful_revision():
    state, client = run(UNCERTAIN, {"ContextReading": [dict(GOOD_CONTEXT, is_sarcastic=False)], "IntentReading": [GOOD_INTENT],
                                    "ResponseText": [GOOD_RESPONSE, REVISED], "CriticReview": [REJECT]})
    first_draft, revised_draft = state["response_history"]
    first_review, second_review = state["critic_history"]
    # Pass 1: initial draft rejected by the LLM review, with its reasons kept
    assert first_draft.explanation == GOOD_RESPONSE["explanation"]
    assert first_review.approved is False and first_review.llm_reviewed is True
    assert first_review.issues == ["The reply is too casual."]
    # The feedback the Response Agent received is exactly the first review's issues
    revision_prompt = [user for name, user in client.calls if name == "ResponseText"][1]
    assert "The reply is too casual." in revision_prompt
    # Pass 2: revised draft approved by rules (no LLM review on the second pass)
    assert revised_draft.explanation == REVISED["explanation"]
    assert second_review.approved is True and second_review.llm_reviewed is False
    # The latest-pass fields and the final answer come from pass 2
    assert state["response"] == revised_draft and state["critic"] == second_review
    assert state["final"].explanation == REVISED["explanation"]
    assert state["final"].emotion == "anger"
    assert_no_duplicates(state)


def test_rejection_twice_keeps_both_failed_reviews():
    state, _ = run(UNCERTAIN, {"ContextReading": [dict(GOOD_CONTEXT, is_sarcastic=False)], "IntentReading": [GOOD_INTENT],
                               "ResponseText": [NO_EMOTION_NAMED]})
    first_review, second_review = state["critic_history"]
    assert first_review.approved is False and second_review.approved is False
    assert first_review.rule_checks["explanation_names_emotion"] is False
    assert any("must name the emotion 'anger'" in issue for issue in first_review.issues)
    assert state["revision_count"] == 1 and "final" in state       # the one-revision limit still holds
    assert state["final"].emotion == "anger"
    assert_no_duplicates(state)


def test_history_is_chronological():
    state, _ = run(UNCERTAIN, {"ContextReading": [dict(GOOD_CONTEXT, is_sarcastic=False)], "IntentReading": [GOOD_INTENT],
                               "ResponseText": [GOOD_RESPONSE, REVISED], "CriticReview": [REJECT]})
    agents = [step.agent for step in state["trace"]]
    assert agents[-4:] == ["response", "critic", "response", "critic"]
    summaries = [step.summary for step in state["trace"] if step.agent == "critic"]
    assert summaries[0].startswith("sent back") and not state["critic_history"][0].approved
    assert summaries[1].startswith("approved") and state["critic_history"][1].approved


def test_faithfulness_rejection_is_recorded_in_history():
    meeting = build_prediction({"love": 0.56, "joy": 0.2, "anger": 0.12, "sadness": 0.12}, p_sarcastic=0.31, text_confidence=0.56)
    wrong = {"explanation": "The post reads as love. The high sarcasm probability shows irony.", "suggested_reply": "Ugh."}
    fixed = {"explanation": "The post reads as love, with a low sarcasm probability (31%).", "suggested_reply": "Ugh."}
    state, client = run(meeting, {"ContextReading": [GOOD_CONTEXT], "IntentReading": [GOOD_INTENT],
                                  "ResponseText": [wrong, fixed]})
    first_review = state["critic_history"][0]
    assert first_review.rule_checks["explanation_sarcasm_level_matches"] is False
    assert any("it is 31%" in issue for issue in first_review.issues)
    assert client.count("CriticReview") == 0
    assert state["critic_history"][1].approved is True


def test_history_serialises_to_json():
    state, _ = run(UNCERTAIN, {"ContextReading": [dict(GOOD_CONTEXT, is_sarcastic=False)], "IntentReading": [GOOD_INTENT],
                               "ResponseText": [GOOD_RESPONSE, REVISED], "CriticReview": [REJECT]})
    data = json.loads(json.dumps(state_to_dict(state)))
    assert [d["explanation"] for d in data["response_history"]] == [GOOD_RESPONSE["explanation"], REVISED["explanation"]]
    assert [c["approved"] for c in data["critic_history"]] == [False, True]
