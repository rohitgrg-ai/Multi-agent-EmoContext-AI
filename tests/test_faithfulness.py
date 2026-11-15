"""Deterministic faithfulness checks (agents/faithfulness.py) and how the Critic uses them."""
import pytest
from conftest import GOOD_CONTEXT, GOOD_INTENT, FakeClient, build_state

from agents import critic_agent, response_agent
from agents.faithfulness import check_explanation
from agents.llm import NullClient
from agents.schemas import ContextReport, IntentReport, ResponseDraft


def problems(state, explanation):
    found = check_explanation(state["text"], state["emotion"], state["sentiment"], explanation)
    return [message for messages in found.values() for message in messages]


PHONE = dict(text="I just love it when my phone dies at 2%", scores={"anger": 0.92, "sadness": 0.05, "joy": 0.03}, p_sarcastic=0.99)
MEETING = dict(text="Oh great, the meeting got moved to Friday evening",
               scores={"love": 0.56, "joy": 0.2, "anger": 0.12, "sadness": 0.12}, p_sarcastic=0.31, text_confidence=0.56)
JOB = dict(text="I got the job!!!", scores={"joy": 0.97, "love": 0.03}, p_sarcastic=0.02)
DOG = dict(text="My dog passed away this morning", scores={"sadness": 0.98, "fear": 0.02}, p_sarcastic=0.02)
EMAIL = dict(text="Thanks so much for replying to my email after three weeks, really appreciate it",
             scores={"love": 0.83, "anger": 0.12, "joy": 0.05}, p_sarcastic=0.75)


# ---------- the seven hand-written cases from the investigation ----------

@pytest.mark.parametrize("post, explanation, expected", [
    (MEETING, "The post was read with an emotion of love. The high sarcasm probability indicates the writer is ironic.",
     ["calls the sarcasm probability high, but it is 31%"]),
    (PHONE, "The model reads this post as anger (92%). This looks sarcastic (99% likely). The battery was at 2%.", []),
    (PHONE, "The model reads this post as joy, and the sarcasm probability is low.",
     ["calls the sarcasm probability low, but it is 99%", "reads as joy, but the model's emotion is anger"]),
    (JOB, "The writer said 'best day ever' which shows joy.", ["quotes 'best day ever'"]),
    (JOB, "The writer's news is great; the post is read as joy with 85% confidence.", ["gives 85%"]),
    (DOG, "The post is read as expressing sadness, as the writer is sharing the news of their dog's passing. "
          "The writer's intent is to vent.", []),
    (EMAIL, "The post is read as expressing 'love,' but the high sarcasm probability suggests sarcasm about the "
            "'three weeks' delay.", []),
])
def test_handwritten_cases(post, explanation, expected):
    found = problems(build_state(**post), explanation)
    assert len(found) == len(expected), found
    for fragment in expected:
        assert any(fragment in message for message in found), (fragment, found)


# ---------- avoiding false positives ----------

@pytest.mark.parametrize("explanation", [
    "The writer's dog didn't survive and they're heartbroken; this reads as sadness.",   # apostrophes, contractions
    "The writer’s grief is clear; the post reads as sadness.",                            # curly apostrophe
    "The phrase \"passed away this morning\" shows sadness.",                              # quote from the post
    "The word 'Dog' appears and the emotion is 'sadness'.",                               # case-insensitive, label quote
    "Intent looks like 'vent', and the sentiment is negative.",                           # intent label quoted
    "The model is 98% sure the post shows sadness.",                                      # exact model value
    "Sadness scores 97%, close to certain.",                                              # within 1 point of 98%
])
def test_no_false_alarm(explanation):
    assert problems(build_state(**DOG), explanation) == []


def test_percentage_from_the_post_is_allowed():
    assert problems(build_state(**PHONE), "The phone died at 2%, and the post reads as anger (92%).") == []


def test_words_only_reading_may_be_mentioned_alongside_final_label():
    state = build_state(text="Wow, I just LOVE waiting two hours for a bus", scores={"anger": 0.8, "sadness": 0.2},
                        before_sarcasm="joy", p_sarcastic=0.99)
    assert problems(state, "The words read as joy, but sarcasm moved the reading to anger.") == []
    assert problems(state, "The post reads as joy.")   # without the final label it is a contradiction


def test_wrong_sentiment_claim():
    assert any("sentiment is positive" in m for m in problems(build_state(**DOG), "The sentiment is positive."))


@pytest.mark.parametrize("post", [PHONE, MEETING, JOB, DOG, EMAIL,
                                  dict(text="so fun 🙄🙄", scores={"anger": 0.9, "joy": 0.1}, p_sarcastic=0.95,
                                       emojis=[("🙄", "anger")] * 2),
                                  dict(text="Wow, I just LOVE waiting", scores={"anger": 0.8, "sadness": 0.2},
                                       before_sarcasm="joy", p_sarcastic=0.99),
                                  dict(text="hmm", scores={"neutral": 0.4, "joy": 0.35, "sadness": 0.25}, text_confidence=0.4)])
def test_rule_based_fallback_explanations_always_pass(post):
    state = build_state(**post)
    state["context"] = ContextReport(implied_meaning="x", is_sarcastic=False, cues=[], reason="x", source="fallback")
    state["intent"] = IntentReport(intent="unknown", urgency="low", needs_support=False, reason="x", source="fallback")
    draft = response_agent.fallback_draft(state)
    assert problems(state, draft.explanation) == [], draft.explanation


# ---------- the Critic uses the checks ----------

def critic_state(explanation, **post):
    state = build_state(**(post or MEETING))
    state["context"] = ContextReport(**dict(GOOD_CONTEXT, is_sarcastic=False, cues=["Oh great"]), source="llm")
    state["intent"] = IntentReport(**GOOD_INTENT, source="llm")
    state["response"] = ResponseDraft(explanation=explanation, suggested_reply="That sounds annoying.", source="llm")
    return state


def test_critic_sends_back_wrong_number_without_an_llm_call():
    client = FakeClient({"CriticReview": [{"approved": True, "issues": []}]})
    state = critic_state("The post reads as love. The high sarcasm probability shows irony.")
    update = critic_agent.run(state, client)
    assert update["revision_count"] == 1 and "final" not in update
    assert update["critic"].rule_checks["explanation_sarcasm_level_matches"] is False
    assert any("it is 31%" in issue and "Describe it as low" in issue for issue in update["critic"].issues)
    assert client.calls == []   # a factual error is caught by rules, not by the LLM


def test_critic_keeps_one_revision_limit_for_faithfulness_failures():
    state = critic_state("The post reads as love. The high sarcasm probability shows irony.")
    state["revision_count"] = 1
    update = critic_agent.run(state, NullClient())
    assert "revision_count" not in update
    assert update["critic"].approved is False
    assert update["final"].emotion == "love"   # RoBERTa's label stays final


def test_faithful_explanation_passes_all_new_checks():
    state = critic_state("The post reads as love (56%), though the sarcasm probability is low (31%).")
    update = critic_agent.run(state, NullClient())
    for name in ("explanation_percentages_match", "explanation_sarcasm_level_matches", "explanation_emotion_matches",
                 "explanation_sentiment_matches", "explanation_quotes_in_post"):
        assert update["critic"].rule_checks[name] is True
