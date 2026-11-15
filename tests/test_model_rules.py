"""Unit tests for the original prediction rules in src/model.py, including the B1 fix.

These test pure functions with hand-made score arrays, so the model is never loaded.
"""
import numpy as np
import pytest

from model import (EMOJI_WEIGHT, EMOTION_ID, EMOTION_SENTIMENT, EMOTIONS, SARCASM_THRESHOLD,
                   apply_sarcasm_rule, blend_emojis, normalise)


def scores_with(**values):
    """A 7-emotion score array from keyword values, e.g. scores_with(joy=0.6, anger=0.4)."""
    scores = np.zeros(len(EMOTIONS))
    for name, value in values.items():
        scores[EMOTION_ID[name]] = value
    return scores


# ---------- normalise ----------

def test_normalise_replaces_links_mentions_and_extra_spaces():
    assert normalise("hey  @sam look https://x.com/a  now") == "hey @user look http now"


# ---------- blend_emojis ----------

def test_blend_without_emojis_returns_scores_unchanged():
    scores = scores_with(anger=0.7, sadness=0.3)
    assert blend_emojis("no emojis here", scores) is scores


def test_blend_with_words_and_emoji_uses_emoji_weight():
    scores = scores_with(anger=1.0)
    blended = blend_emojis("great day 😂", scores)
    assert blended[EMOTION_ID["joy"]] == pytest.approx(EMOJI_WEIGHT)
    assert blended[EMOTION_ID["anger"]] == pytest.approx(1 - EMOJI_WEIGHT)


def test_blend_emoji_only_post_is_decided_by_emojis_alone():
    blended = blend_emojis("😢😢", scores_with(joy=1.0))
    assert blended[EMOTION_ID["sadness"]] == pytest.approx(1.0)
    assert blended[EMOTION_ID["joy"]] == pytest.approx(0.0)


# ---------- apply_sarcasm_rule ----------

def test_sarcasm_rule_switches_joy_when_sarcasm_is_at_threshold():
    adjusted = apply_sarcasm_rule(scores_with(joy=0.6, anger=0.3, neutral=0.1), SARCASM_THRESHOLD)
    assert EMOTIONS[int(np.argmax(adjusted))] == "anger"
    assert adjusted[EMOTION_ID["joy"]] == adjusted[EMOTION_ID["love"]] == adjusted[EMOTION_ID["neutral"]] == 0
    assert adjusted.sum() == pytest.approx(1.0)


def test_sarcasm_rule_does_nothing_just_below_threshold():
    scores = scores_with(joy=0.6, anger=0.4)
    assert apply_sarcasm_rule(scores, SARCASM_THRESHOLD - 0.0001) is scores


def test_sarcasm_rule_ignores_non_positive_top_emotion():
    scores = scores_with(sadness=0.6, joy=0.4)
    assert apply_sarcasm_rule(scores, 0.99) is scores


def test_b1_emoji_only_joy_post_with_high_sarcasm_does_not_produce_nan():
    """B1: before the fix this divided by zero, gave NaN, and argmax silently returned 'anger'."""
    vote_only = blend_emojis("😂😍", scores_with(anger=1.0))   # emoji-only: weight 1.0, joy+love only
    adjusted = apply_sarcasm_rule(vote_only, 0.95)
    assert not np.isnan(adjusted).any()
    assert EMOTIONS[int(np.argmax(adjusted))] in ("joy", "love")
    assert adjusted.sum() == pytest.approx(1.0)


def test_b1_guard_returns_original_scores_unchanged():
    scores = scores_with(joy=0.5, love=0.5)
    assert apply_sarcasm_rule(scores, 0.99) is scores


# ---------- label mapping ----------

def test_every_emotion_has_a_sentiment():
    assert set(EMOTION_SENTIMENT) == set(EMOTIONS)
    assert set(EMOTION_SENTIMENT.values()) == {"negative", "neutral", "positive"}
