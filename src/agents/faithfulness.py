"""Deterministic faithfulness checks on the Response Agent's explanation. No LLM call.

The system already knows the exact model outputs, so simple factual claims in the
explanation can be checked with code instead of asking the LLM:

1. percentages    every "N%" must match a model output (within 1 point) or appear in the post
2. sarcasm level  "high"/"low" sarcasm probability must agree with the real probability
3. emotion        "reads as X" must name RoBERTa's emotion
4. sentiment      "sentiment is X" must name the actual sentiment
5. quotes         quoted words must appear in the post

These checks only catch explicit claims written in these common English patterns; they do
not understand free paraphrase. They are written to avoid false alarms: apostrophes and
contractions are not quotes, label words may be quoted, and numbers from the post are allowed.
"""
from __future__ import annotations

import re
from typing import Dict, List

from agents.schemas import EmotionResult, SentimentResult

EMOTIONS = ("anger", "fear", "joy", "love", "neutral", "sadness", "surprise")
SENTIMENTS = ("negative", "neutral", "positive")
INTENTS = ("vent", "complain", "seek_help", "celebrate", "inform", "joke", "ask", "other", "unknown")
PERCENT_TOLERANCE = 1          # percentage points; explanations round to whole percentages
SARCASM_BAND = 0.5             # "high" needs at least 50%, "low" needs below 50%

CHECK_NAMES = ("explanation_percentages_match", "explanation_sarcasm_level_matches",
               "explanation_emotion_matches", "explanation_sentiment_matches", "explanation_quotes_in_post")

PERCENT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s?%")
NUMBER_WORDS = r"(?:probability|score|likelihood|chance)"
SARCASM_HIGH = re.compile(
    r"\b(?:high|strong|very high|significant)\b[\w\s-]{0,15}\bsarcas\w*\s+" + NUMBER_WORDS
    + r"|\bsarcas\w*\s+" + NUMBER_WORDS + r"\s+(?:is|was|of)\s+(?:high|strong|very high)", re.I)
SARCASM_LOW = re.compile(
    r"\b(?:low|very low|minimal|little)\b[\w\s-]{0,15}\bsarcas\w*\s+" + NUMBER_WORDS
    + r"|\bsarcas\w*\s+" + NUMBER_WORDS + r"\s+(?:is|was|of)\s+(?:low|very low)", re.I)
READ_AS = re.compile(
    r"\b(?:read|reads|classified|labelled|labeled|predicted|identified)\s+(?:(?:this|the)\s+(?:post|text|message)\s+)?"
    r"(?:as\s+)?(?:an?\s+)?(?:expressing\s+|showing\s+)?(?:an?\s+)?(?:emotion\s+of\s+)?['\"“‘]?(" + "|".join(EMOTIONS) + r")\b", re.I)
SENTIMENT_IS = re.compile(r"\bsentiment\s+(?:is|was|of)\s+(?:a\s+)?['\"“‘]?(" + "|".join(SENTIMENTS) + r")\b", re.I)
# "..." and “...” always quote; '...' and ‘...’ only at word boundaries, so dog's / writer's / don't are not quotes.
QUOTED = re.compile(r"[\"“]([^\"”]{2,})[\"”]|(?:(?<=[\s(])|^)['‘]([^'‘’]{2,}?)['’](?=[\s.,!?;:)]|$)")


def normalise(text: str) -> str:
    return " ".join(text.replace("’", "'").replace("‘", "'").lower().split())


def known_percentages(emotion: EmotionResult, sentiment: SentimentResult) -> List[float]:
    values = list(emotion.scores.values()) + list(sentiment.scores.values())
    values += [emotion.sarcasm_probability, emotion.text_confidence, emotion.margin]
    return [value * 100 for value in values]


def percentage_problems(text: str, emotion: EmotionResult, sentiment: SentimentResult, explanation: str) -> List[str]:
    in_post = {float(number) for number in PERCENT.findall(text)}
    known = known_percentages(emotion, sentiment)
    problems = []
    for number in PERCENT.findall(explanation):
        value = float(number)
        if value not in in_post and not any(abs(value - k) <= PERCENT_TOLERANCE for k in known):
            problems.append(f"The explanation gives {number}%, which matches no model output. Use the exact values: "
                            f"emotion {emotion.label} {emotion.scores[emotion.label]:.0%}, "
                            f"sarcasm probability {emotion.sarcasm_probability:.0%}.")
    return problems


def sarcasm_level_problems(emotion: EmotionResult, explanation: str) -> List[str]:
    p = emotion.sarcasm_probability
    if SARCASM_HIGH.search(explanation) and p < SARCASM_BAND:
        return [f"The explanation calls the sarcasm probability high, but it is {p:.0%}. Describe it as low "
                f"or state the exact value; any sarcasm reading comes from the Context Agent, not the classifier."]
    if SARCASM_LOW.search(explanation) and p >= SARCASM_BAND:
        return [f"The explanation calls the sarcasm probability low, but it is {p:.0%}. Describe it as high "
                f"or state the exact value."]
    return []


def emotion_problems(emotion: EmotionResult, explanation: str) -> List[str]:
    allowed = {emotion.label}
    if emotion.label in explanation.lower():
        # Mentioning the words-only reading is a fact as long as the final label is also named.
        allowed |= {emotion.emotion_from_text, emotion.emotion_before_sarcasm}
    return [f"The explanation says the post reads as {found.lower()}, but the model's emotion is {emotion.label}. "
            f"Name {emotion.label} as the reading."
            for found in READ_AS.findall(explanation) if found.lower() not in allowed]


def sentiment_problems(sentiment: SentimentResult, explanation: str) -> List[str]:
    return [f"The explanation says the sentiment is {found.lower()}, but it is {sentiment.label}."
            for found in SENTIMENT_IS.findall(explanation) if found.lower() != sentiment.label]


def quote_problems(text: str, explanation: str) -> List[str]:
    post = normalise(text)
    labels = set(EMOTIONS) | set(SENTIMENTS) | set(INTENTS)
    problems = []
    for double, single in QUOTED.findall(explanation):
        quote = normalise(double or single).strip(" .,!?;:")
        if quote and quote not in labels and quote not in post:
            problems.append(f"The explanation quotes '{(double or single).strip()}', which does not appear in the post. "
                            f"Quote only words that are in the post.")
    return problems


def check_explanation(text: str, emotion: EmotionResult, sentiment: SentimentResult, explanation: str) -> Dict[str, List[str]]:
    """Problems found, by check name. An empty list means the check passed."""
    return {
        "explanation_percentages_match": percentage_problems(text, emotion, sentiment, explanation),
        "explanation_sarcasm_level_matches": sarcasm_level_problems(emotion, explanation),
        "explanation_emotion_matches": emotion_problems(emotion, explanation),
        "explanation_sentiment_matches": sentiment_problems(sentiment, explanation),
        "explanation_quotes_in_post": quote_problems(text, explanation),
    }
