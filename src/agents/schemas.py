"""Structured outputs that the six agents write into the shared state.

This file is the contract between agents, the API and the dashboard. It is frozen:
change it only by agreement, because every agent and test is built against it.

Two kinds of model live here:
- "Reading" models are what the LLM is asked to return. Their JSON schema is sent to
  Ollama, so they contain only fields the LLM can fill in.
- "Report"/"Result" models are what the agents store. LLM reports add a `source` field
  saying whether the content came from the LLM or from the deterministic fallback.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# Must match model.EMOTIONS and model.EMOTION_SENTIMENT. A test checks this.
EmotionLabel = Literal["anger", "fear", "joy", "love", "neutral", "sadness", "surprise"]
SentimentLabel = Literal["negative", "neutral", "positive"]
IntentLabel = Literal["vent", "complain", "seek_help", "celebrate", "inform", "joke", "ask", "other", "unknown"]
Urgency = Literal["low", "medium", "high"]
Source = Literal["llm", "fallback"]
AgentName = Literal["emotion", "sentiment", "context", "intent", "response", "critic"]


# ---------- Machine-learning agents (no LLM) ----------

class EmojiVote(BaseModel):
    char: str
    emotion: EmotionLabel


class EmotionResult(BaseModel):
    """Output of the Emotion Agent: the trained RoBERTa pipeline, unchanged."""
    label: EmotionLabel
    scores: Dict[str, float]               # final scores after emoji vote and sarcasm rule; sum to 1
    top3: List[EmotionLabel]
    margin: float                          # top score minus second score
    text_confidence: float                 # top score from the text alone, before emojis
    emotion_from_text: EmotionLabel
    emotion_before_sarcasm: EmotionLabel
    sarcasm_probability: float
    sarcasm_flagged: bool                  # sarcasm_probability >= model.SARCASM_THRESHOLD
    sarcasm_rule_applied: bool             # the sarcasm rule changed the label
    emojis_used: List[EmojiVote]
    baseline_label: Optional[str] = None   # TF-IDF + SVM emotion, for comparison only


class SentimentResult(BaseModel):
    """Output of the Sentiment Agent: sentiment follows the emotion, as in the original app."""
    label: SentimentLabel
    scores: Dict[str, float]               # emotion scores summed per sentiment
    consistent_with_scores: bool           # False when the label is not the highest summed score
    baseline_label: Optional[SentimentLabel] = None   # TF-IDF + SVM sentiment, for comparison only


# ---------- LLM agents: what the LLM returns ----------

class ContextReading(BaseModel):
    implied_meaning: str = Field(description="What the writer most likely means, in one sentence.")
    is_sarcastic: bool
    cues: List[str] = Field(description="Exact words or emojis copied from the post that support the reading.")
    proposed_emotion: Optional[EmotionLabel] = Field(
        default=None, description="A different emotion only if the classifier's label seems wrong, else null.")
    reason: str


class IntentReading(BaseModel):
    intent: IntentLabel
    urgency: Urgency
    needs_support: bool = Field(description="True if the writer seems distressed and may need support.")
    reason: str


class ResponseText(BaseModel):
    explanation: str = Field(description="Plain-language explanation of how the post was read.")
    suggested_reply: str = Field(description="A short, kind reply someone could send to the writer.")


class CriticReview(BaseModel):
    approved: bool
    issues: List[str]


# ---------- LLM agents: what is stored in the state ----------

class ContextReport(ContextReading):
    source: Source


class IntentReport(IntentReading):
    source: Source


class ResponseDraft(ResponseText):
    source: Source


class CriticReport(BaseModel):
    approved: bool
    issues: List[str]
    rule_checks: Dict[str, bool]
    llm_reviewed: bool
    override_eligible: bool                # the Context alternative passed every override condition
    alternative_emotion: Optional[EmotionLabel] = None


# ---------- Final answer and trace ----------

class FinalResult(BaseModel):
    emotion: EmotionLabel
    sentiment: SentimentLabel
    emotion_source: Literal["model", "context_override"]   # "model" while overrides are disabled
    alternative_emotion: Optional[EmotionLabel] = None    # shown separately, never replaces the model label
    intent: IntentLabel
    urgency: Urgency
    needs_support: bool
    explanation: str
    suggested_reply: str


class TraceStep(BaseModel):
    agent: AgentName
    used_llm: bool
    duration_ms: float
    summary: str
