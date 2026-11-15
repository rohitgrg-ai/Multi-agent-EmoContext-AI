"""The single shared state that every agent reads from and writes to.

Agents never call each other. Each agent is a function that reads what it needs from
this state and returns a dict with only the keys it owns; LangGraph merges that dict back.

`trace` and `errors` use `operator.add` so each agent's entries are appended instead of
replacing the previous ones.
"""
from __future__ import annotations

import operator
from typing import Annotated, List, TypedDict

from agents.schemas import (ContextReport, CriticReport, EmotionResult, FinalResult, IntentReport,
                            ResponseDraft, SentimentResult, TraceStep)


class AgentState(TypedDict, total=False):
    text: str                          # input post, set once at the start
    emotion: EmotionResult             # written by the Emotion Agent
    sentiment: SentimentResult         # written by the Sentiment Agent
    context: ContextReport             # written by the Context Agent
    intent: IntentReport               # written by the Intent Agent
    response: ResponseDraft            # written by the Response Agent
    critic: CriticReport               # written by the Critic Agent
    final: FinalResult                 # written by the Critic Agent
    revision_count: int                # how many times the Critic sent the response back (max 1)
    trace: Annotated[List[TraceStep], operator.add]
    errors: Annotated[List[str], operator.add]
    # Every pass is kept, in order. `response` and `critic` above hold only the latest pass;
    # these lists also keep the first draft and the first review when a revision happens.
    # The revision feedback is critic_history[0].issues: that is what the Response Agent was given.
    response_history: Annotated[List[ResponseDraft], operator.add]   # one entry per Response Agent run
    critic_history: Annotated[List[CriticReport], operator.add]      # one entry per Critic Agent run
