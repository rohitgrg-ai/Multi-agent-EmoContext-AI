"""FastAPI backend: three endpoints over the trained model and the six-agent workflow.

    uvicorn --app-dir src api.main:app --port 8000

    GET  /health           is the model loaded, and is the LLM available?
    POST /analyze          Emotion + Sentiment agents only (trained models, fast, no LLM)
    POST /agents/analyze   all six agents through the LangGraph workflow

The model and the compiled graph are loaded once when the server starts (FastAPI
"lifespan"), not on every request. If the model files are missing the server still
starts, `/health` reports it, and the analysis endpoints return 503.

Ollama is optional: `get_llm_client()` falls back to NullClient, and the LLM agents then
use their deterministic fallbacks. The LLM status is checked once at start-up, so start
Ollama before the API (or restart the API) for it to be used.

Privacy: the submitted text is never written to the logs, only error types.
The endpoints are plain `def` functions, so FastAPI runs them in its thread pool and the
slow model call does not block the server's event loop.
"""
from __future__ import annotations

import logging
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Callable, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, StringConstraints

from agents import emotion_agent, sentiment_agent
from agents.graph import analyse, build_graph
from agents.llm import NullClient, get_llm_client
from agents.schemas import (ContextReport, CriticReport, EmotionResult, FinalResult, IntentReport, ResponseDraft,
                            SentimentResult, TraceStep)

MAX_TEXT_CHARS = 2000
logger = logging.getLogger("emocontext.api")


# ---------- request and response bodies ----------

class AnalyzeRequest(BaseModel):
    # Surrounding spaces are stripped first, so "   " counts as empty and is rejected.
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_CHARS)]


class AnalyzeResponse(BaseModel):
    emotion: EmotionResult
    sentiment: SentimentResult
    errors: List[str]


class AgentAnalyzeResponse(BaseModel):
    final: FinalResult
    emotion: EmotionResult
    sentiment: SentimentResult
    context: ContextReport
    intent: IntentReport
    response: ResponseDraft
    critic: CriticReport
    revision_count: int
    trace: List[TraceStep]
    errors: List[str]
    # Added in P2 (additive): every Response and Critic pass in order. `response` and `critic`
    # above are unchanged and still equal the last entry of these lists.
    response_history: List[ResponseDraft] = []
    critic_history: List[CriticReport] = []


class LLMStatus(BaseModel):
    provider: str              # "ollama" or "none"
    model: Optional[str]
    available: bool
    reason: str                # why the LLM is off, if it is


class HealthResponse(BaseModel):
    status: str                # "ok", or "degraded" when the model is not loaded
    model_loaded: bool
    model_error: Optional[str]
    llm: LLMStatus


# ---------- loaded services ----------

@dataclass
class Services:
    predictor: object = None
    graph: object = None
    llm: object = None
    emotion_baseline: Optional[Callable[[str], str]] = None
    sentiment_baseline: Optional[Callable[[str], str]] = None
    model_error: Optional[str] = None


def load_services() -> Services:
    from baseline import load_keyword_model, load_sentiment_baseline
    from model import Predictor

    with warnings.catch_warnings():
        # The baseline .joblib files were saved with a newer scikit-learn; outputs are unchanged.
        warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
        emotion_baseline, sentiment_baseline = load_keyword_model(), load_sentiment_baseline()
    llm = get_llm_client()
    try:
        predictor = Predictor()
    except OSError as error:   # model files missing or unreadable
        logger.error("trained model could not be loaded: %s", error.__class__.__name__)
        return Services(llm=llm, model_error="trained model not found in models/emotion_sarcasm")
    graph = build_graph(predictor, llm, emotion_baseline, sentiment_baseline)
    return Services(predictor, graph, llm, emotion_baseline, sentiment_baseline)


def create_app(services: Optional[Services] = None) -> FastAPI:
    """Pass `services` to use ready-made (e.g. fake) ones; otherwise the real ones load at start-up."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services if services is not None else load_services()
        yield

    app = FastAPI(title="EmoContext AI", version="1.0", lifespan=lifespan)

    def loaded(request: Request) -> Services:
        current = request.app.state.services
        if current.predictor is None:
            raise HTTPException(status_code=503, detail=current.model_error or "trained model not loaded")
        return current

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        current = request.app.state.services
        llm = current.llm or NullClient("LLM not configured")
        return HealthResponse(
            status="ok" if current.predictor is not None else "degraded",
            model_loaded=current.predictor is not None,
            model_error=current.model_error,
            llm=LLMStatus(provider=llm.provider, model=llm.model, available=llm.provider != "none",
                          reason=llm.reason),
        )

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze(body: AnalyzeRequest, request: Request) -> AnalyzeResponse:
        current = loaded(request)
        try:
            state = {"text": body.text}
            state.update(emotion_agent.run(state, current.predictor, current.emotion_baseline))
            sentiment = sentiment_agent.run(state, current.sentiment_baseline)
            return AnalyzeResponse(emotion=state["emotion"], sentiment=sentiment["sentiment"],
                                   errors=state["errors"] + sentiment["errors"])
        except Exception as error:
            logger.error("/analyze failed: %s", error.__class__.__name__)   # type only, never the text
            raise HTTPException(status_code=500, detail="analysis failed") from None

    @app.post("/agents/analyze", response_model=AgentAnalyzeResponse)
    def agents_analyze(body: AnalyzeRequest, request: Request) -> AgentAnalyzeResponse:
        current = loaded(request)
        try:
            state = analyse(current.graph, body.text)
            return AgentAnalyzeResponse(**{field: state[field] for field in AgentAnalyzeResponse.model_fields
                                           if field in state})
        except Exception as error:
            logger.error("/agents/analyze failed: %s", error.__class__.__name__)
            raise HTTPException(status_code=500, detail="agent analysis failed") from None

    return app


app = create_app()
