"""HTTP client and display logic for the dashboard. No Streamlit and no ML code here.

The dashboard only talks to the FastAPI backend over HTTP. It never loads RoBERTa, never
imports the agents and never runs LangGraph; tests/test_dashboard.py checks those imports.
Everything shown comes from the API's JSON. These helpers only decide how to label it.

Set EMOCONTEXT_API_URL to point at another server (default http://127.0.0.1:8000).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

API_URL = os.getenv("EMOCONTEXT_API_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_TEXT_CHARS = 2000            # same limit the API enforces
HEALTH_TIMEOUT_SECONDS = 3.0
# Measured with real Gemma: median 12 s, slowest 19 s per post. The extra margin covers
# the first request after start-up, when Ollama still has to load the model.
ANALYSIS_TIMEOUT_SECONDS = 180.0
START_API_COMMAND = "uvicorn --app-dir src api.main:app --port 8000"
LLM_AGENTS = ("context", "intent", "response")
AGENT_TITLES = {"emotion": "Emotion Agent", "sentiment": "Sentiment Agent", "context": "Context Agent",
                "intent": "Intent Agent", "response": "Response Agent", "critic": "Critic Agent"}


@dataclass
class SystemStatus:
    api_online: bool
    model_loaded: bool
    llm_active: bool                  # Ollama was reachable when the API started
    llm_label: str                    # e.g. "ollama · gemma4:latest" or "Not available"
    overall: str                      # "Online", "Degraded", "LLM unavailable" or "Offline"
    detail: str                       # one sentence for the viewer


# ---------- talking to the API ----------

def fetch_health() -> Optional[Dict[str, Any]]:
    """The /health JSON, or None if the API cannot be reached."""
    try:
        response = httpx.get(f"{API_URL}/health", timeout=HEALTH_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        return None


def analyse_post(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """POST the text to /agents/analyze. Returns (result, None) or (None, message for the viewer)."""
    try:
        response = httpx.post(f"{API_URL}/agents/analyze", json={"text": text}, timeout=ANALYSIS_TIMEOUT_SECONDS)
    except httpx.TimeoutException:
        return None, (f"The analysis took longer than {ANALYSIS_TIMEOUT_SECONDS:.0f} seconds and was stopped. "
                      "The LLM may still be loading; try again in a moment.")
    except httpx.HTTPError:
        return None, f"Cannot reach the EmoContext API at {API_URL}. Start it with: {START_API_COMMAND}"
    return read_response(response)


def read_response(response: httpx.Response) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if response.status_code == 200:
        try:
            return response.json(), None
        except ValueError:
            return None, "The API returned a response that could not be read."
    messages = {
        422: f"The post was rejected. It must contain between 1 and {MAX_TEXT_CHARS} characters.",
        503: "The API is running but the trained model is not loaded, so posts cannot be analysed.",
        500: "The API hit an internal error while analysing this post. Please try again.",
    }
    # Only fixed messages are shown: server error bodies are never displayed.
    return None, messages.get(response.status_code, f"The API answered with an unexpected status ({response.status_code}).")


def validate_text(text: str) -> Optional[str]:
    """A message if the text cannot be sent, otherwise None (the API checks again)."""
    stripped = text.strip()
    if not stripped:
        return "Type or pick a post first."
    if len(stripped) > MAX_TEXT_CHARS:
        return f"The post is {len(stripped)} characters long; the limit is {MAX_TEXT_CHARS}."
    return None


# ---------- turning API data into labels ----------

def describe_status(health: Optional[Dict[str, Any]]) -> SystemStatus:
    if health is None:
        return SystemStatus(False, False, False, "Unknown", "Offline",
                            f"The API at {API_URL} is not responding.")
    llm = health.get("llm") or {}
    model_loaded = bool(health.get("model_loaded"))
    llm_active = bool(llm.get("available")) and llm.get("provider") not in (None, "none")
    llm_label = f"{llm.get('provider')} · {llm.get('model')}" if llm_active else "Not available"
    if not model_loaded:
        return SystemStatus(True, False, llm_active, llm_label, "Degraded",
                            health.get("model_error") or "The trained model is not loaded on the server.")
    if not llm_active:
        reason = llm.get("reason") or "no LLM configured"
        return SystemStatus(True, True, False, llm_label, "LLM unavailable",
                            f"Running in fallback mode: the LLM agents use rule-based fallbacks ({reason}).")
    return SystemStatus(True, True, True, llm_label, "Online",
                        "All six agents are available. The LLM status was checked just now.")


def agent_mode(agent: str, used_llm: bool, source: Optional[str] = None) -> str:
    """Honest label for how an agent produced its output."""
    if agent in ("emotion", "sentiment"):
        return "Trained model"
    if agent == "critic":
        return "Rules + LLM review" if used_llm else "Rules only"
    if used_llm or source == "llm":
        return "LLM"
    return "Rule-based fallback"


def source_label(source: Optional[str], llm_name: str) -> str:
    if source == "llm":
        return f"Generated by the LLM ({llm_name})"
    if source == "fallback":
        return "Rule-based fallback (no LLM output)"
    return "Source not reported"


def fallback_agents(result: Dict[str, Any]) -> List[str]:
    """LLM agents whose output in this result came from the fallback."""
    return [name for name in LLM_AGENTS if (result.get(name) or {}).get("source") == "fallback"]


def top_scores(scores: Optional[Dict[str, float]], limit: int = 7) -> List[Tuple[str, float]]:
    return sorted((scores or {}).items(), key=lambda item: item[1], reverse=True)[:limit]


def revision_passes(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pair each Response draft with the Critic review of that draft, in order.

    Pass N's review is the feedback the Response Agent received before pass N+1. Returns an
    empty list when the API sent no history (an older backend).
    """
    drafts, reviews = result.get("response_history") or [], result.get("critic_history") or []
    passes = []
    for index, (draft, review) in enumerate(zip(drafts, reviews)):
        passes.append({
            "number": index + 1,
            "draft": draft,
            "review": review,
            "sent_back": index < len(reviews) - 1,          # another pass followed this review
            "is_final": index == len(reviews) - 1,
            "failed_checks": [name for name, passed in (review.get("rule_checks") or {}).items() if not passed],
        })
    return passes


def agent_output(result: Dict[str, Any], agent: str) -> Dict[str, Any]:
    """The part of the API response that belongs to one agent."""
    if agent == "critic":
        return {**(result.get("critic") or {}), "revision_count": result.get("revision_count", 0),
                "critic_history": result.get("critic_history", [])}
    if agent == "response":
        return {**(result.get("response") or {}), "response_history": result.get("response_history", [])}
    return result.get(agent) or {}
