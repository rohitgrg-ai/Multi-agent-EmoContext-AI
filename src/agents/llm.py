"""One shared LLM client for every agent that needs one.

All LLM agents use the same local model (Ollama, `gemma4:latest` by default). They differ
only in their prompt and in the pydantic schema they ask for. Ollama's "structured
outputs" feature receives the schema in the `format` field and constrains the reply to
JSON of that shape; the reply is still validated with pydantic, because a model can
return truncated or wrong JSON.

`ask_json` never raises. It returns a validated object, or None. Agents treat None as
"the LLM is not available" and use their deterministic fallback, so the pipeline always
finishes even when Ollama is not installed or not running.

Configuration (environment variables):
    EMOCONTEXT_LLM   "ollama" (default) or "none" to disable the LLM completely
    OLLAMA_URL       default http://localhost:11434
    OLLAMA_MODEL     default gemma4:latest
    OLLAMA_TIMEOUT   seconds per request, default 120 (a 9.6 GB model can be slow to load)
"""
from __future__ import annotations

import os
from typing import Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:latest"
DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_REPLY_TOKENS = 400

Schema = TypeVar("Schema", bound=BaseModel)


class NullClient:
    """Used when the LLM is disabled or unreachable. Every request returns None."""
    provider = "none"

    def __init__(self, reason: str = "LLM disabled"):
        self.model = None
        self.reason = reason
        self.last_error: Optional[str] = None

    def ask_json(self, system: str, user: str, schema: Type[Schema]) -> Optional[Schema]:
        return None


class OllamaClient:
    provider = "ollama"

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_URL,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS, transport: Optional[httpx.BaseTransport] = None):
        self.model = model
        self.reason = ""
        self.last_error: Optional[str] = None
        # `transport` lets tests replace the network with a fake one.
        self.http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def is_available(self, timeout: float = 3.0) -> bool:
        """True if the server answers and has the configured model installed.

        Uses GET /api/tags, which only lists installed models: it never loads the model.
        """
        try:
            response = self.http.get("/api/tags", timeout=timeout)
            response.raise_for_status()
            names = [entry.get("name") for entry in response.json().get("models", [])]
        except (httpx.HTTPError, ValueError) as error:
            self.reason = f"Ollama not reachable: {error.__class__.__name__}"
            return False
        if self.model not in names:
            self.reason = f"model {self.model} is not installed in Ollama"
            return False
        return True

    def ask_json(self, system: str, user: str, schema: Type[Schema]) -> Optional[Schema]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema.model_json_schema(),
            "stream": False,
            # gemma4 "thinks" before answering by default. Thinking tokens count against
            # num_predict, so with thinking on most replies were cut off before the JSON
            # (measured: 13 of 16 agent calls failed). A 14-post comparison found thinking
            # gave the same sarcasm accuracy at 3-6x the latency, so it is turned off.
            "think": False,
            "keep_alive": "10m",   # keep the model loaded between the agents' calls
            "options": {"temperature": 0, "num_predict": MAX_REPLY_TOKENS},
        }
        # Two attempts, but only a bad reply is worth retrying. A timeout or a refused
        # connection would just fail again, so those give up at once.
        for _ in range(2):
            try:
                response = self.http.post("/api/chat", json=payload)
                response.raise_for_status()
                content = response.json()["message"]["content"]
            except httpx.HTTPError as error:
                self.last_error = f"{error.__class__.__name__}: {error}"
                return None
            except (KeyError, TypeError, ValueError) as error:   # response body is not the expected shape
                self.last_error = f"unexpected Ollama response: {error}"
                return None
            try:
                return schema.model_validate_json(content)
            except ValidationError as error:
                self.last_error = f"invalid JSON for {schema.__name__}: {error.error_count()} error(s)"
        return None


def get_llm_client():
    """Build the client from environment variables.

    EMOCONTEXT_LLM=none gives a NullClient. Otherwise an OllamaClient is returned even if Ollama is
    not running yet: each call either succeeds or fails fast and the agent uses its fallback, so
    Ollama can be started or stopped later without restarting the API. `/health` probes it live.
    """
    if os.getenv("EMOCONTEXT_LLM", "ollama").lower() == "none":
        return NullClient("LLM disabled by EMOCONTEXT_LLM=none")
    client = OllamaClient(model=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
                          base_url=os.getenv("OLLAMA_URL", DEFAULT_URL),
                          timeout=float(os.getenv("OLLAMA_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)))
    client.is_available()   # sets `reason` for logs if Ollama is down now; the client is kept either way
    return client
