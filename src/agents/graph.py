"""The LangGraph workflow that connects the six agents through the shared state.

    START -> emotion -> sentiment -> context -> intent -> response -> critic
    critic -> END        when the Critic wrote `final`
    critic -> response   when the Critic sent the draft back (at most once)

The agents run one after another. A local Ollama server normally generates one reply at a
time, so running agents in parallel would not make it faster, and the Intent Agent reads
the Context Agent's output.

The graph is built once (loading the model is the slow part) and reused for every post.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, START, StateGraph

from agents import context_agent, critic_agent, emotion_agent, intent_agent, response_agent, sentiment_agent
from agents.state import AgentState

# One revision means at most 8 node runs; this limit is only a safety net.
RECURSION_LIMIT = 12


def route_after_critic(state: AgentState) -> str:
    return END if "final" in state else "response_agent"


def build_graph(predictor, llm, emotion_baseline: Optional[Callable[[str], str]] = None,
                sentiment_baseline: Optional[Callable[[str], str]] = None):
    """`predictor` is a loaded model.Predictor and `llm` an OllamaClient or NullClient.

    Node names end in "_agent" because LangGraph does not allow a node to share its name
    with a state key such as "emotion".
    """
    graph = StateGraph(AgentState)
    graph.add_node("emotion_agent", lambda state: emotion_agent.run(state, predictor, emotion_baseline))
    graph.add_node("sentiment_agent", lambda state: sentiment_agent.run(state, sentiment_baseline))
    graph.add_node("context_agent", lambda state: context_agent.run(state, llm))
    graph.add_node("intent_agent", lambda state: intent_agent.run(state, llm))
    graph.add_node("response_agent", lambda state: response_agent.run(state, llm))
    graph.add_node("critic_agent", lambda state: critic_agent.run(state, llm))

    graph.add_edge(START, "emotion_agent")
    graph.add_edge("emotion_agent", "sentiment_agent")
    graph.add_edge("sentiment_agent", "context_agent")
    graph.add_edge("context_agent", "intent_agent")
    graph.add_edge("intent_agent", "response_agent")
    graph.add_edge("response_agent", "critic_agent")
    graph.add_conditional_edges("critic_agent", route_after_critic, {END: END, "response_agent": "response_agent"})
    return graph.compile()


def analyse(graph, text: str) -> AgentState:
    """Run one post through every agent and return the final shared state."""
    initial: AgentState = {"text": text, "revision_count": 0, "trace": [], "errors": []}
    return graph.invoke(initial, config={"recursion_limit": RECURSION_LIMIT})


def state_to_dict(state: AgentState) -> Dict[str, Any]:
    """Turn the final state into plain JSON-ready data (pydantic models become dicts)."""
    def plain(value):
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, list):
            return [plain(item) for item in value]
        return value

    return {key: plain(value) for key, value in state.items()}
