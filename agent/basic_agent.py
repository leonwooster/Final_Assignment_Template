"""Basic LangGraph-powered agent implementation."""

from __future__ import annotations

from typing import Optional

from .graph import AgentState, build_agent_graph


class BasicAgent:
    """Route questions through the LangGraph and return answers."""

    def __init__(self) -> None:
        self.graph = build_agent_graph()

    def __call__(self, question: str) -> str:
        state: AgentState = {"question": question}
        result = self.graph.invoke(state)
        answer: Optional[str] = result.get("answer")
        if answer is None:
            return "UNHANDLED"
        return answer
