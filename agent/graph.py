"""LangGraph state graph construction for the BasicAgent."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

from .constants import CACHE_DIR, PATTERN_TOOL_MAP
from .tools import TOOL_REGISTRY


class AgentState(TypedDict, total=False):
    question: str
    answer: Optional[str]
    tool: Optional[str]


def build_agent_graph() -> StateGraph[AgentState]:
    graph_builder = StateGraph(AgentState)

    def router_node(state: AgentState) -> AgentState:
        question = state.get("question", "")
        for pattern, tool_name in PATTERN_TOOL_MAP:
            if pattern.search(question):
                state["tool"] = tool_name
                return state
        state["tool"] = None
        return state

    def route_condition(state: AgentState) -> str:
        return "execute" if state.get("tool") else "fallback"

    def execute_tool_node(state: AgentState) -> AgentState:
        tool_name = state.get("tool")
        question = state.get("question", "")
        if not tool_name:
            state["answer"] = "UNHANDLED"
            return state
        tool_instance = TOOL_REGISTRY[tool_name]
        try:
            result = tool_instance.invoke({"question": question})
        except Exception as exc:  # noqa: BLE001
            state["answer"] = f"AGENT ERROR: {exc}"
            return state
        state["answer"] = str(result)
        state["tool"] = tool_name
        return state

    def fallback_node(state: AgentState) -> AgentState:
        state["answer"] = "UNHANDLED"
        return state

    graph_builder.add_node("router", router_node)
    graph_builder.add_node("execute", execute_tool_node)
    graph_builder.add_node("fallback", fallback_node)

    graph_builder.set_entry_point("router")
    graph_builder.add_conditional_edges(
        "router",
        route_condition,
        {"execute": "execute", "fallback": "fallback"},
    )
    graph_builder.add_edge("execute", END)
    graph_builder.add_edge("fallback", END)

    return graph_builder.compile()


def agent_graph_mermaid() -> str:
    """Return the LangGraph structure in Mermaid format."""
    graph = build_agent_graph()
    compiled_graph = graph.get_graph()
    if hasattr(compiled_graph, "draw_mermaid"):
        return compiled_graph.draw_mermaid()
    raise RuntimeError("LangGraph version does not expose draw_mermaid(); please update LangGraph.")


def agent_graph_png_base64(filename: str = "agent_graph.png") -> Optional[str]:
    """Generate a PNG of the agent graph and return it as a base64 string."""
    graph = build_agent_graph()
    compiled_graph = graph.get_graph()
    if not hasattr(compiled_graph, "draw_png"):
        print("Warning: LangGraph graph object does not support draw_png(); install graphviz.")
        return None

    output_path = CACHE_DIR / filename
    try:
        compiled_graph.draw_png(str(output_path))
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Failed to render agent graph PNG: {exc}")
        return None

    try:
        encoded = base64.b64encode(output_path.read_bytes()).decode("ascii")
        return encoded
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Unable to read rendered graph PNG: {exc}")
        return None
