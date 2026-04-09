"""
Founder Research Agent – LangGraph-based autonomous research workflow.

Graph topology:
  plan → search → scrape → analyse → memory_write → [loop or finalise] → report
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent.nodes import (
    analyse_node,
    finalise_node,
    memory_write_node,
    plan_node,
    report_node,
    scrape_node,
    search_node,
)
from agent.state import ResearchState

logger = logging.getLogger(__name__)


# ── routing helpers ──────────────────────────────────────────────────────────

def route_after_analysis(state: ResearchState) -> Literal["search", "finalise"]:
    """Continue searching if there are pending queries; otherwise finalise."""
    if state.get("pending_queries") and state.get("iterations", 0) < state.get("max_iterations", 5):
        return "search"
    return "finalise"


def route_after_finalise(state: ResearchState) -> Literal["report"]:
    return "report"


# ── graph builder ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("plan",         plan_node)
    graph.add_node("search",       search_node)
    graph.add_node("scrape",       scrape_node)
    graph.add_node("analyse",      analyse_node)
    graph.add_node("memory_write", memory_write_node)
    graph.add_node("finalise",     finalise_node)
    graph.add_node("report",       report_node)

    # Edges
    graph.add_edge(START,           "plan")
    graph.add_edge("plan",          "search")
    graph.add_edge("search",        "scrape")
    graph.add_edge("scrape",        "analyse")
    graph.add_edge("analyse",       "memory_write")
    graph.add_conditional_edges(
        "memory_write",
        route_after_analysis,
        {"search": "search", "finalise": "finalise"},
    )
    graph.add_edge("finalise",      "report")
    graph.add_edge("report",        END)

    return graph


def compile_graph():
    return build_graph().compile()
