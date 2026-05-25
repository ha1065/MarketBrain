"""
graph.py

LangGraph orchestration for MarketMind.

Assumed structure:
    project_root/
    ├── app.py
    ├── graph.py
    ├── state.py
    └── agents/
        ├── __init__.py
        ├── price_agent.py
        ├── sentiment_agent.py
        ├── macro_agent.py
        ├── onchain_agent.py
        ├── risk_agent.py
        └── synthesis_agent.py

This graph runs the analysis agents in parallel and then sends their combined
signals into synthesis_agent.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from state import AgentSignal

# Agent files are inside the agents/ package.
from agents.price_agent import price_agent
from agents.sentiment_agent import sentiment_agent
from agents.macro_agent import macro_agent
from agents.onchain_agent import onchain_agent
from agents.risk_agent import risk_agent
from agents.synthesis_agent import synthesis_agent


AssetType = Literal["stock", "crypto"]


class MarketGraphState(TypedDict, total=False):
    ticker: str
    asset_type: AssetType

    # Multiple parallel nodes write to this key.
    # operator.add concatenates lists safely during fan-in.
    agent_signals: Annotated[list[Any], operator.add]

    metadata: dict[str, Any]


def _fallback_signal(agent_name: str, message: str) -> AgentSignal:
    return AgentSignal(
        agent=agent_name,
        signal="NEUTRAL",
        confidence=0.0,
        summary=message[:120],
        raw_data={},
    )


def _safe_agent_node(agent_fn, agent_name: str):
    """Wrap each agent so one broken agent does not kill the whole graph."""

    def node(state: MarketGraphState) -> dict[str, list[Any]]:
        try:
            result = agent_fn(state)

            if result is None:
                return {
                    "agent_signals": [
                        _fallback_signal(agent_name, f"{agent_name} returned None")
                    ]
                }

            if isinstance(result, dict):
                signals = result.get("agent_signals", [])
                if signals is None:
                    signals = []
                if not isinstance(signals, list):
                    signals = [signals]
                return {"agent_signals": signals}

            # If an agent accidentally returns one signal object directly.
            return {"agent_signals": [result]}

        except Exception as exc:
            return {
                "agent_signals": [
                    _fallback_signal(agent_name, f"{agent_name} failed: {str(exc)}")
                ]
            }

    return node


def _synthesis_node(state: MarketGraphState) -> dict[str, list[Any]]:
    """Run synthesis and always return a valid graph update."""
    try:
        result = synthesis_agent(state)

        if result is None:
            return {
                "agent_signals": [
                    _fallback_signal("synthesis_agent", "synthesis_agent returned None")
                ]
            }

        if isinstance(result, dict):
            signals = result.get("agent_signals", [])
            if signals is None:
                signals = []
            if not isinstance(signals, list):
                signals = [signals]
            return {"agent_signals": signals}

        return {"agent_signals": [result]}

    except Exception as exc:
        return {
            "agent_signals": [
                _fallback_signal("synthesis_agent", f"synthesis_agent failed: {str(exc)}")
            ]
        }


def build_market_graph():
    workflow = StateGraph(MarketGraphState)

    parallel_nodes = {
        "price_agent": _safe_agent_node(price_agent, "price_agent"),
        "sentiment_agent": _safe_agent_node(sentiment_agent, "sentiment_agent"),
        "macro_agent": _safe_agent_node(macro_agent, "macro_agent"),
        "onchain_agent": _safe_agent_node(onchain_agent, "onchain_agent"),
        "risk_agent": _safe_agent_node(risk_agent, "risk_agent"),
    }

    for node_name, node_fn in parallel_nodes.items():
        workflow.add_node(node_name, node_fn)

    workflow.add_node("synthesis_agent", _synthesis_node)

    # Fan out: run all analysis agents in parallel.
    for node_name in parallel_nodes:
        workflow.add_edge(START, node_name)

    # Fan in: synthesis waits for all upstream nodes.
    for node_name in parallel_nodes:
        workflow.add_edge(node_name, "synthesis_agent")

    workflow.add_edge("synthesis_agent", END)

    return workflow.compile()


market_graph = build_market_graph()


def analyze_asset(
    ticker: str,
    asset_type: AssetType = "stock",
    metadata: Optional[dict[str, Any]] = None,
) -> MarketGraphState:
    initial_state: MarketGraphState = {
        "ticker": ticker.strip().upper(),
        "asset_type": asset_type,
        "agent_signals": [],
        "metadata": metadata or {},
    }

    result = market_graph.invoke(initial_state)

    # Guardrail for Streamlit: app.py expects a dict-like result.
    if result is None:
        return {
            **initial_state,
            "agent_signals": [
                _fallback_signal("graph", "LangGraph returned None")
            ],
        }

    return result


if __name__ == "__main__":
    output = analyze_asset("AAPL", "stock")
    print(output)
