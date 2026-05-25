"""
graph.py

LangGraph orchestration for MarketMind's agentic stock/crypto analysis system.

Flow:
    START
      ├── price_agent
      ├── sentiment_agent
      ├── macro_agent
      ├── onchain_agent
      └── risk_agent
          ↓ parallel fan-in
    synthesis_agent
          ↓
        END

Each analysis agent returns {"agent_signals": [AgentSignal(...)]}. The graph uses
an additive reducer so parallel nodes can safely append their signals into the
same state key before synthesis runs.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from state import AgentSignal
from price_agent import price_agent
from sentiment_agent import sentiment_agent
from macro_agent import macro_agent
from onchain_agent import onchain_agent
from risk_agent import risk_agent
from synthesis_agent import synthesis_agent


AssetType = Literal["stock", "crypto"]


class MarketGraphState(TypedDict, total=False):
    """Shared graph state.

    `agent_signals` needs a reducer because multiple parallel nodes write to it
    at the same time. `operator.add` concatenates the lists returned by agents.
    """

    ticker: str
    asset_type: AssetType
    agent_signals: Annotated[list[AgentSignal], operator.add]
    metadata: dict[str, Any]


def _safe_agent_node(agent_fn, agent_name: str):
    """Wrap an agent so graph execution continues even if one node fails."""

    def node(state: MarketGraphState) -> dict[str, list[AgentSignal]]:
        try:
            result = agent_fn(state)

            # Normal expected format: {"agent_signals": [AgentSignal(...)]}
            if isinstance(result, dict) and "agent_signals" in result:
                return {"agent_signals": result["agent_signals"]}

            # Fallback if an agent accidentally returns a single AgentSignal.
            if isinstance(result, AgentSignal):
                return {"agent_signals": [result]}

            raise ValueError(f"Unexpected return format from {agent_name}: {type(result)}")

        except Exception as exc:
            return {
                "agent_signals": [
                    AgentSignal(
                        agent=agent_name,
                        signal="NEUTRAL",
                        confidence=0.0,
                        summary=f"{agent_name} failed: {str(exc)[:80]}",
                        raw_data={},
                    )
                ]
            }

    return node


def build_market_graph():
    """Build and compile the MarketMind parallel agent graph."""

    graph = StateGraph(MarketGraphState)

    # Parallel analysis nodes
    graph.add_node("price_agent", _safe_agent_node(price_agent, "price_agent"))
    graph.add_node("sentiment_agent", _safe_agent_node(sentiment_agent, "sentiment_agent"))
    graph.add_node("macro_agent", _safe_agent_node(macro_agent, "macro_agent"))
    graph.add_node("onchain_agent", _safe_agent_node(onchain_agent, "onchain_agent"))
    graph.add_node("risk_agent", _safe_agent_node(risk_agent, "risk_agent"))

    # Reducer / final synthesis node
    graph.add_node("synthesis_agent", synthesis_agent)

    parallel_agents = [
        "price_agent",
        "sentiment_agent",
        "macro_agent",
        "onchain_agent",
        "risk_agent",
    ]

    # Fan out from START so all analysis dimensions compute in parallel.
    for agent in parallel_agents:
        graph.add_edge(START, agent)

    # Fan in all agent outputs into synthesis_agent.
    for agent in parallel_agents:
        graph.add_edge(agent, "synthesis_agent")

    graph.add_edge("synthesis_agent", END)

    return graph.compile()


# Reusable compiled graph instance
market_graph = build_market_graph()


def analyze_asset(
    ticker: str,
    asset_type: AssetType = "stock",
    metadata: Optional[dict[str, Any]] = None,
) -> MarketGraphState:
    """Convenience runner for one stock or crypto asset.

    Example:
        result = analyze_asset("AAPL", "stock")
        result = analyze_asset("BTC", "crypto")
    """

    initial_state: MarketGraphState = {
        "ticker": ticker.upper(),
        "asset_type": asset_type,
        "agent_signals": [],
        "metadata": metadata or {},
    }

    return market_graph.invoke(initial_state)


if __name__ == "__main__":
    # Lightweight smoke test. Requires your API keys/env vars to be configured.
    result = analyze_asset("AAPL", "stock")
    for signal in result.get("agent_signals", []):
        print(signal)
