"""
app.py

Streamlit dashboard for the MarketMind agentic stock/crypto analysis system.

Run with:
    streamlit run app.py

Expected project files in the same folder:
    graph.py
    state.py
    agents.py
    price_agent.py
    sentiment_agent.py
    macro_agent.py
    onchain_agent.py
    risk_agent.py
    synthesis_agent.py

Required environment variables depend on the agents you enable, for example:
    GROQ_API_KEY
    NEWSAPI_API_KEY
    FRED_API_KEY
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable

import pandas as pd
import streamlit as st

try:
    from graph import analyze_asset, market_graph
except Exception as import_error:  # Dashboard should still load and show the issue.
    analyze_asset = None
    market_graph = None
    GRAPH_IMPORT_ERROR = import_error
else:
    GRAPH_IMPORT_ERROR = None


st.set_page_config(
    page_title="MarketMind Agent Dashboard",
    page_icon="📈",
    layout="wide",
)


SIGNAL_COLORS = {
    "BULLISH": "#16a34a",
    "BEARISH": "#dc2626",
    "NEUTRAL": "#64748b",
    "LOW_RISK": "#16a34a",
    "MODERATE_RISK": "#f59e0b",
    "HIGH_RISK": "#dc2626",
}


def _as_dict(obj: Any) -> dict[str, Any]:
    """Convert Pydantic models, dataclasses, TypedDicts, and plain objects to dicts."""
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    if hasattr(obj, "dict"):
        return obj.dict()

    return {
        "agent": getattr(obj, "agent", "unknown"),
        "signal": getattr(obj, "signal", "NEUTRAL"),
        "confidence": getattr(obj, "confidence", 0.0),
        "summary": getattr(obj, "summary", ""),
        "raw_data": getattr(obj, "raw_data", {}),
    }


def _normalize_signals(signals: Iterable[Any]) -> list[dict[str, Any]]:
    return [_as_dict(signal) for signal in signals or []]


def _get_final_synthesis(signals: list[dict[str, Any]]) -> dict[str, Any] | None:
    for signal in reversed(signals):
        if str(signal.get("agent", "")).lower() == "synthesis_agent":
            return signal
    return None


def _get_analysis_signals(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        signal
        for signal in signals
        if str(signal.get("agent", "")).lower() != "synthesis_agent"
    ]


def _confidence_to_pct(confidence: Any) -> int:
    try:
        value = float(confidence)
    except Exception:
        return 0

    if value <= 1:
        value *= 100

    return max(0, min(100, int(round(value))))


def _signal_badge(signal: str) -> str:
    normalized = str(signal or "NEUTRAL").upper()
    color = SIGNAL_COLORS.get(normalized, "#64748b")
    return (
        f"<span style='background:{color}; color:white; padding:0.25rem 0.6rem; "
        f"border-radius:999px; font-size:0.8rem; font-weight:700;'>{normalized}</span>"
    )


def _display_signal_card(signal: dict[str, Any]) -> None:
    agent = signal.get("agent", "unknown_agent")
    signal_value = str(signal.get("signal", "NEUTRAL")).upper()
    confidence = _confidence_to_pct(signal.get("confidence", 0.0))
    summary = signal.get("summary", "No summary returned.")
    raw_data = signal.get("raw_data", {}) or {}

    with st.container(border=True):
        left, right = st.columns([0.7, 0.3])

        with left:
            st.subheader(str(agent).replace("_", " ").title())
            st.markdown(_signal_badge(signal_value), unsafe_allow_html=True)
            st.write(summary)

        with right:
            st.metric("Confidence", f"{confidence}%")
            st.progress(confidence / 100)

        if raw_data:
            with st.expander("Raw data"):
                st.json(raw_data)


def _signals_dataframe(signals: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for signal in signals:
        rows.append(
            {
                "Agent": signal.get("agent", "unknown"),
                "Signal": signal.get("signal", "NEUTRAL"),
                "Confidence": round(float(signal.get("confidence", 0) or 0), 3),
                "Summary": signal.get("summary", ""),
            }
        )
    return pd.DataFrame(rows)


st.title("📈 MarketMind Agent Dashboard")
st.caption("Parallel stock/crypto analysis using LangGraph agents and a final synthesis layer.")

with st.sidebar:
    st.header("Analysis Inputs")

    asset_type = st.selectbox(
        "Asset type",
        options=["stock", "crypto"],
        index=0,
        help="Choose stock for tickers like AAPL/MSFT, or crypto for BTC/ETH.",
    )

    default_ticker = "AAPL" if asset_type == "stock" else "BTC"
    ticker = st.text_input("Ticker", value=default_ticker).strip().upper()

    st.divider()

    with st.expander("Optional metadata"):
        portfolio_context = st.text_area(
            "Portfolio / thesis context",
            placeholder="Example: long-term growth holding, short-term swing trade, risk-averse portfolio...",
        )
        notes = st.text_area("Extra notes", placeholder="Any constraints or assumptions.")

    run_button = st.button("Run Analysis", type="primary", use_container_width=True)

    st.divider()
    st.caption("Make sure your API keys are available in your environment before running.")


if GRAPH_IMPORT_ERROR is not None:
    st.error("The LangGraph workflow could not be imported.")
    st.exception(GRAPH_IMPORT_ERROR)
    st.stop()

if not ticker:
    st.info("Enter a ticker in the sidebar to start.")
    st.stop()

if run_button:
    metadata = {
        "portfolio_context": portfolio_context,
        "notes": notes,
        "dashboard_run_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    with st.status("Running parallel agents...", expanded=True) as status:
        st.write("Launching price, sentiment, macro, on-chain/market-structure, and risk agents.")

        try:
            result = analyze_asset(
                ticker=ticker,
                asset_type=asset_type,
                metadata=metadata,
            )
        except Exception as exc:
            status.update(label="Analysis failed", state="error")
            st.error("The graph failed while running the analysis.")
            st.exception(exc)
            st.stop()

        status.update(label="Analysis complete", state="complete")

    st.session_state["last_result"] = result
    st.session_state["last_ticker"] = ticker
    st.session_state["last_asset_type"] = asset_type


result = st.session_state.get("last_result")

if result is None:
    st.info("Choose an asset and click **Run Analysis**.")
    st.stop()

signals = _normalize_signals(result.get("agent_signals", []))
analysis_signals = _get_analysis_signals(signals)
synthesis = _get_final_synthesis(signals)

st.header(f"Results for {st.session_state.get('last_ticker', ticker)}")
st.caption(f"Asset type: {st.session_state.get('last_asset_type', asset_type)}")

if synthesis:
    st.subheader("Final Synthesis")
    with st.container(border=True):
        col1, col2, col3 = st.columns([0.35, 0.25, 0.4])

        with col1:
            st.markdown(_signal_badge(str(synthesis.get("signal", "NEUTRAL"))), unsafe_allow_html=True)

        with col2:
            confidence = _confidence_to_pct(synthesis.get("confidence", 0.0))
            st.metric("Confidence", f"{confidence}%")

        with col3:
            st.write(synthesis.get("summary", "No synthesis summary returned."))
else:
    st.warning("No synthesis signal was returned by the graph.")

st.divider()

st.subheader("Agent Signals")

if not analysis_signals:
    st.warning("No individual agent signals were returned.")
else:
    cols = st.columns(2)
    for idx, signal in enumerate(analysis_signals):
        with cols[idx % 2]:
            _display_signal_card(signal)

    st.divider()
    st.subheader("Signal Table")
    st.dataframe(_signals_dataframe(signals), use_container_width=True, hide_index=True)

    csv = _signals_dataframe(signals).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download signals as CSV",
        data=csv,
        file_name=f"marketmind_{st.session_state.get('last_ticker', ticker)}_signals.csv",
        mime="text/csv",
    )

with st.expander("Full graph output"):
    try:
        st.json(json.loads(json.dumps(result, default=str)))
    except Exception:
        st.write(result)
