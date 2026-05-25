import json
import numpy as np
import yfinance as yf

from agents import llm, state_parser
from state import MarketMindState, AgentSignal


def risk_agent(state: MarketMindState) -> dict:
    ticker = state.get("ticker")

    try:
        stock = yf.Ticker(ticker)

        hist = stock.history(period="6mo")

        if hist.empty:
            raise ValueError("No historical market data found")

        closes = hist["Close"]

        returns = closes.pct_change().dropna()

        volatility = returns.std() * np.sqrt(252)

        max_drawdown = (
            (closes / closes.cummax()) - 1
        ).min()

        sharpe_ratio = (
            returns.mean() / returns.std()
        ) * np.sqrt(252)

        raw_data = {
            "annualized_volatility": round(float(volatility), 4),
            "max_drawdown": round(float(max_drawdown), 4),
            "sharpe_ratio": round(float(sharpe_ratio), 4),
            "recent_price": round(float(closes.iloc[-1]), 2),
        }

        prompt = f"""
You are a financial risk analyst.

Analyze this risk data for {ticker}:

{json.dumps(raw_data, indent=2)}

Focus on:
- volatility
- downside risk
- risk-adjusted returns
- portfolio stability

Return only valid JSON with this format:

{{
    "signal": "LOW_RISK" or "HIGH_RISK" or "MODERATE_RISK",
    "confidence": float between 0 and 1,
    "summary": "one sentence under 20 words"
}}
"""

        result_raw = llm.invoke(prompt)

        result = state_parser(result_raw.content)

        return {
            "agent_signals": [
                AgentSignal(
                    agent="risk_agent",
                    signal=result["signal"],
                    confidence=float(result["confidence"]),
                    summary=result["summary"],
                    raw_data=raw_data,
                )
            ]
        }

    except Exception as e:
        return {
            "agent_signals": [
                AgentSignal(
                    agent="risk_agent",
                    signal="MODERATE_RISK",
                    confidence=0,
                    summary=f"Risk agent error: {str(e)[:60]}",
                    raw_data={},
                )
            ]
        }