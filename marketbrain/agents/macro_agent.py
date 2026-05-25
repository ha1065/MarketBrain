import os
import json
import requests
from agents import llm, state_parser
from state import MarketMindState, AgentSignal


def fetch_fred(series_id: str, api_key: str, limit: int = 5) -> list:
    url = "https://api.stlouisfed.org/fred/series/observations"

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "limit": limit,
        "sort_order": "desc",
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    observations = response.json()["observations"]

    return [
        {
            "date": o["date"],
            "value": round(float(o["value"]), 2)
            if o["value"] != "."
            else None,
        }
        for o in observations
    ]


def macro_agent(state: MarketMindState) -> dict:
    ticker = state.get("ticker")

    fred_api_key = os.getenv("FRED_API_KEY")

    FRED_SERIES = {
        "CPI": "CPIAUCSL",
        "UNEMPLOYMENT": "UNRATE",
        "FED_RATE": "FEDFUNDS",
        "GDP": "GDP",
        "VIX": "VIXCLS",
    }

    try:
        raw_data = {}

        for name, series_id in FRED_SERIES.items():
            raw_data[name] = fetch_fred(
                series_id=series_id,
                api_key=fred_api_key,
                limit=3,
            )

        prompt = f"""
You are a macroeconomic analyst.

Analyze this macroeconomic data for {ticker}:

{json.dumps(raw_data, indent=2)}

Focus on:
- inflation trends
- interest rates
- unemployment
- economic growth
- market volatility

Return only valid JSON with this format:

{{
    "signal": "BULLISH" or "BEARISH" or "NEUTRAL",
    "confidence": float between 0 and 1,
    "summary": "one sentence under 20 words"
}}
"""

        result_raw = llm.invoke(prompt)
        result = state_parser(result_raw.content)

        return {
            "agent_signals": [
                AgentSignal(
                    agent="macro_agent",
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
                    agent="macro_agent",
                    signal="NEUTRAL",
                    confidence=0,
                    summary=f"Macro agent error: {str(e)[:60]}",
                    raw_data={},
                )
            ]
        }