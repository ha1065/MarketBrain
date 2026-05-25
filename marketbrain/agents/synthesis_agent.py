import json

from agents import llm, state_parser
from state import MarketMindState, AgentSignal


def synthesis_agent(state: MarketMindState) -> dict:
    try:
        agent_signals = state.get("agent_signals", [])

        summarized_signals = []

        for signal in agent_signals:
            if isinstance(signal, dict):
                    summarized_signals.append({
            "agent": signal.get("agent", "unknown_agent"),
            "signal": signal.get("signal", "NEUTRAL"),
            "confidence": signal.get("confidence", 0),
            "summary": signal.get("summary", ""),
        })
            else:
                summarized_signals.append({
            "agent": getattr(signal, "agent", "unknown_agent"),
            "signal": getattr(signal, "signal", "NEUTRAL"),
            "confidence": getattr(signal, "confidence", 0),
            "summary": getattr(signal, "summary", ""),
        })

        prompt = f"""
You are a chief investment strategist.

You are given outputs from multiple market analysis agents.

Agent Outputs:
{json.dumps(summarized_signals, indent=2)}

Your task:
- synthesize the signals
- identify consensus or conflicts
- determine overall market outlook

Return only valid JSON in this format:

{{
    "signal": "BULLISH" or "BEARISH" or "NEUTRAL",
    "confidence": float between 0 and 1,
    "summary": "one sentence under 25 words"
}}
"""

        result_raw = llm.invoke(prompt)

        result = state_parser(result_raw.content)

        return {
            "agent_signals": [
                AgentSignal(
                    agent="synthesis_agent",
                    signal=result["signal"],
                    confidence=float(result["confidence"]),
                    summary=result["summary"],
                    raw_data={
                        "source_signals": summarized_signals
                    },
                )
            ]
        }

    except Exception as e:
        return {
            "agent_signals": [
                AgentSignal(
                    agent="synthesis_agent",
                    signal="NEUTRAL",
                    confidence=0,
                    summary=f"Synthesis agent error: {str(e)[:60]}",
                    raw_data={},
                )
            ]
        }