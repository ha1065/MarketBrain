import json 
import yfinance as yf
from pycoingecko import CoinGeckoAPI
from agents import llm, state_parser
from state import MarketMindState, AgentSignal

COINGECKO_IDS = {
    "BTC": "bitcoin",
 "ETH": "ethereum",
 "SOL": "solana",
 "BNB": "binance coin",
 "ADA": "cardano",
 "AVAX": "avalanche",
 "DOT": "polkadot",
 "MATIC": "polygon",
 "LTC": "litecoin",
 "LINK": "chainlink",
 "XRP": "ripple"
}

def onchain_agent(state: MarketMindState) -> dict:
    ticker = state.get("ticker")
    asset_type = state.get("asset_type")
    try:
        if asset_type == "crypto":
            coin_id = COINGECKO_IDS.get(ticker.upper())
            if not coin_id:
                return {"agent_signals": [AgentSignal("onchain_agent",
                                                      signal="NEUTRAL",
                                                      confidence=0.5,
                                                      summary=f"{ticker} not in supported crypto list", raw_data = {})]}

            cg = CoinGeckoAPI()
            data = cg.get_coin_by_id(coin_id, localization=False, tickers=False, market_data=True, community_data=False, developer_data=False, sparkline=False)
            market = data.get("market_data", {})
            market_cap = market.get("market_cap", {}).get("usd")
            total_vol = market.get("total_volume", {}).get("usd")

            raw_data = {
                "market_cap_rank": data.get("market_cap_rank","N/A"),
                "price_change_24h_pct": round(data.get("price_change_24h", 0), 2),
                "price_change_7d_pct": round(data.get("price_change_percentage_7d", 0), 2),
                "volume_to_market_cap_ratio": round(total_vol / market_cap, 4) if market_cap >  0 else 0 ,
                "ath_change_pct": round(data.get("ath_change_percentage", {}).get("usd", 0), 2),
            }

        else:
                stock = yf.Ticker(ticker)
                info = stock.info
                avg_vol = max(info.get("averageVolume", 1), 1)  # Avoid division by zero
                raw_data = {
                    "volume_ratio" : round(info.get("volume", 0) / avg_vol, 3),
                    "market_cap": info.get("marketCap", 0),
                    "short_ratio": info.get("shortRatio", 0),
                    "beta": info.get("beta", 1.0),
                }

                prompt = f"""You are a market microstructure analyst. Analyze this data for {ticker} {{asset_type}}:
{json.dumps(raw_data, indent=2)}

Focus on : volume patterns, market activty, structural signals.

Return only valid JSON with this format:
{{
    "signal": "BULLISH" or "BEARISH" or "NEUTRAL",
    "confidence": float between 0 and 1,
    "summary": one sentence under 20 words
}}
Do not include markdown. Do not include explanations.
"""
                result_raw = llm.invoke(prompt)
                result = state_parser(result_raw.content)

                return {"agent_signals": [AgentSignal(agent="onchain",
                                                  signal = result["signal"],
                                                  confidence = float(result["confidence"]),
                                                  summary = result["summary"],
                                                  raw_data = raw_data)]}

    except Exception as e:
        return {"agent_signals": [AgentSignal(agent="onchain_agent",
                                              signal="NEUTRAL",
                                              confidence=0,
                                              summary=f"On-chain agent Error: {str(e)[:60]}", raw_data = {})]}



                                              