import os 
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from agents import llm, state_parser
from state import MarketMindState, AgentSignal

CRYPTO_NAMES = {"BTC": "Bitcoin",
 "ETH": "Ethereum",
 "SOL": "Solana",
 "BNB": "Binance Coin",
 "ADA": "Cardano",
 "AVAX": "Avalanche",
 "DOT": "Polkadot",
 "MATIC": "Polygon",
 "LTC": "Litecoin",
 "LINK": "Chainlink",
 "XRP": "Ripple"}



def sentiment_agent(state: MarketMindState) -> AgentSignal:
    ticker = state["ticker"]
    asset_type = state["asset_type"]
    try :
        newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_API_KEY"))

        query = CRYPTO_NAMES.get(ticker, ticker) if asset_type == "crypto" else ticker
        from_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')

        response = newsapi.get_everything(q=query, from_param=from_date, language='en', sort_by='relevancy', page_size=10)

        headlines = [a["title"] for a in response.get("articles", [])]
        if not headlines:
            return {"agent_signals": [AgentSignal(agent="sentiment",signal="neutral", confidence=0.3,
                                                  summary="No relevant news articles found for sentiment analysis.",
                                                  raw_data={"headline_count": 0})]}
        headlines_text= "\n".join(f"- {h}" for h in headlines
                                  )
        
        prompt= f"""You are a financial sentiment analyst. Rate the market sentiment for {ticker} based on the following recent news headlines:

{headlines_text}
Return ONLY valid JSON with exactly these fields, and no other text:
{{signal: "BULLISH", "BEARISH" or "NEUTRAl", "confidence" : 0.0 to 1.0,
"summary" : "one sentence under 20 words", "positive_count": number, "negative_count": number}}
"""
        result_raw = llm.invoke(prompt)
        result  = state_parser(result_raw.content)

        return {"agent_signals": [AgentSignal(agent="sentiment", signal=result["signal"], confidence=result["confidence"],
                                                summary=result["summary"], raw_data={"headline_count": len(headlines), 
                                                                                     "positive_count": result.get("positive_count", 0),
                                                                                     "negative_count": result.get("negative_count", 0),
                                                                                     "sample_headline": headlines[0]})]}
    
    except Exception as e:
        return {"agent_signals": [AgentSignal(agent="sentiment", signal="neutral", confidence=0.0,
                                                summary=f"Error during sentiment analysis: {str(e)[:60]}",
                                                raw_data={})]}