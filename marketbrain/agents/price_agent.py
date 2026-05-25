import yfinance as yf
import pandas as pd
from agents import llm, state_parser
from state import MarketMindState, AgentSignal

def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss    
    return round(100 - (100 / (1 + rs)).iloc[-1], 2)

def compute_macd(prices: pd.Series):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return round(float(macd_line.iloc[-1]),4), round(float(signal_line.iloc[-1]),4)

def compute_bollinger(prices: pd.Series, period: int = 20):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    band_range = float(upper_band.iloc[-1]) - float(lower_band.iloc[-1]) 
    if band_range == 0.5: 
        return 0.5
    return round((float(prices.iloc[-1]) - float(lower_band.iloc[-1])) / band_range, 3)

def price_agent(state: MarketMindState) -> dict:
    ticker = state['ticker']

    yfin_ticker = f"{ticker}-USD" if state['asset_type'] == 'crypto' else ticker
    
    try:
        data = yf.download(yfin_ticker, period = "90d", interval = "1d",progress=False)

        if data.empty:
            return {"agent_signals": 
                    [AgentSignal(agent= "price", signal = "NEUTRAL", confidence = 0.0,
                                 summary = f"No data found for {ticker}", raw_data = {})]}
    
        close = data['Close'].squeeze()
        current_price = round(float(close.iloc[-1]), 4)
        price_7d_ago = round(float(close.iloc[-7]), 4)
        change_7d = round(((current_price - price_7d_ago) / price_7d_ago) * 100, 2)
        rsi = compute_rsi(close)
        macd_line, signal_line = compute_macd(close)
        bollinger = compute_bollinger(close)

        raw_data = {
            "current_price": current_price,
            "price_change_7d_pct": change_7d,
            "rsi": rsi,
            "macd_line" : macd_line,
            "macd_signal": signal_line,
            "macd_crossover": "postive" if macd_line > signal_line else "negative",
            "bollinger_position": bollinger}
        
        prompt = f"""You are a technical analyst. Analyze these indicators for {ticker}:

        RSI: {rsi} (below 30 is oversold, above 70 is overbought)
        macd_crossover: {raw_data['macd_crossover']} (positive is bullish momentum)
        Bollinger Position: {bollinger} (0 is lower band, 1 is upper band)
        7-day Price Change: {change_7d}% 

        Return ONLY valid JSON with exactly these fields, no other text:
        {{
            "signal": "BULLISH" or "BEARISH" or "NEUTRAL",
            "confidence": 0.0-1.0,
            "summary": "one sentence under 20 words
        }}"""
        
        response = llm.invoke(prompt)
        result = state_parser(response.content)

        return {"agent_signals": [AgentSignal(agent= "PriceAgent", signal = result['signal'], confidence = result['confidence'],
                                 summary = result['summary'], raw_dat = raw_data)]}
    except Exception as e:
        return {"agent_signals": [AgentSignal(agent= "price", signal = "NEUTRAL", confidence = 0.0,
                                 summary = f"Error fetching data for {ticker}: {str(e)}", raw_data = {})]}

    data = yf.download(ticker, period="1mo", interval="1d")
    if data.empty:
        return {"error": f"No data found for {ticker}"}

    rsi = compute_rsi(data['Close'])
    macd, signal = compute_macd(data['Close'])
    bollinger_pos = compute_bollinger(data['Close'])

    signal_summary = f"RSI: {rsi}, MACD: {macd} (Signal: {signal}), Bollinger Position: {bollinger_pos}"
    
    agent_signal: AgentSignal = {
        "agent": "PriceAgent",
        "signal": signal_summary,
        "confidence": 0.7,
        "summary": signal_summary,
        "raw_dat": {
            "rsi": rsi,
            "macd": macd,
            "signal_line": signal,
            "bollinger_position": bollinger_pos
        }
    }

    return agent_signal