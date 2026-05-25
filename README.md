# MarketMind

An AI-powered multi-agent financial analysis platform for stocks and cryptocurrencies built with LangGraph, Streamlit, and LLM-based reasoning.

MarketMind orchestrates multiple specialized AI agents in parallel to generate market intelligence, risk assessments, sentiment analysis, macroeconomic insights, and final synthesized investment-style summaries.

---

## Features

* Multi-agent architecture using LangGraph
* Parallel execution of financial analysis agents
* Stock and cryptocurrency support
* AI-generated market synthesis and reasoning
* Risk analysis engine
* Macro-economic analysis integration
* On-chain crypto analytics
* News and sentiment analysis
* Interactive Streamlit dashboard
* Fault-tolerant graph orchestration
* Modular agent-based design for extensibility

---

## Architecture

```text
                    ┌──────────────────┐
                    │   User Input     │
                    │ (Ticker + Type)  │
                    └────────┬─────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │   LangGraph Flow   │
                  └────────┬───────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
 ┌────────────┐    ┌──────────────┐    ┌────────────┐
 │Price Agent │    │Sentiment     │    │Macro Agent │
 │            │    │Agent         │    │            │
 └────────────┘    └──────────────┘    └────────────┘
        │                  │                  │
        ├──────────────────┼──────────────────┤
        │                  │                  │
        ▼                  ▼                  ▼
 ┌────────────┐    ┌──────────────┐
 │Onchain     │    │Risk Agent    │
 │Agent       │    │              │
 └────────────┘    └──────────────┘
                
                           ▼
                 ┌─────────────────┐
                 │ Synthesis Agent │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │ Final Analysis  │
                 └─────────────────┘
```

---

## Project Structure

```text
marketmind/
│
├── agents/
│   ├── __init__.py
│   ├── macro_agent.py
│   ├── onchain_agent.py
│   ├── price_agent.py
│   ├── risk_agent.py
│   ├── sentiment_agent.py
│   └── synthesis_agent.py
│
├── app.py                # Streamlit dashboard
├── graph.py              # LangGraph orchestration
├── state.py              # Shared state models
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Agents Overview

### Price Agent

Performs price-based market analysis using historical market data and technical indicators.

### Sentiment Agent

Analyzes market sentiment using financial news and external information sources.

### Macro Agent

Evaluates macroeconomic conditions using FRED economic indicators.

### Onchain Agent

Performs blockchain and crypto market analysis using CoinGecko and related data sources.

### Risk Agent

Calculates market and portfolio-style risk levels based on aggregated agent outputs.

### Synthesis Agent

Combines all agent outputs into a final AI-generated market summary and recommendation-style insight.

---

## Tech Stack

* Python 3.10+
* Streamlit
* LangGraph
* LangChain
* Groq LLM API
* Yahoo Finance API
* CoinGecko API
* FRED API
* Pandas
* NumPy

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/marketmind.git
cd marketmind
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment:

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS/Linux

```bash
source .venv/bin/activate
```

### 3. Install dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Or using uv:

```bash
uv sync
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
NEWSAPI_API_KEY=your_newsapi_key
FRED_API_KEY=your_fred_api_key
```

---

## Running the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will launch in your browser.

---

## Example Usage

### Analyze a stock

```python
from graph import analyze_asset

result = analyze_asset("AAPL", "stock")
print(result)
```

### Analyze a cryptocurrency

```python
from graph import analyze_asset

result = analyze_asset("BTC", "crypto")
print(result)
```

---

## Fault Tolerance

Each agent is wrapped in a protected execution layer so that a single failing agent does not crash the entire workflow.

Fallback signals are automatically generated when:

* An agent throws an exception
* An agent returns invalid data
* An API request fails
* A model response is malformed

---

## Streamlit Dashboard Features

* Interactive ticker selection
* Asset type switching (stock/crypto)
* Agent signal visualization
* Confidence metrics
* Final synthesized recommendation
* Raw data inspection
* Error visibility for debugging

---

## Future Improvements

* Portfolio optimization agent
* Vector database memory
* Multi-timeframe technical analysis
* Backtesting engine
* Real-time websocket market feeds
* RAG-based financial research
* Autonomous trading workflows
* AWS deployment support
* Docker containerization
* Agent performance analytics

---

## Example Workflow

1. User enters ticker symbol
2. LangGraph initializes workflow state
3. Analysis agents run in parallel
4. Signals are aggregated
5. Risk evaluation is generated
6. Synthesis agent produces final analysis
7. Streamlit dashboard renders results

---

## Development Notes

The project uses:

* Typed graph state management
* Parallel fan-out/fan-in orchestration
* Modular agent architecture
* LLM-powered structured reasoning
* Resilient execution wrappers

This makes the system easy to extend with additional agents or alternative data sources.

---

## License

MIT License

---

## Acknowledgements

Built using:

* urlLangGraph[https://www.langchain.com/langgraph](https://www.langchain.com/langgraph)
* urlStreamlit[https://streamlit.io](https://streamlit.io)
* urlYahoo Finance API via yfinance[https://pypi.org/project/yfinance/](https://pypi.org/project/yfinance/)
* urlCoinGecko API[https://www.coingecko.com/en/api](https://www.coingecko.com/en/api)
* urlFRED API[https://fred.stlouisfed.org/docs/api/fred/](https://fred.stlouisfed.org/docs/api/fred/)
