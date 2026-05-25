Marketmind Professional Github Readme
MarketMind

An AI-powered multi-agent financial analysis platform for stocks and cryptocurrencies built with LangGraph, Streamlit, and LLM-based reasoning.

MarketMind orchestrates multiple specialized AI agents in parallel to generate market intelligence, risk assessments, sentiment analysis, macroeconomic insights, and final synthesized investment-style summaries.

Features
Multi-agent architecture using LangGraph
Parallel execution of financial analysis agents
Stock and cryptocurrency support
AI-generated market synthesis and reasoning
Risk analysis engine
Macro-economic analysis integration
On-chain crypto analytics
News and sentiment analysis
Interactive Streamlit dashboard
Fault-tolerant graph orchestration
Modular agent-based design for extensibility
Architecture
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
Project Structure
