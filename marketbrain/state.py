import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict

class AgentSignal(TypedDict):
    agent : str
    signal : str
    confidence : float
    summary: str
    raw_dat: dict 

class MarketMindState(TypedDict):
    ticker : str
    asset_type :str

    agent_signals : Annotated[list, operator.add]

    final_verdict: Optional[str]
    final_confidence: Optional[float]
    final_reasoning: Optional[str]

    