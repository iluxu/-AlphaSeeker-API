# Pydantic models for /scan endpoint
# Example: app/models/scan.py
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict

# Define ScanRequest, ScanResultItem, ScanResponse models here...
class ScanRequest(BaseModel):
    ticker_start_index: Optional[int] = Field(default=0, ge=0)
    ticker_end_index: Optional[int] = Field(default=None, ge=0)
    timeframe: str = Field(default="1h", description="Candle timeframe (e.g., '1m', '5m', '1h').") # Default 1h might be more stable than 1m
    max_tickers: Optional[int] = Field(default=100)
    top_n: int = Field(default=10, ge=1)

    # Core Filters
    min_gpt_confidence: float = Field(default=0.60, ge=0, le=1) # Slightly relaxed default? User had 0.65
    min_backtest_score: float = Field(default=0.55, ge=0, le=1) # Slightly relaxed default? User had 0.60
    trade_direction: Optional[str] = Field(default=None, pattern="^(long|short)$")
    filter_by_btc_trend: Optional[bool] = Field(default=True)

    # Backtest Filters
    min_backtest_trades: Optional[int] = Field(default=10, ge=0) # Relaxed default? User had 15
    min_backtest_win_rate: Optional[float] = Field(default=0.50, ge=0, le=1) # Relaxed default? User had 0.52
    min_backtest_profit_factor: Optional[float] = Field(default=1.3, ge=0) # Relaxed default? User had 1.5

    # GPT/Risk Filter
    min_risk_reward_ratio: Optional[float] = Field(default=1.5, ge=0) # Relaxed default? User had 1.8

    # Indicator Filters
    min_adx: Optional[float] = Field(default=20.0, ge=0) # Relaxed default for signal trigger, user had 25
    require_sma_alignment: Optional[bool] = Field(default=True) # Keep user's strict alignment for now

    # Analysis Config
    lookback: int = Field(default=1000, ge=250) # User had 2000, 1000 is often enough unless specific long term needed
    accountBalance: float = Field(default=5000.0, ge=0)
    maxLeverage: float = Field(default=10.0, ge=1) # User had 20, 10 is safer default
    max_concurrent_tasks: int = Field(default=10, ge=1) # User had 16, adjust based on system/API limits

# ScanResultItem (Unchanged)
class ScanResultItem(BaseModel):
    rank: int; symbol: str; timeframe: str; currentPrice: Optional[float] = None
    gptConfidence: Optional[float] = None; backtestScore: Optional[float] = None; combinedScore: Optional[float] = None
    tradeDirection: Optional[str] = None; optimalEntry: Optional[float] = None; stopLoss: Optional[float] = None
    takeProfit: Optional[float] = None; gptAnalysisSummary: Optional[str] = None

# ScanResponse (Unchanged structure)
class ScanResponse(BaseModel):
    scan_parameters: ScanRequest; total_tickers_attempted: int; total_tickers_succeeded: int
    ticker_start_index: Optional[int] = Field(default=0, ge=0); ticker_end_index: Optional[int] = Field(default=None, ge=0)
    total_opportunities_found: int; top_opportunities: List[ScanResultItem]
    errors: Dict[str, str] = Field(default={})

# Remember to add imports like BaseModel, Field etc.