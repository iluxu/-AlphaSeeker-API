# Pydantic models for /analyze endpoint
# app/models/analysis.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional

# --- Pydantic Models for Analysis ---

class IndicatorsData(BaseModel):
    RSI: Optional[float] = None
    ATR: Optional[float] = None
    SMA_50: Optional[float] = None
    SMA_200: Optional[float] = None
    EMA_12: Optional[float] = None
    EMA_26: Optional[float] = None
    MACD: Optional[float] = None
    Signal_Line: Optional[float] = None
    Bollinger_Upper: Optional[float] = None
    Bollinger_Middle: Optional[float] = None
    Bollinger_Lower: Optional[float] = None
    Momentum: Optional[float] = None
    Stochastic_K: Optional[float] = None
    Stochastic_D: Optional[float] = None
    Williams_R: Optional[float] = Field(None, alias="Williams_%R") # Handle alias if needed
    ADX: Optional[float] = None
    CCI: Optional[float] = None
    OBV: Optional[float] = None
    returns: Optional[float] = None
    # Use ConfigDict for Pydantic v2 compatibility if needed
    model_config = ConfigDict(populate_by_name=True, extra='allow')

class ModelOutputData(BaseModel):
    garchVolatility: Optional[float] = None
    var95: Optional[float] = None

class GptAnalysisText(BaseModel):
    technical_analysis: Optional[str] = None
    risk_assessment: Optional[str] = None
    market_outlook: Optional[str] = None
    raw_text: Optional[str] = None
    signal_evaluation: Optional[str] = None

class GptTradingParams(BaseModel):
    optimal_entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trade_direction: Optional[str] = None # Can be 'long', 'short', or 'hold'
    leverage: Optional[int] = Field(None, ge=1)
    position_size_usd: Optional[float] = Field(None, ge=0)
    estimated_profit: Optional[float] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=1)

class BacktestTradeAnalysis(BaseModel):
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: Optional[float] = None
    avg_profit: Optional[float] = None
    avg_loss: Optional[float] = None
    profit_factor: Optional[float] = None
    total_profit: Optional[float] = None
    largest_win: Optional[float] = None
    largest_loss: Optional[float] = None
    average_trade_duration: Optional[float] = None

class BacktestResultsData(BaseModel):
    strategy_score: Optional[float] = Field(None, ge=0, le=1)
    trade_analysis: Optional[BacktestTradeAnalysis] = None
    recommendation: Optional[str] = None
    warnings: List[str] = Field([])

class AnalysisRequest(BaseModel): # <<< --- THIS IS THE CLASS YOU NEED
    symbol: str = Field(..., example="BTC/USDT:USDT")
    timeframe: str = Field(default="1h", example="1h")
    lookback: int = Field(default=1000, ge=250)
    accountBalance: float = Field(default=1000.0, ge=0)
    maxLeverage: float = Field(default=10.0, ge=1)
    # Add other fields if they were part of the original request model

class AnalysisResponse(BaseModel):
    symbol: str
    timeframe: str
    currentPrice: Optional[float] = None
    indicators: Optional[IndicatorsData] = None
    modelOutput: Optional[ModelOutputData] = None
    gptParams: Optional[GptTradingParams] = None
    gptAnalysis: Optional[GptAnalysisText] = None
    backtest: Optional[BacktestResultsData] = None
    error: Optional[str] = None