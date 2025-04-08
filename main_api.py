# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import time
import ccxt
# import ccxt.async_support as ccxt_async
import logging
from datetime import datetime
from arch import arch_model
from sklearn.preprocessing import StandardScaler
# Silence TensorFlow warnings BEFORE importing Keras/TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# --- LSTM DISABLED: Comment out TF/Keras imports ---
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Input
# --- END LSTM DISABLED ---
import warnings
import openai
import json
import re
import asyncio
from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional
import uvicorn
from dotenv import load_dotenv
import sys

# --- Logging Configuration (Ensure this runs BEFORE any logging happens) ---
# Set level to INFO by default, easy to change to DEBUG if needed
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Use a stream handler to force output to console, overriding potential Uvicorn capture
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
log_handler = logging.StreamHandler(sys.stdout) # Force output to stdout
log_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(LOG_LEVEL)
# Remove existing handlers if any to avoid duplicates
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(log_handler)

# --- Test Log Message ---
logger.critical("--- Logging Initialized (Level: %s). Logs should now appear on the console. ---", LOG_LEVEL)


# --- LSTM DISABLED: Remove TF_LOCK or keep if other TF usage exists (GARCH doesn't use TF) ---
# TF_LOCK = asyncio.Lock()
# logging.info("TensorFlow Lock initialized.") # Commented out

# --- Configuration ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="ConvergenceWarning", category=UserWarning)

load_dotenv()

# --- LSTM DISABLED: Configuration no longer needed ---
# LSTM_TIME_STEPS = 60
# LSTM_EPOCHS = 15
# LSTM_BATCH_SIZE = 64
# --- END LSTM DISABLED ---

DEFAULT_MAX_CONCURRENT_TASKS = 5

# --- CCXT Initialization (Unchanged) ---
binance_futures = None
try:
    binance_futures = ccxt.binanceusdm({
        'enableRateLimit': True,
        'options': { 'adjustForTimeDifference': True },
        'timeout': 30000,
        'rateLimit': 200
    })
    logger.info("CCXT Binance Futures instance created.")
except Exception as e:
    logger.error(f"Error initializing CCXT: {e}", exc_info=True)
    binance_futures = None

# --- Load Markets Function (Unchanged) ---
async def load_exchange_markets(exchange):
    if not exchange: return False
    try:
        logger.info(f"Attempting to load markets for {exchange.id}...")
        markets = await asyncio.to_thread(exchange.load_markets, True)
        if markets:
             logger.info(f"Successfully loaded {len(markets)} markets for {exchange.id}.")
             return True
        else:
             logger.warning(f"Market loading returned empty for {exchange.id}.")
             return False
    except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
        logger.error(f"Failed to load markets for {exchange.id} due to network error: {e}", exc_info=False)
        return False
    except ccxt.ExchangeError as e:
        logger.error(f"Failed to load markets for {exchange.id} due to exchange error: {e}", exc_info=False)
        return False
    except Exception as e:
        logger.error(f"Unexpected error loading markets for {exchange.id}: {e}", exc_info=True)
        return False

# --- OpenAI Initialization (Unchanged) ---
openai_client = None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Use getenv
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable not found. GPT features will be disabled.")
else:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized.")
    except Exception as e:
        logger.error(f"Error initializing OpenAI: {e}", exc_info=True)
        openai_client = None

# --- Test OpenAI Connection (Unchanged) ---
async def test_openai_connection(client):
     if not client: return
     try:
         await asyncio.to_thread(client.models.list)
         logger.info("OpenAI connection test successful.")
     except Exception as e:
         logger.error(f"OpenAI connection test failed: {e}")

# --- FastAPI App (Unchanged) ---
app = FastAPI(
    title="Crypto Trading Analysis & Scanning API",
    description="API for technical analysis, GPT-driven strategy evaluation, backtesting, and market scanning.",
    version="1.3.0 Simplified" # Version bump
)

# --- Pydantic Models ---

# Tickers (Unchanged)
class TickerRequest(BaseModel): pass
class TickersResponse(BaseModel): tickers: List[str]

# AnalysisRequest - LSTM params removed
class AnalysisRequest(BaseModel):
    symbol: str = Field(..., example="BTC/USDT:USDT")
    timeframe: str = Field(default="1h", example="1h")
    lookback: int = Field(default=1000, ge=250)
    accountBalance: float = Field(default=1000.0, ge=0)
    maxLeverage: float = Field(default=10.0, ge=1)
    # --- LSTM DISABLED ---
    # lstm_time_steps: int = Field(default=LSTM_TIME_STEPS, ge=10)
    # lstm_epochs: int = Field(default=LSTM_EPOCHS, ge=1, le=100)
    # lstm_batch_size: int = Field(default=LSTM_BATCH_SIZE, ge=16)
    # --- END LSTM DISABLED ---

# IndicatorsData (Unchanged)
class IndicatorsData(BaseModel):
    RSI: Optional[float] = None; ATR: Optional[float] = None; SMA_50: Optional[float] = None; SMA_200: Optional[float] = None
    EMA_12: Optional[float] = None; EMA_26: Optional[float] = None; MACD: Optional[float] = None; Signal_Line: Optional[float] = None
    Bollinger_Upper: Optional[float] = None; Bollinger_Middle: Optional[float] = None; Bollinger_Lower: Optional[float] = None
    Momentum: Optional[float] = None; Stochastic_K: Optional[float] = None; Stochastic_D: Optional[float] = None
    Williams_R: Optional[float] = Field(None, alias="Williams_%R")
    ADX: Optional[float] = None; CCI: Optional[float] = None; OBV: Optional[float] = None; returns: Optional[float] = None
    model_config = ConfigDict(populate_by_name=True, extra='allow')

# ModelOutputData - LSTM removed
class ModelOutputData(BaseModel):
    # --- LSTM DISABLED ---
    # lstmForecast: Optional[float] = None
    # --- END LSTM DISABLED ---
    garchVolatility: Optional[float] = None
    var95: Optional[float] = None

# GptAnalysisText - LSTM analysis removed
class GptAnalysisText(BaseModel):
    # --- LSTM DISABLED ---
    # lstm_analysis: Optional[str] = None
    # --- END LSTM DISABLED ---
    technical_analysis: Optional[str] = None # Justification for evaluation
    risk_assessment: Optional[str] = None
    market_outlook: Optional[str] = None
    raw_text: Optional[str] = None
    signal_evaluation: Optional[str] = None # Added field for evaluation summary

# GptTradingParams (Unchanged, but interpretation changes)
class GptTradingParams(BaseModel):
    optimal_entry: Optional[float] = None; stop_loss: Optional[float] = None; take_profit: Optional[float] = None
    trade_direction: Optional[str] = None; leverage: Optional[int] = Field(None, ge=1)
    position_size_usd: Optional[float] = Field(None, ge=0); estimated_profit: Optional[float] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=1) # Confidence in the *evaluated* technical signal

# Backtest Models (Unchanged)
class BacktestTradeAnalysis(BaseModel):
    total_trades: int = 0; winning_trades: int = 0; losing_trades: int = 0; win_rate: Optional[float] = None
    avg_profit: Optional[float] = None; avg_loss: Optional[float] = None; profit_factor: Optional[float] = None
    total_profit: Optional[float] = None; largest_win: Optional[float] = None; largest_loss: Optional[float] = None
    average_trade_duration: Optional[float] = None

class BacktestResultsData(BaseModel):
    strategy_score: Optional[float] = Field(None, ge=0, le=1); trade_analysis: Optional[BacktestTradeAnalysis] = None
    recommendation: Optional[str] = None; warnings: List[str] = Field([])

# AnalysisResponse - Updated ModelOutput and GptAnalysis
class AnalysisResponse(BaseModel):
    symbol: str; timeframe: str; currentPrice: Optional[float] = None
    indicators: Optional[IndicatorsData] = None
    modelOutput: Optional[ModelOutputData] = None # LSTM removed
    gptParams: Optional[GptTradingParams] = None
    gptAnalysis: Optional[GptAnalysisText] = None # LSTM removed, evaluation added
    backtest: Optional[BacktestResultsData] = None
    error: Optional[str] = None

# ScanRequest - LSTM params removed
class ScanRequest(BaseModel):
    # Core Scan Params (Match user JSON where applicable)
    ticker_start_index: Optional[int] = Field(default=0, ge=0, description="0-based index to start scanning.")
    ticker_end_index: Optional[int] = Field(default=None, ge=0, description="0-based index to end scanning (exclusive).")
    timeframe: str = Field(default="1m", description="Candle timeframe (e.g., '1m', '5m', '1h').") # Default '1m' from user JSON
    max_tickers: Optional[int] = Field(default=100, description="Maximum tickers per run.") # Default 100 from user JSON
    top_n: int = Field(default=10, ge=1, description="Number of top results to return.")

    # Core Filters (Match user JSON)
    min_gpt_confidence: float = Field(default=0.65, ge=0, le=1) # From user JSON
    min_backtest_score: float = Field(default=0.60, ge=0, le=1) # From user JSON
    trade_direction: Optional[str] = Field(default=None, pattern="^(long|short)$") # null in user JSON -> default None
    
        # --- NEW BTC TREND FILTER ---
    filter_by_btc_trend: Optional[bool] = Field(default=True, description="If True, only show LONG signals if BTC is in Uptrend, and SHORT signals if BTC is in Downtrend (based on Price/SMA50/SMA200).")
    # --- END NEW BTC TREND FILTER ---
    
    
    # --- New Backtest Filters (from user JSON) ---
    min_backtest_trades: Optional[int] = Field(default=15, ge=0) # From user JSON
    min_backtest_win_rate: Optional[float] = Field(default=0.52, ge=0, le=1) # From user JSON
    min_backtest_profit_factor: Optional[float] = Field(default=1.5, ge=0) # From user JSON

    # --- New GPT/Risk Filter (from user JSON) ---
    min_risk_reward_ratio: Optional[float] = Field(default=1.8, ge=0) # From user JSON

    # --- New Indicator Filters (from user JSON) ---
    min_adx: Optional[float] = Field(default=25.0, ge=0) # From user JSON
    require_sma_alignment: Optional[bool] = Field(default=True) # From user JSON

    # Analysis Config (Match user JSON)
    lookback: int = Field(default=2000, ge=250) # From user JSON
    accountBalance: float = Field(default=5000.0, ge=0) # From user JSON
    maxLeverage: float = Field(default=20.0, ge=1) # From user JSON
    max_concurrent_tasks: int = Field(default=16, ge=1) # From user JSON

# ScanResultItem (Unchanged)
class ScanResultItem(BaseModel):
    rank: int; symbol: str; timeframe: str; currentPrice: Optional[float] = None
    gptConfidence: Optional[float] = None; backtestScore: Optional[float] = None; combinedScore: Optional[float] = None
    tradeDirection: Optional[str] = None; optimalEntry: Optional[float] = None; stopLoss: Optional[float] = None
    takeProfit: Optional[float] = None; gptAnalysisSummary: Optional[str] = None # Will now contain evaluation summary

# ScanResponse (Unchanged structure, data reflects changes)
class ScanResponse(BaseModel):
    scan_parameters: ScanRequest; total_tickers_attempted: int; total_tickers_succeeded: int
    ticker_start_index: Optional[int] = Field(default=0, ge=0); ticker_end_index: Optional[int] = Field(default=None, ge=0)
    total_opportunities_found: int; top_opportunities: List[ScanResultItem]
    errors: Dict[str, str] = Field(default={})


# --- Helper Functions ---

# --- Data Fetcher (Unchanged) ---
def get_real_time_data(symbol: str, timeframe: str = "1d", limit: int = 1000) -> pd.DataFrame:
    """Fetch OHLCV data. Raises exceptions on failure."""
    logger.debug(f"[{symbol}] Attempting to fetch {limit} candles for timeframe {timeframe}")
    if binance_futures is None: raise ConnectionError("CCXT exchange instance is not available.")
    if not binance_futures.markets:
         logger.warning(f"[{symbol}] Markets not loaded, attempting synchronous load...")
         try: binance_futures.load_markets(True); logger.info(f"[{symbol}] Markets loaded successfully (sync).")
         except Exception as e: raise ConnectionError(f"Failed to load markets synchronously: {e}") from e
    try:
        ohlcv = binance_futures.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            logger.warning(f"[{symbol}] No OHLCV data returned from fetch_ohlcv.")
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        # Keep rows with valid price/volume, even if some indicators are NaN later
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        if df.empty: logger.warning(f"[{symbol}] DataFrame became empty after type conversion/NaN drop.")
        else: logger.debug(f"[{symbol}] Fetched {len(df)} valid candles.")
        return df
    except ccxt.BadSymbol as e:
        logger.error(f"[{symbol}] Invalid symbol error: {e}")
        raise ValueError(f"Invalid symbol '{symbol}'") from e
    except ccxt.RateLimitExceeded as e:
        logger.warning(f"[{symbol}] Rate limit exceeded: {e}")
        raise ConnectionAbortedError(f"Rate limit exceeded") from e
    except ccxt.NetworkError as e:
        logger.error(f"[{symbol}] Network error: {e}")
        raise ConnectionError(f"Network error fetching {symbol}") from e
    except ccxt.AuthenticationError as e:
        logger.error(f"[{symbol}] Authentication error: {e}")
        raise PermissionError("CCXT Authentication Error") from e
    except Exception as e:
        logger.error(f"[{symbol}] Unexpected error fetching data: {e}", exc_info=True)
        raise RuntimeError(f"Failed to fetch data for {symbol}") from e


# --- Indicator Functions (compute_*, apply_technical_indicators - Unchanged) ---
def compute_rsi(series, window=14):
    delta = series.diff(); gain = delta.where(delta > 0, 0.0).fillna(0); loss = -delta.where(delta < 0, 0.0).fillna(0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean(); avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan); rsi = 100.0 - (100.0 / (1.0 + rs)); return rsi.fillna(100) # Fill NaN with 100 (or 0?) maybe fillna(50) better? Let's keep 100 for now.
def compute_atr(df, window=14):
    high_low = df['high'] - df['low']; high_close = abs(df['high'] - df['close'].shift()); low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False); atr = tr.ewm(com=window - 1, min_periods=window).mean(); return atr
def compute_bollinger_bands(series, window=20):
    sma = series.rolling(window=window, min_periods=window).mean(); std = series.rolling(window=window, min_periods=window).std()
    upper_band = sma + 2 * std; lower_band = sma - 2 * std; return upper_band, sma, lower_band
def compute_stochastic_oscillator(df, window=14, smooth_k=3):
    lowest_low = df['low'].rolling(window=window, min_periods=window).min(); highest_high = df['high'].rolling(window=window, min_periods=window).max()
    range_hh_ll = (highest_high - lowest_low).replace(0, np.nan); k_percent = 100 * ((df['close'] - lowest_low) / range_hh_ll)
    d_percent = k_percent.rolling(window=smooth_k, min_periods=smooth_k).mean(); return k_percent, d_percent
def compute_williams_r(df, window=14):
    highest_high = df['high'].rolling(window=window, min_periods=window).max(); lowest_low = df['low'].rolling(window=window, min_periods=window).min()
    range_ = (highest_high - lowest_low).replace(0, np.nan); williams_r = -100 * ((highest_high - df['close']) / range_); return williams_r
def compute_adx(df, window=14):
    df_adx = df.copy(); df_adx['H-L'] = df_adx['high'] - df_adx['low']; df_adx['H-PC'] = abs(df_adx['high'] - df_adx['close'].shift(1))
    df_adx['L-PC'] = abs(df_adx['low'] - df_adx['close'].shift(1)); df_adx['TR_calc'] = df_adx[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df_adx['DMplus'] = np.where((df_adx['high'] - df_adx['high'].shift(1)) > (df_adx['low'].shift(1) - df_adx['low']), df_adx['high'] - df_adx['high'].shift(1), 0)
    df_adx['DMplus'] = np.where(df_adx['DMplus'] < 0, 0, df_adx['DMplus'])
    df_adx['DMminus'] = np.where((df_adx['low'].shift(1) - df_adx['low']) > (df_adx['high'] - df_adx['high'].shift(1)), df_adx['low'].shift(1) - df_adx['low'], 0)
    df_adx['DMminus'] = np.where(df_adx['DMminus'] < 0, 0, df_adx['DMminus'])
    alpha = 1 / window; TR_smooth = df_adx['TR_calc'].ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    DMplus_smooth = df_adx['DMplus'].ewm(alpha=alpha, adjust=False, min_periods=window).mean(); DMminus_smooth = df_adx['DMminus'].ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    DIplus = 100 * (DMplus_smooth / TR_smooth.replace(0, np.nan)).fillna(0); DIminus = 100 * (DMminus_smooth / TR_smooth.replace(0, np.nan)).fillna(0)
    DIsum = (DIplus + DIminus).replace(0, np.nan); DX = 100 * (abs(DIplus - DIminus) / DIsum); ADX = DX.ewm(alpha=alpha, adjust=False, min_periods=window).mean(); return ADX
def compute_cci(df, window=20):
    tp = (df['high'] + df['low'] + df['close']) / 3; sma = tp.rolling(window=window, min_periods=window).mean()
    mad = tp.rolling(window=window, min_periods=window).apply(lambda x: np.nanmean(np.abs(x - np.nanmean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad.replace(0, np.nan)); return cci
def compute_obv(df):
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum(); return obv

def apply_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply technical indicators."""
    df_copy = df.copy()
    # Ensure 'close' is float before calculations
    df_copy['close'] = df_copy['close'].astype(float)
    df_copy['returns'] = np.log(df_copy['close'] / df_copy['close'].shift(1)).fillna(0)
    df_len = len(df_copy)
    symbol_log = df_copy.get('symbol', 'Unknown') # Get symbol if added earlier
    logger.debug(f"[{symbol_log}] Applying indicators to {df_len} candles.")

    def assign_if_enough_data(col_name, min_len_needed, calculation_lambda):
        if df_len >= min_len_needed:
             try: df_copy[col_name] = calculation_lambda()
             except Exception as e: logger.error(f"[{symbol_log}] Error calculating {col_name}: {e}", exc_info=False); df_copy[col_name] = np.nan
        else:
             logger.debug(f"[{symbol_log}] Skipping {col_name}, need {min_len_needed}, got {df_len}.")
             df_copy[col_name] = np.nan

    assign_if_enough_data('SMA_50', 50, lambda: df_copy['close'].rolling(window=50, min_periods=50).mean())
    assign_if_enough_data('SMA_200', 200, lambda: df_copy['close'].rolling(window=200, min_periods=200).mean())
    assign_if_enough_data('EMA_12', 26, lambda: df_copy['close'].ewm(span=12, adjust=False, min_periods=12).mean())
    assign_if_enough_data('EMA_26', 26, lambda: df_copy['close'].ewm(span=26, adjust=False, min_periods=26).mean())

    # MACD requires EMA_12 and EMA_26
    if df_len >= 26 and 'EMA_12' in df_copy and 'EMA_26' in df_copy and df_copy['EMA_12'].notna().any() and df_copy['EMA_26'].notna().any():
         df_copy['MACD'] = df_copy['EMA_12'] - df_copy['EMA_26']
         # Signal line requires MACD
         if df_len >= 35 and 'MACD' in df_copy and df_copy['MACD'].notna().any():
             assign_if_enough_data('Signal_Line', 35, lambda: df_copy['MACD'].ewm(span=9, adjust=False, min_periods=9).mean())
         else:
             logger.debug(f"[{symbol_log}] Skipping Signal_Line (req MACD/35 candles).")
             df_copy['Signal_Line'] = np.nan
    else:
        logger.debug(f"[{symbol_log}] Skipping MACD/Signal_Line (req EMAs/26 candles).")
        df_copy['MACD'], df_copy['Signal_Line'] = np.nan, np.nan

    assign_if_enough_data('RSI', 15, lambda: compute_rsi(df_copy['close'], window=14))
    assign_if_enough_data('ATR', 15, lambda: compute_atr(df_copy, window=14))

    if df_len >= 21:
        try:
            upper, middle, lower = compute_bollinger_bands(df_copy['close'], window=20)
            df_copy['Bollinger_Upper'], df_copy['Bollinger_Middle'], df_copy['Bollinger_Lower'] = upper, middle, lower
        except Exception as e:
            logger.error(f"[{symbol_log}] Error calculating Bollinger Bands: {e}", exc_info=False)
            df_copy['Bollinger_Upper'], df_copy['Bollinger_Middle'], df_copy['Bollinger_Lower'] = np.nan, np.nan, np.nan
    else:
        logger.debug(f"[{symbol_log}] Skipping Bollinger Bands (req 21 candles).")
        df_copy['Bollinger_Upper'], df_copy['Bollinger_Middle'], df_copy['Bollinger_Lower'] = np.nan, np.nan, np.nan

    assign_if_enough_data('Momentum', 11, lambda: df_copy['close'] - df_copy['close'].shift(10))

    if df_len >= 17:
        try:
            k, d = compute_stochastic_oscillator(df_copy, window=14, smooth_k=3)
            df_copy['Stochastic_K'], df_copy['Stochastic_D'] = k, d
        except Exception as e:
            logger.error(f"[{symbol_log}] Error calculating Stochastic: {e}", exc_info=False)
            df_copy['Stochastic_K'], df_copy['Stochastic_D'] = np.nan, np.nan
    else:
        logger.debug(f"[{symbol_log}] Skipping Stochastic (req 17 candles).")
        df_copy['Stochastic_K'], df_copy['Stochastic_D'] = np.nan, np.nan

    assign_if_enough_data('Williams_%R', 15, lambda: compute_williams_r(df_copy, window=14))
    assign_if_enough_data('ADX', 28, lambda: compute_adx(df_copy, window=14))
    assign_if_enough_data('CCI', 21, lambda: compute_cci(df_copy, window=20))
    assign_if_enough_data('OBV', 2, lambda: compute_obv(df_copy))

    logger.debug(f"[{symbol_log}] Finished applying indicators.")
    return df_copy


# --- Statistical Models (GARCH, VaR - LSTM functions removed) ---
def fit_garch_model(returns: pd.Series, symbol_log: str = "Unknown") -> Optional[float]:
    """Fit GARCH(1,1) model. Returns NEXT PERIOD conditional volatility."""
    valid_returns = returns.dropna() * 100
    logger.debug(f"[{symbol_log}] GARCH input len: {len(valid_returns)}")
    if len(valid_returns) < 50:
        logger.warning(f"[{symbol_log}] Skipping GARCH, need 50 returns, got {len(valid_returns)}.")
        return None
    try:
        # Model definition uses upgraded GARCH name
        am = arch_model(valid_returns, vol='GARCH', p=1, q=1, dist='Normal')
        # Suppress convergence warnings during fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(update_freq=0, disp='off', show_warning=False)
        if res.convergence_flag == 0:
            forecasts = res.forecast(horizon=1, reindex=False)
            cond_vol_forecast = np.sqrt(forecasts.variance.iloc[-1, 0]) / 100.0 # Convert back from variance %^2
            logger.debug(f"[{symbol_log}] GARCH fit successful. Forecast Vol: {cond_vol_forecast:.4f}")
            return float(cond_vol_forecast) if np.isfinite(cond_vol_forecast) else None
        else:
            logger.warning(f"[{symbol_log}] GARCH did not converge (Flag: {res.convergence_flag}).")
            return None
    except Exception as e:
        logger.error(f"[{symbol_log}] GARCH fitting error: {e}", exc_info=False) # Keep logs cleaner
        return None

def calculate_var(returns: pd.Series, confidence_level: float = 0.95, symbol_log: str = "Unknown") -> Optional[float]:
    """Calculate Value at Risk (VaR) at specified confidence level."""
    valid_returns = returns.dropna()
    logger.debug(f"[{symbol_log}] VaR input len: {len(valid_returns)}")
    if len(valid_returns) < 20:
        logger.warning(f"[{symbol_log}] Skipping VaR, need 20 returns, got {len(valid_returns)}.")
        return None
    try:
        var_value = np.percentile(valid_returns, (1.0 - confidence_level) * 100.0)
        logger.debug(f"[{symbol_log}] VaR calculated: {var_value:.4f} at {confidence_level*100:.0f}% confidence.")
        # VaR is usually negative, representing loss
        return float(var_value) if np.isfinite(var_value) else None
    except Exception as e:
        logger.error(f"[{symbol_log}] Error calculating VaR: {e}", exc_info=False)
        return None

# --- LSTM Functions Removed ---
# def prepare_lstm_data(...)
# def build_lstm_model(...)
# def train_lstm_model(...)
# def forecast_with_lstm(...)
# --- END LSTM REMOVED ---


# --- GPT Integration (Prompt Heavily Modified) ---

def gpt_generate_trading_parameters(
    df_with_indicators: pd.DataFrame,
    symbol: str,
    timeframe: str,
    account_balance: float,
    max_leverage: float,
    garch_volatility: Optional[float],
    var95: Optional[float],
    technically_derived_direction: str,
    min_requested_rr: Optional[float] # NEW: Pass the direction from RSI check
) -> str:
    """Generate trading parameters using GPT to EVALUATE a technical signal."""
    log_prefix = f"[{symbol} ({timeframe}) GPT]"
    if openai_client is None:
        logger.warning(f"{log_prefix} OpenAI client not available.")
        return json.dumps({"error": "OpenAI client not available"})

    df_valid = df_with_indicators.dropna(subset=['close', 'RSI', 'ATR']) # Ensure core indicators are present
    if df_valid.empty:
        logger.warning(f"{log_prefix} No valid data (Close/RSI/ATR) for GPT.")
        return json.dumps({"error": "Insufficient indicator data for GPT"})

    latest_data = df_valid.iloc[-1].to_dict()
    technical_indicators = {}

    # Extract finite indicators for context
    for field_name, model_field in IndicatorsData.model_fields.items():
        key_in_df = model_field.alias or field_name
        value = latest_data.get(key_in_df)
        if pd.notna(value) and np.isfinite(value):
             if abs(value) >= 1e4 or (abs(value) < 1e-4 and value != 0):
                 technical_indicators[key_in_df] = f"{value:.3e}"
             else:
                 technical_indicators[key_in_df] = round(float(value), 4)

    current_price = latest_data.get('close')
    if current_price is None or not np.isfinite(current_price):
         logger.error(f"{log_prefix} Missing current price for GPT context")
         return json.dumps({"error": "Missing current price for GPT context"})
    current_price = round(float(current_price), 4)

    garch_vol_str = f"{garch_volatility:.4%}" if garch_volatility is not None else "N/A"
    var95_str = f"{var95:.4%}" if var95 is not None else "N/A"

    # Prepare market context for GPT
    market_info = {
        "symbol": symbol,
        "timeframe": timeframe,
        "current_price": current_price,
        "garch_forecast_volatility": garch_vol_str,
        "value_at_risk_95": var95_str,
        "key_technical_indicators": technical_indicators,
        "potential_signal_direction": technically_derived_direction # Include the signal
    }
    data_json = json.dumps(market_info, indent=2)
    logger.debug(f"{log_prefix} Data prepared for GPT:\n{data_json}")

    # --- REVISED GPT PROMPT ---
    prompt = f"""You are a cryptocurrency trading analyst evaluating a potential trade setup.
A technical signal (RSI + basic SMA trend filter) suggests a potential trade direction: '{technically_derived_direction}'.
Your task is to EVALUATE this signal using the provided market data and technical indicators, and provide actionable parameters if appropriate.

Market Data & Indicators:
{data_json}

Instructions:
1.  **Evaluate Signal:** Assess the provided '{technically_derived_direction}' signal. Look for **confirming** factors (e.g., MACD alignment, price near relevant Bollinger Band, supportive Stochastic/CCI levels) and **strong contradicting** factors (e.g., clear divergence, price hitting major resistance/support against the signal, very low ADX < 15-20 suggesting chop).
2.  **Determine Action:**
    *   If the initial signal has some confirmation OR lacks strong contradictions, **lean towards confirming the `trade_direction`** ('long' or 'short').
    *   Only output `trade_direction: 'hold'` if there are **significant contradictions** from multiple important indicators OR if the market context (e.g., extreme chop, major news event imminent - though you don't have news data) makes the signal very unreliable.
3.  **Refine Parameters (if action suggested):**
    *   `optimal_entry`: Suggest a tactical entry, considering pullbacks/rallies to support/resistance (SMAs, BBands) or breakout/down levels relative to the signal candle. Justify briefly. Use `current_price` only if no better tactical level is apparent.
    *   `stop_loss`: Place logically based on volatility (ATR) or structure.
    # <<< --- CHANGE THIS LINE --- >>>
    *   `take_profit`: Aim for R/R >= {min_requested_rr or 1.5}.
    # <<< --- END OF CHANGE --- >>>
4.  **Provide Confidence:** Assign a `confidence_score` (0.0-1.0) based on the *degree of confirmation* and the *absence of strong contradictions*. A score > 0.6 requires decent confirmation.
5.  **Justify:** Explain your reasoning in the `analysis` sections (`signal_evaluation`, `technical_analysis`, `risk_assessment`, `market_outlook`).

Respond ONLY with a single, valid JSON object containing the specified fields.
"""
    # --- END OF REVISED PROMPT ---

    try:
        logger.info(f"{log_prefix} Sending request to GPT to evaluate '{technically_derived_direction}' signal...")
        response = openai_client.chat.completions.create(
            model="gpt-4o", # Use a capable model
            messages=[
                {"role": "system", "content": "You are a crypto trading analyst evaluating technical signals provided by a user. Respond in JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3, # Lower temperature for more deterministic analysis
            max_tokens=1800 # Allow slightly more tokens for detailed justification
        )
        gpt_output = response.choices[0].message.content
        logger.debug(f"{log_prefix} Raw GPT Output received:\n```json\n{gpt_output}\n```") # Log raw output at DEBUG level
        return gpt_output or "{}"
    except openai.RateLimitError as e:
        logger.error(f"{log_prefix} OpenAI Rate Limit Error: {e}")
        return json.dumps({"error": "OpenAI API rate limit exceeded", "details": str(e)})
    except openai.APIError as e:
        logger.error(f"{log_prefix} OpenAI API Error: {e}", exc_info=False)
        return json.dumps({"error": "OpenAI API error", "details": str(e)})
    except Exception as e:
        logger.error(f"{log_prefix} Error querying OpenAI: {e}", exc_info=False)
        return json.dumps({"error": "Failed to query OpenAI", "details": str(e)})


# --- Parse GPT Parameters (Modified to handle new 'analysis' structure) ---
def parse_gpt_trading_parameters(gpt_output_str: str, symbol_for_log: str = "") -> Dict[str, Any]:
    """Parse GPT-generated trading parameters (evaluation focused)."""
    log_prefix = f"[{symbol_for_log} Parse]"
    # Default structure, including new analysis fields
    parsed_data = {
        'optimal_entry': None, 'stop_loss': None, 'take_profit': None, 'trade_direction': 'hold', # Default to hold
        'leverage': None, 'position_size_usd': None, 'estimated_profit': None, 'confidence_score': 0.0, # Default confidence 0
        'analysis': {'signal_evaluation': None, 'technical_analysis': None, 'risk_assessment': None, 'market_outlook': None, 'raw_text': gpt_output_str}
    }
    try:
        data = json.loads(gpt_output_str)
        if not isinstance(data, dict):
            raise json.JSONDecodeError("GPT output was not a JSON object", gpt_output_str, 0)
        logger.debug(f"{log_prefix} Successfully decoded JSON from GPT.")

        # Helper to safely get float values
        def get_float(key):
            val = data.get(key)
            return float(val) if isinstance(val, (int, float)) and np.isfinite(val) else None

        # Get core trade parameters
        parsed_data['optimal_entry'] = get_float('optimal_entry')
        parsed_data['stop_loss'] = get_float('stop_loss')
        parsed_data['take_profit'] = get_float('take_profit')
        parsed_data['position_size_usd'] = get_float('position_size_usd')
        parsed_data['estimated_profit'] = get_float('estimated_profit')
        parsed_data['confidence_score'] = get_float('confidence_score')

        # Leverage
        leverage_val = data.get('leverage')
        if isinstance(leverage_val, int) and leverage_val >= 1:
            parsed_data['leverage'] = leverage_val
        elif leverage_val is not None:
            logger.warning(f"{log_prefix} Invalid leverage value from GPT: {leverage_val}.")

        # Trade Direction (Crucial - reflects GPT's evaluation)
        direction = data.get('trade_direction')
        if direction in ['long', 'short', 'hold']:
            parsed_data['trade_direction'] = direction
            logger.info(f"{log_prefix} GPT evaluated signal resulted in direction: '{direction}'")
        elif direction:
            logger.warning(f"{log_prefix} Invalid trade_direction from GPT: '{direction}'. Defaulting to 'hold'.")
            parsed_data['trade_direction'] = 'hold'
        else:
             logger.warning(f"{log_prefix} Missing trade_direction from GPT. Defaulting to 'hold'.")
             parsed_data['trade_direction'] = 'hold'


        # Validate parameters *if* GPT suggests a trade
        if parsed_data['trade_direction'] in ['long', 'short']:
             if not all([parsed_data['optimal_entry'], parsed_data['stop_loss'], parsed_data['take_profit'], parsed_data['confidence_score'] is not None]):
                 logger.warning(f"{log_prefix} GPT suggested '{parsed_data['trade_direction']}' but missing Entry/SL/TP/Confidence. Forcing 'hold'.")
                 parsed_data['trade_direction'] = 'hold'
             # Optional: Check R/R ratio validity here
             elif parsed_data['optimal_entry'] and parsed_data['stop_loss'] and parsed_data['take_profit']:
                 risk = abs(parsed_data['optimal_entry'] - parsed_data['stop_loss'])
                 reward = abs(parsed_data['take_profit'] - parsed_data['optimal_entry'])
                 if risk < 1e-9 or reward / risk < 1.0: # Ensure R > 0 and R/R >= 1.0 at least
                      logger.warning(f"{log_prefix} GPT suggested '{parsed_data['trade_direction']}' with invalid R/R ({reward=}, {risk=}). Forcing 'hold'.")
                      parsed_data['trade_direction'] = 'hold'


        # Process analysis section - handle new structure
        analysis_dict = data.get('analysis')
        if isinstance(analysis_dict, dict):
            for key in ['signal_evaluation', 'technical_analysis', 'risk_assessment', 'market_outlook']:
                val = analysis_dict.get(key)
                if isinstance(val, str) and val.strip():
                    parsed_data['analysis'][key] = val.strip()
                elif val is not None:
                    logger.warning(f"{log_prefix} Invalid type or empty value for analysis key '{key}': {type(val)}.")
        elif analysis_dict is not None:
            logger.warning(f"{log_prefix} Invalid type for 'analysis' section: {type(analysis_dict)}.")

        # Add raw text if analysis section was missing/invalid
        if parsed_data['analysis']['signal_evaluation'] is None:
             parsed_data['analysis']['raw_text'] = gpt_output_str # Keep raw if parsing failed

    except json.JSONDecodeError as e:
        logger.error(f"{log_prefix} Failed to decode JSON from GPT: {e}. Raw: {gpt_output_str[:300]}...")
        parsed_data['trade_direction'] = 'hold'
        parsed_data['analysis']['signal_evaluation'] = f"Error: Failed to parse GPT JSON response. {e}"
    except Exception as e:
        logger.error(f"{log_prefix} Unexpected error parsing GPT response: {e}", exc_info=True)
        parsed_data['trade_direction'] = 'hold'
        parsed_data['analysis']['signal_evaluation'] = f"Error: Unexpected error parsing GPT response. {e}"

    # Clear trade params if final decision is 'hold'
    if parsed_data['trade_direction'] == 'hold':
        logger.info(f"{log_prefix} Final direction is 'hold', clearing trade parameters.")
        parsed_data['optimal_entry'] = None
        parsed_data['stop_loss'] = None
        parsed_data['take_profit'] = None
        parsed_data['leverage'] = None
        parsed_data['position_size_usd'] = None
        parsed_data['estimated_profit'] = None
        # Keep confidence score as it might reflect confidence in *not* trading

    logger.debug(f"{log_prefix} Parsed GPT Params: {parsed_data}")
    return parsed_data


# --- Backtesting (Unchanged - still uses RSI trigger for historical simulation) ---
def backtest_strategy(df_with_indicators: pd.DataFrame, gpt_params: Dict[str, Any]) -> Dict[str, Any]:
    """Backtest strategy based on *similar historical conditions* (RSI Trigger)."""
    symbol_log = df_with_indicators.get('symbol', 'Unknown')
    log_prefix = f"[{symbol_log} Backtest]"
    results = {'strategy_score': 0.0, 'trade_analysis': {'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': None, 'avg_profit': None, 'avg_loss': None, 'profit_factor': None, 'total_profit': None, 'largest_win': None, 'largest_loss': None, 'average_trade_duration': None}, 'recommendation': 'N/A', 'warnings': []}
    trade_analysis = results['trade_analysis']

    # Backtest uses the direction *evaluated* by GPT for consistency check
    direction_evaluated = gpt_params.get('trade_direction')
    if direction_evaluated not in ['long', 'short']:
        results['recommendation'] = f"Backtest skipped: Evaluated trade direction is '{direction_evaluated}'."
        logger.info(f"{log_prefix} {results['recommendation']}")
        return results

    required_cols = ['close', 'low', 'high', 'ATR', 'RSI'] # Core columns for the simple RSI strategy
    df_clean = df_with_indicators[required_cols].dropna()
    if len(df_clean) < 50:
        results['recommendation'] = "Backtest skipped: Insufficient data."
        results['warnings'].append(f"Insufficient data points ({len(df_clean)}) for backtest.")
        logger.warning(f"{log_prefix} {results['recommendation']}")
        return results
    logger.info(f"{log_prefix} Starting backtest for direction '{direction_evaluated}' over {len(df_clean)} bars.")

    # Backtest parameters (using the simplified RSI entry logic)
    min_rr_ratio_target = 1.5 # Minimum R:R for a simulated trade
    rsi_oversold = 35 # Corresponds to the technical trigger
    rsi_overbought = 65 # Corresponds to the technical trigger
    atr_sl_multiplier = 1.5 # Stop loss based on ATR at signal time
    max_trade_duration_bars = 96 # ~4 days on 1h timeframe
    min_bars_between_trades = 5 # Cooldown period

    trades = []; last_entry_index = -1 - min_bars_between_trades
    # Ensure indices are valid and align with df_clean
    try:
        search_start_index = df_clean.index.min()
        search_end_index_ts = df_clean.index[-6] # Stop a few bars before the end
        search_indices = df_with_indicators.loc[search_start_index:search_end_index_ts].index
        if len(search_indices) < 2: # Need at least start and next bar
             raise ValueError("Not enough bars for simulation loop")
        logger.debug(f"{log_prefix} Simulation loop range: {search_indices[0]} to {search_indices[-1]} ({len(search_indices)} bars)")
    except Exception as e:
        results['recommendation'] = "Backtest skipped: Cannot determine simulation range."
        results['warnings'].append(f"Error setting up simulation index: {e}")
        logger.error(f"{log_prefix} {results['recommendation']} - {e}")
        return results

    # --- Simulation Loop ---
    entry_count = 0
    for i_idx, current_ts in enumerate(search_indices[:-1]): # Iterate up to second to last timestamp
        signal_row = df_with_indicators.loc[current_ts]
        entry_ts = search_indices[i_idx+1] # Timestamp of the next bar
        entry_row = df_with_indicators.loc[entry_ts]

        # Check cooldown
        current_loc = df_with_indicators.index.get_loc(current_ts)
        if current_loc <= last_entry_index + min_bars_between_trades:
             continue

        # Check data validity for signal and entry bars
        if pd.isna(signal_row['RSI']) or pd.isna(signal_row['ATR']) or pd.isna(signal_row['close']) or \
           pd.isna(entry_row['open']) or signal_row['ATR'] <= 1e-9:
            continue

        current_rsi, current_atr, signal_close = signal_row['RSI'], signal_row['ATR'], signal_row['close']
        entry_price = entry_row['open'] # Enter at open of next bar

        setup_found, stop_loss_calc, take_profit_calc = False, 0.0, 0.0

        # Check for RSI-based entry signal (matching the backtest logic)
        if direction_evaluated == 'long' and current_rsi < rsi_oversold:
            stop_loss_calc = signal_close - current_atr * atr_sl_multiplier # SL based on signal bar's close/ATR
            risk_amount = abs(entry_price - stop_loss_calc)
            if risk_amount > 1e-9:
                take_profit_calc = entry_price + risk_amount * min_rr_ratio_target
                setup_found = True
                logger.debug(f"{log_prefix} Long setup found at {current_ts} (RSI: {current_rsi:.2f}). Entry: {entry_price:.4f}, SL: {stop_loss_calc:.4f}, TP: {take_profit_calc:.4f}")
        elif direction_evaluated == 'short' and current_rsi > rsi_overbought:
            stop_loss_calc = signal_close + current_atr * atr_sl_multiplier
            risk_amount = abs(stop_loss_calc - entry_price)
            if risk_amount > 1e-9:
                take_profit_calc = entry_price - risk_amount * min_rr_ratio_target
                setup_found = True
                logger.debug(f"{log_prefix} Short setup found at {current_ts} (RSI: {current_rsi:.2f}). Entry: {entry_price:.4f}, SL: {stop_loss_calc:.4f}, TP: {take_profit_calc:.4f}")

        if setup_found:
            # Validate levels make sense (SL < Entry < TP for long, TP < Entry < SL for short)
            valid_levels = False
            if direction_evaluated == 'long' and stop_loss_calc < entry_price < take_profit_calc: valid_levels = True
            elif direction_evaluated == 'short' and take_profit_calc < entry_price < stop_loss_calc: valid_levels = True

            if valid_levels:
                entry_count += 1
                outcome, exit_price, exit_idx_loc = None, None, -1
                entry_idx_loc = df_with_indicators.index.get_loc(entry_ts)

                # Simulate trade progression bar-by-bar
                max_exit_loc = min(len(df_with_indicators) - 1, entry_idx_loc + max_trade_duration_bars)
                for k_loc in range(entry_idx_loc + 1, max_exit_loc + 1): # Check from bar *after* entry open
                    current_bar = df_with_indicators.iloc[k_loc]
                    current_low, current_high = current_bar['low'], current_bar['high']

                    if direction_evaluated == 'long':
                        if current_low <= stop_loss_calc: # Stop Loss Hit
                            outcome, exit_price, exit_idx_loc = 'loss', stop_loss_calc, k_loc
                            logger.debug(f"{log_prefix} Trade {entry_count} (Long) SL hit at {df_with_indicators.index[k_loc]}")
                            break
                        elif current_high >= take_profit_calc: # Take Profit Hit
                            outcome, exit_price, exit_idx_loc = 'win', take_profit_calc, k_loc
                            logger.debug(f"{log_prefix} Trade {entry_count} (Long) TP hit at {df_with_indicators.index[k_loc]}")
                            break
                    elif direction_evaluated == 'short':
                        if current_high >= stop_loss_calc: # Stop Loss Hit
                            outcome, exit_price, exit_idx_loc = 'loss', stop_loss_calc, k_loc
                            logger.debug(f"{log_prefix} Trade {entry_count} (Short) SL hit at {df_with_indicators.index[k_loc]}")
                            break
                        elif current_low <= take_profit_calc: # Take Profit Hit
                            outcome, exit_price, exit_idx_loc = 'win', take_profit_calc, k_loc
                            logger.debug(f"{log_prefix} Trade {entry_count} (Short) TP hit at {df_with_indicators.index[k_loc]}")
                            break

                # If neither SL nor TP hit within duration, exit at close of last bar
                if outcome is None:
                    exit_idx_loc = max_exit_loc
                    exit_bar = df_with_indicators.iloc[exit_idx_loc]
                    exit_price = exit_bar['close']
                    # Determine win/loss based on exit price relative to entry
                    if direction_evaluated == 'long': outcome = 'win' if exit_price > entry_price else 'loss'
                    else: outcome = 'win' if exit_price < entry_price else 'loss'
                    logger.debug(f"{log_prefix} Trade {entry_count} exited due to duration at {df_with_indicators.index[exit_idx_loc]} - Outcome: {outcome}")


                # Record the trade if an outcome was determined
                if outcome and exit_price is not None and exit_idx_loc != -1:
                    profit_points = (exit_price - entry_price) if direction_evaluated == 'long' else (entry_price - exit_price)
                    trade_duration = exit_idx_loc - entry_idx_loc # Duration in bars
                    trades.append({'profit': profit_points, 'duration': trade_duration, 'outcome': outcome})
                    last_entry_index = entry_idx_loc # Update cooldown index (use location)
            else:
                 logger.debug(f"{log_prefix} Setup found at {current_ts} but levels invalid (Entry:{entry_price:.4f}, SL:{stop_loss_calc:.4f}, TP:{take_profit_calc:.4f})")


    # --- Analyze Backtest Results ---
    trade_analysis['total_trades'] = len(trades)
    logger.info(f"{log_prefix} Backtest simulation completed. Found {len(trades)} historical trades.")
    if trades:
        # --- Calculations (Unchanged) ---
        profits = np.array([t['profit'] for t in trades]); durations = np.array([t['duration'] for t in trades]); outcomes = [t['outcome'] for t in trades]
        trade_analysis['winning_trades'] = sum(1 for o in outcomes if o == 'win'); trade_analysis['losing_trades'] = len(trades) - trade_analysis['winning_trades']
        if len(trades) > 0: trade_analysis['win_rate'] = round(trade_analysis['winning_trades'] / len(trades), 3)
        winning_profits = profits[profits > 0]; losing_profits = profits[profits <= 0]; gross_profit = np.sum(winning_profits); gross_loss = abs(np.sum(losing_profits))
        trade_analysis['avg_profit'] = round(np.mean(winning_profits), 4) if len(winning_profits) > 0 else 0.0
        trade_analysis['avg_loss'] = round(abs(np.mean(losing_profits)), 4) if len(losing_profits) > 0 else 0.0
        if gross_loss > 1e-9: trade_analysis['profit_factor'] = round(gross_profit / gross_loss, 2)
        elif gross_profit > 1e-9: trade_analysis['profit_factor'] = float('inf')
        else: trade_analysis['profit_factor'] = 0.0
        trade_analysis['total_profit'] = round(np.sum(profits), 4); trade_analysis['largest_win'] = round(np.max(winning_profits), 4) if len(winning_profits) > 0 else 0.0
        trade_analysis['largest_loss'] = round(np.min(losing_profits), 4) if len(losing_profits) > 0 else 0.0; trade_analysis['average_trade_duration'] = round(np.mean(durations), 1)

        # --- Calculate Strategy Score (Unchanged logic, applied to RSI strategy results) ---
        score = 0.0; pf = trade_analysis['profit_factor']; wr = trade_analysis['win_rate']; num_trades = len(trades)
        avg_w = trade_analysis['avg_profit']; avg_l = trade_analysis['avg_loss']
        if pf is not None:
            if pf == float('inf') or pf >= 2.5: score += 0.40
            elif pf >= 1.7: score += 0.30
            elif pf >= 1.3: score += 0.20
            elif pf >= 1.0: score += 0.10
            else: score -= 0.20
        if wr is not None:
            if wr >= 0.6: score += 0.30
            elif wr >= 0.5: score += 0.20
            elif wr >= 0.4: score += 0.10
            else: score -= 0.10
        if num_trades >= 30: score += 0.20
        elif num_trades >= 15: score += 0.10
        elif num_trades < 5: score -= 0.15
        if avg_w is not None and avg_l is not None and avg_l > 1e-9:
             ratio = avg_w / avg_l
             if ratio >= 1.5: score += 0.10
             elif ratio >= 1.0: score += 0.05
        if trade_analysis['total_profit'] is not None and trade_analysis['total_profit'] <= 0: score -= 0.30
        results['strategy_score'] = max(0.0, min(1.0, round(score, 2)))
        logger.info(f"{log_prefix} Backtest Score: {results['strategy_score']:.2f} (Trades:{num_trades}, WR:{wr:.1%}, PF:{pf:.2f})")

        # --- Recommendation and Warnings (Unchanged) ---
        if results['strategy_score'] >= 0.75: results['recommendation'] = "Strong historical performance for RSI trigger."
        elif results['strategy_score'] >= 0.60: results['recommendation'] = "Good historical performance for RSI trigger."
        elif results['strategy_score'] >= 0.45: results['recommendation'] = "Moderate/Mixed historical performance for RSI trigger."
        else: results['recommendation'] = "Poor historical performance for RSI trigger."
        if wr is not None and wr < 0.40: results['warnings'].append(f"Low Win Rate ({wr:.1%})")
        if pf is not None and pf < 1.2: results['warnings'].append(f"Low Profit Factor ({pf:.2f})")
        if num_trades < 10: results['warnings'].append(f"Low Trade Count ({num_trades}).")
        if trade_analysis['total_profit'] is not None and trade_analysis['total_profit'] <= 0: results['warnings'].append("Overall loss in backtest.")
        if avg_l is not None and trade_analysis['largest_loss'] is not None and abs(trade_analysis['largest_loss']) > 3 * avg_l: results['warnings'].append("Largest loss significantly exceeds average loss.")
    else:
        results['recommendation'] = "No similar historical RSI setups found to backtest."; results['warnings'].append("No qualifying historical trade setups found."); results['strategy_score'] = 0.0
        logger.info(f"{log_prefix} {results['recommendation']}")

    return results


# --- Core Analysis Function (LSTM Disabled, Technical Trigger Added) ---
# --- Core Analysis Function (perform_single_analysis) ---
async def perform_single_analysis(
    symbol: str, timeframe: str, lookback: int, account_balance: float, max_leverage: float, min_requested_rr: Optional[float] = None
) -> AnalysisResponse:
    start_time_ticker = time.time()
    log_prefix = f"[{symbol} ({timeframe})]"
    logger.info(f"{log_prefix} Starting analysis...")
    analysis_result = AnalysisResponse(symbol=symbol, timeframe=timeframe)

    # ... (Steps 1: Fetch Data, Step 2: Apply Indicators - remain the same) ...
    # --- Make sure these steps complete successfully ---
    try:
        # Step 1: Fetch Data
        df_raw = await asyncio.to_thread(get_real_time_data, symbol, timeframe, lookback)
        min_req = 200
        if df_raw.empty or len(df_raw) < min_req:
            analysis_result.error = f"Insufficient data ({len(df_raw)}/{min_req})"; logger.warning(f"{log_prefix} {analysis_result.error}"); return analysis_result
        analysis_result.currentPrice = float(df_raw['close'].iloc[-1])
        logger.info(f"{log_prefix} Data fetched. Price: {analysis_result.currentPrice:.4f}")

        # Step 2: Apply Indicators
        df_raw['symbol'] = symbol
        df_indicators = await asyncio.to_thread(apply_technical_indicators, df_raw)
        latest_indicators_raw = df_indicators.iloc[-1].to_dict()
        indicator_payload = {
            field_name: float(value)
            for field_name, model_field in IndicatorsData.model_fields.items()
            if pd.notna(value := latest_indicators_raw.get(model_field.alias or field_name)) and np.isfinite(value)
        }
        analysis_result.indicators = IndicatorsData(**indicator_payload)
        logger.info(f"{log_prefix} Indicators applied.")

    except Exception as e:
        logger.error(f"{log_prefix} Error during Data/Indicator steps: {e}", exc_info=True)
        analysis_result.error = f"Data/Indicator Error: {e}"
        return analysis_result # Stop if basic data/indicators fail


    # --- Step 2.5: Determine Technical Trigger (RSI ONLY - VERY SIMPLE) ---
    technical_signal_direction = "hold" # Default
    latest_rsi = analysis_result.indicators.RSI if analysis_result.indicators else None
    rsi_oversold_trigger = 35
    rsi_overbought_trigger = 65

    # Determine Trigger based ONLY on RSI level
    if latest_rsi is not None and np.isfinite(latest_rsi):
        if latest_rsi < rsi_oversold_trigger:
            technical_signal_direction = "long"
        elif latest_rsi > rsi_overbought_trigger:
            technical_signal_direction = "short"

    # Log the outcome
    rsi_str = f'{latest_rsi:.2f}' if latest_rsi is not None else 'N/A'
    logger.info(f"{log_prefix} Technical signal trigger: {technical_signal_direction.upper()} (RSI: {rsi_str})")
    # --- End of Step 2.5 ---


    # Step 3: Statistical Models (GARCH, VaR)
    garch_vol, var95 = None, None
    try:
        if df_indicators is not None and 'returns' in df_indicators:
            returns = df_indicators['returns']
            # --- Run concurrently for slight speed up ---
            garch_task = asyncio.to_thread(fit_garch_model, returns, symbol)
            var_task = asyncio.to_thread(calculate_var, returns, 0.95, symbol)
            garch_vol, var95 = await asyncio.gather(garch_task, var_task)
            # --- End concurrent run ---
            analysis_result.modelOutput = ModelOutputData(garchVolatility=garch_vol, var95=var95)
            logger.info(f"{log_prefix} GARCH/VaR calculated (Vol: {garch_vol}, VaR95: {var95}).")
        else:
            logger.warning(f"{log_prefix} Skipping GARCH/VaR (no returns data).")
            analysis_result.modelOutput = ModelOutputData() # Ensure modelOutput exists
    except Exception as e:
        logger.error(f"{log_prefix} Stat model error: {e}", exc_info=True)
        analysis_result.error = (analysis_result.error or "") + f"; Stat Model Error: {e}"
        analysis_result.modelOutput = ModelOutputData() # Ensure modelOutput exists


    # --- Step 4: GPT Evaluation (Conditional Call) ---
    gpt_parsed_output = None
    if openai_client and technical_signal_direction != 'hold':
        logger.info(f"{log_prefix} Technical signal is '{technical_signal_direction}', proceeding with GPT evaluation.")
        try:
            # <<< --- MODIFY THIS CALL --- >>>
            gpt_raw_output = await asyncio.to_thread(
                gpt_generate_trading_parameters,
                df_indicators, symbol, timeframe, account_balance, max_leverage,
                garch_vol, var95, # Pass stat model results
                technical_signal_direction, # Pass the RSI-derived signal
                min_requested_rr # Pass the R/R value received by this function
            )
            logger.debug(f"{log_prefix} RAW GPT Response Received:\n{gpt_raw_output}")
            gpt_parsed_output = parse_gpt_trading_parameters(gpt_raw_output, symbol)

            if gpt_parsed_output:
                 analysis_result.gptParams = GptTradingParams(**{k: v for k, v in gpt_parsed_output.items() if k != 'analysis'})
                 analysis_result.gptAnalysis = GptAnalysisText(**gpt_parsed_output.get('analysis', {}))
                 logger.info(f"{log_prefix} GPT evaluation complete. EvalDir: {analysis_result.gptParams.trade_direction}, Conf: {analysis_result.gptParams.confidence_score}")
                 # Check for GPT internal errors
                 gpt_internal_error_msg = gpt_parsed_output.get("error") or (gpt_parsed_output.get('analysis', {}).get('signal_evaluation') or "").startswith("Error:")
                 if gpt_internal_error_msg:
                     err_detail = gpt_parsed_output.get("details", gpt_internal_error_msg)
                     logger.warning(f"{log_prefix} GPT issue: {err_detail}")
                     analysis_result.error = (analysis_result.error or "") + f"; GPT Warning: {str(err_detail)[:100]}"
            else:
                 # Handle case where parsing fails even after successful API call
                 logger.error(f"{log_prefix} GPT parsing failed unexpectedly.")
                 analysis_result.error = (analysis_result.error or "") + "; GPT parsing failed"
                 analysis_result.gptAnalysis = GptAnalysisText(raw_text="GPT parsing failed.")
                 # Ensure gptParams is set to hold if parsing fails
                 analysis_result.gptParams = GptTradingParams(trade_direction='hold')


        except Exception as e:
            logger.error(f"{log_prefix} GPT processing error: {e}", exc_info=True)
            analysis_result.error = (analysis_result.error or "") + f"; GPT Processing Error: {e}"
            gpt_parsed_output = None
            # Set default 'hold' parameters on error
            analysis_result.gptAnalysis = GptAnalysisText(raw_text=f"Error during GPT step: {e}")
            analysis_result.gptParams = GptTradingParams(trade_direction='hold')

    # <<< --- ADD THIS ELSE BLOCK --- >>>
    else:
        if not openai_client:
            logger.warning(f"{log_prefix} Skipping GPT evaluation: OpenAI client not configured.")
            reason = "OpenAI client missing"
        else: # openai_client exists, but technical_signal_direction == 'hold'
            logger.info(f"{log_prefix} Skipping GPT evaluation: Technical signal is 'hold'.")
            reason = "Technical signal is 'hold'"
        # Set default 'hold' parameters when skipping GPT
        analysis_result.gptAnalysis = GptAnalysisText(signal_evaluation=f"GPT skipped: {reason}")
        analysis_result.gptParams = GptTradingParams(trade_direction='hold', confidence_score=None) # Explicitly hold, confidence is N/A
        gpt_parsed_output = {'trade_direction': 'hold'} # Ensure gpt_parsed_output reflects hold for backtest check


    # Step 5: Backtesting
    # Use the final evaluated direction (which might be 'hold' if GPT was skipped or returned 'hold')
    evaluated_direction = analysis_result.gptParams.trade_direction if analysis_result.gptParams else 'hold'

    if evaluated_direction in ['long', 'short']:
        logger.info(f"{log_prefix} Running backtest for evaluated direction: {evaluated_direction}")
        try:
            # Pass a dictionary that definitely has the trade_direction key
            backtest_input_params = gpt_parsed_output if gpt_parsed_output else {'trade_direction': evaluated_direction}
            backtest_raw = await asyncio.to_thread(backtest_strategy, df_indicators.copy(), backtest_input_params)

            analysis_result.backtest = BacktestResultsData(
                strategy_score=backtest_raw.get('strategy_score'),
                trade_analysis=BacktestTradeAnalysis(**backtest_raw.get('trade_analysis', {})),
                recommendation=backtest_raw.get('recommendation'), warnings=backtest_raw.get('warnings', [])
            )
            logger.info(f"{log_prefix} Backtest done. Score: {analysis_result.backtest.strategy_score}, Rec: {analysis_result.backtest.recommendation}")
        except Exception as e:
            logger.error(f"{log_prefix} Backtesting error: {e}", exc_info=True)
            analysis_result.error = (analysis_result.error or "") + f"; Backtesting Error: {e}"
            analysis_result.backtest = BacktestResultsData(recommendation=f"Backtest failed: {e}")
    else:
        # This log now correctly reflects why backtest is skipped (hold from RSI or hold from GPT)
        logger.info(f"{log_prefix} Skipping backtest (Final EvalDir: {evaluated_direction}).")
        # Ensure backtest object exists even if skipped
        analysis_result.backtest = BacktestResultsData(recommendation=f"Backtest skipped: EvalDir={evaluated_direction}")


    # Finalization
    duration = time.time() - start_time_ticker
    if not analysis_result.error:
        logger.info(f"{log_prefix} Analysis successful ({duration:.2f}s).")
    else:
        logger.warning(f"{log_prefix} Analysis finished with issues ({duration:.2f}s). Status: {analysis_result.error}")

    return analysis_result

    # --- Step 2: Apply Indicators ---
    df_indicators = None
    try:
        df_raw['symbol'] = symbol # Pass symbol for logging within indicator functions
        df_indicators = await asyncio.to_thread(apply_technical_indicators, df_raw)
        latest_indicators_raw = df_indicators.iloc[-1].to_dict()
        indicator_payload = {}
        for field_name, model_field in IndicatorsData.model_fields.items():
            alias = model_field.alias or field_name
            value = latest_indicators_raw.get(alias)
            if pd.notna(value) and np.isfinite(value):
                indicator_payload[field_name] = float(value)
        analysis_result.indicators = IndicatorsData(**indicator_payload)
        logger.info(f"{log_prefix} Technical indicators applied.")
        # Log key indicators
        logger.debug(f"{log_prefix} Latest Indicators - RSI: {analysis_result.indicators.RSI:.2f}, "
                     f"SMA50: {analysis_result.indicators.SMA_50:.4f}, SMA200: {analysis_result.indicators.SMA_200:.4f}, "
                     f"ATR: {analysis_result.indicators.ATR:.4f}, ADX: {analysis_result.indicators.ADX:.2f}")
    except Exception as e:
        logger.error(f"{log_prefix} Indicator calculation error: {e}", exc_info=True)
        analysis_result.error = f"Indicator Error: {e}"
        return analysis_result

    # --- Step 2.5: Determine Technical Trigger ---
    technical_signal_direction = "hold" # Default
    latest_rsi = analysis_result.indicators.RSI
    rsi_oversold_trigger = 35 # Use the same value as backtester
    rsi_overbought_trigger = 65 # Use the same value as backtester

    if latest_rsi is not None and np.isfinite(latest_rsi):
        if latest_rsi < rsi_oversold_trigger:
            technical_signal_direction = "long"
            logger.info(f"{log_prefix} Technical signal trigger: LONG (RSI={latest_rsi:.2f} < {rsi_oversold_trigger})")
        elif latest_rsi > rsi_overbought_trigger:
            technical_signal_direction = "short"
            logger.info(f"{log_prefix} Technical signal trigger: SHORT (RSI={latest_rsi:.2f} > {rsi_overbought_trigger})")
        else:
            logger.info(f"{log_prefix} Technical signal trigger: HOLD (RSI={latest_rsi:.2f} between {rsi_oversold_trigger}-{rsi_overbought_trigger})")
    else:
        logger.warning(f"{log_prefix} Cannot determine technical signal: RSI is None or non-finite.")
        analysis_result.error = (analysis_result.error + "; " if analysis_result.error else "") + "Warning: RSI calculation failed"
        # Keep technical_signal_direction as 'hold'

    # --- Step 3: Statistical Models (GARCH, VaR - LSTM Disabled) ---
    garch_vol, var95 = None, None
    try:
        if df_indicators is not None and 'returns' in df_indicators:
            returns_series = df_indicators['returns']
            symbol_for_logs = symbol # Pass symbol for context
            # Run GARCH and VaR calculation (can be done concurrently if needed, but simpler sequentially)
            garch_vol = await asyncio.to_thread(fit_garch_model, returns_series, symbol_for_logs)
            var95 = await asyncio.to_thread(calculate_var, returns_series, 0.95, symbol_for_logs)
            analysis_result.modelOutput = ModelOutputData(garchVolatility=garch_vol, var95=var95)
            logger.info(f"{log_prefix} GARCH/VaR models executed (GARCH Vol: {garch_vol}, VaR95: {var95}).")
        else:
            logger.warning(f"{log_prefix} Skipping GARCH/VaR: Indicator DataFrame or 'returns' column missing.")
            analysis_result.modelOutput = ModelOutputData(garchVolatility=None, var95=None) # Ensure it exists but is empty

    except Exception as e:
        logger.error(f"{log_prefix} Statistical model execution error: {e}", exc_info=True)
        analysis_result.error = (analysis_result.error + "; " if analysis_result.error else "") + f"Stat Model Error: {e}"
        analysis_result.modelOutput = ModelOutputData(garchVolatility=None, var95=None) # Set empty on error

    # --- Step 4: GPT Evaluation ---
    gpt_parsed_output = None
    if openai_client:
        # Only call GPT if there's a technical signal to evaluate (optional, can evaluate 'hold' too)
        # if technical_signal_direction != 'hold': # <<< Uncomment this line to ONLY evaluate actual signals
        try:
            # Pass necessary data, including the technical signal direction
            gpt_raw_output = await asyncio.to_thread(
                gpt_generate_trading_parameters,
                df_indicators, symbol, timeframe, account_balance, max_leverage,
                garch_vol, var95, # Pass stat model results (LSTM is None implicitly)
                technical_signal_direction # Pass the RSI-derived signal
            )
            gpt_parsed_output = parse_gpt_trading_parameters(gpt_raw_output, symbol_for_log=symbol)

            if gpt_parsed_output:
                 # Assign GPT's evaluation results
                 analysis_result.gptParams = GptTradingParams(**{k: v for k, v in gpt_parsed_output.items() if k != 'analysis'})
                 analysis_result.gptAnalysis = GptAnalysisText(**gpt_parsed_output.get('analysis', {}))

                 # Check for errors reported *by* GPT or during parsing
                 gpt_internal_error_msg = gpt_parsed_output.get("error") or \
                                         (gpt_parsed_output.get('analysis', {}).get('signal_evaluation') or "").startswith("Error:")
                 if gpt_internal_error_msg:
                     error_detail = gpt_parsed_output.get("details", gpt_internal_error_msg)
                     logger.warning(f"{log_prefix} GPT reported an error/issue: {error_detail}")
                     analysis_result.error = (analysis_result.error + "; " if analysis_result.error else "") + f"GPT Warning: {str(error_detail)[:100]}"

                 logger.info(f"{log_prefix} GPT evaluation complete. Evaluated Direction: {analysis_result.gptParams.trade_direction}, Confidence: {analysis_result.gptParams.confidence_score}")

            else:
                 logger.error(f"{log_prefix} GPT parsing returned None or empty dictionary.")
                 analysis_result.error = (analysis_result.error + "; " if analysis_result.error else "") + "GPT parsing failed"
                 analysis_result.gptAnalysis = GptAnalysisText(raw_text="GPT parsing failed.") # Add placeholder

        except Exception as e:
            logger.error(f"{log_prefix} GPT processing error: {e}", exc_info=True)
            analysis_result.error = (analysis_result.error + "; " if analysis_result.error else "") + f"GPT Processing Error: {e}"
            gpt_parsed_output = None # Ensure it's None on error
            if analysis_result.gptAnalysis is None: analysis_result.gptAnalysis = GptAnalysisText()
            analysis_result.gptAnalysis.raw_text = f"Error during GPT step: {e}"
        # else: # <<< Corresponding else for the "if technical_signal_direction != 'hold':" check
        #    logger.info(f"{log_prefix} Skipping GPT evaluation as technical signal is 'hold'.")
        #    # Set default empty GPT results
        #    analysis_result.gptParams = GptTradingParams(trade_direction='hold', confidence_score=None)
        #    analysis_result.gptAnalysis = GptAnalysisText(signal_evaluation="No technical signal to evaluate.")

    else:
        logger.warning(f"{log_prefix} Skipping GPT evaluation: OpenAI client not configured.")
        analysis_result.gptAnalysis = GptAnalysisText(signal_evaluation="OpenAI client not configured.")
        analysis_result.gptParams = GptTradingParams(trade_direction='hold') # Ensure params exist


    # --- Step 5: Backtesting ---
    # Backtest is run based on GPT's *evaluated* direction to see if similar past signals (RSI based) were profitable
    if gpt_parsed_output and analysis_result.gptParams.trade_direction in ['long', 'short']:
        logger.info(f"{log_prefix} Running backtest for evaluated direction: {analysis_result.gptParams.trade_direction}")
        try:
            # Pass gpt_parsed_output which contains the evaluated direction
            backtest_raw_results = await asyncio.to_thread(backtest_strategy, df_indicators.copy(), gpt_parsed_output)
            trade_analysis_data = backtest_raw_results.get('trade_analysis', {})
            analysis_result.backtest = BacktestResultsData(
                strategy_score=backtest_raw_results.get('strategy_score'),
                trade_analysis=BacktestTradeAnalysis(**trade_analysis_data),
                recommendation=backtest_raw_results.get('recommendation'),
                warnings=backtest_raw_results.get('warnings', [])
            )
            logger.info(f"{log_prefix} Backtest completed. Score: {analysis_result.backtest.strategy_score}, Rec: {analysis_result.backtest.recommendation}")
        except Exception as e:
            logger.error(f"{log_prefix} Backtesting error: {e}", exc_info=True)
            analysis_result.error = (analysis_result.error + "; " if analysis_result.error else "") + f"Backtesting Error: {e}"
            analysis_result.backtest = BacktestResultsData(recommendation=f"Backtest failed: {e}", warnings=[f"Failed: {e}"])
    else:
        evaluated_dir = analysis_result.gptParams.trade_direction if analysis_result.gptParams else "N/A"
        reason = f"Evaluated trade direction is '{evaluated_dir}' or GPT params missing."
        logger.info(f"{log_prefix} Skipping backtest: {reason}")
        if analysis_result.backtest is None: # Initialize if not already set by an error
             analysis_result.backtest = BacktestResultsData(
                 recommendation=f"Backtest skipped: {reason}",
                 warnings=[f"Skipped: {reason}"]
             )

    # --- Finalization ---
    duration_ticker = time.time() - start_time_ticker
    final_error_status = analysis_result.error
    bt_recommendation = analysis_result.backtest.recommendation if analysis_result.backtest else "N/A"

    if not final_error_status:
        logger.info(f"{log_prefix} Analysis successful in {duration_ticker:.2f}s.")
    else:
        # Log warnings/skips clearly
        logger.warning(f"{log_prefix} Analysis completed with issues in {duration_ticker:.2f}s. Status: {final_error_status}, BT: {bt_recommendation}")
        analysis_result.error = final_error_status # Ensure error field reflects final status

    return analysis_result


# --- API Endpoints ---

# Startup (Unchanged)
@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI startup: Loading markets...")
    loaded = await load_exchange_markets(binance_futures)
    if not loaded: logger.warning("Market loading failed during startup.")
    if openai_client: asyncio.create_task(test_openai_connection(openai_client)) # Run in background

# Tickers (Unchanged)
@app.get("/api/crypto/tickers", response_model=TickersResponse, tags=["Utility"])
async def get_crypto_tickers_endpoint():
    logger.info("API: Request received for /tickers")
    if binance_futures is None: raise HTTPException(status_code=503, detail="Exchange unavailable")
    if not binance_futures.markets:
        logger.warning("API Tickers: Markets not loaded, attempting load.")
        if not await load_exchange_markets(binance_futures): raise HTTPException(status_code=503, detail="Failed load markets.")
    try:
        markets = binance_futures.markets
        if not markets:
            logger.error("API Tickers: Markets dictionary is empty after load attempt.")
            raise HTTPException(status_code=500, detail="Markets loaded empty.")
        tickers = sorted([m['symbol'] for m in markets.values() if m.get('swap') and m.get('quote')=='USDT' and m.get('settle')=='USDT' and m.get('active')])
        logger.info(f"API Tickers: Found {len(tickers)} active USDT perpetuals.")
        if not tickers: logger.warning("API Tickers: No active USDT linear perpetuals found.")
        return TickersResponse(tickers=tickers)
    except Exception as e:
        logger.error(f"API Tickers Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

# Analyze (Updated to reflect removed LSTM params)
@app.post("/api/crypto/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_crypto_endpoint(request: AnalysisRequest):
    logger.info(f"API: Analyze request received for {request.symbol} ({request.timeframe})")
    if binance_futures is None:
        raise HTTPException(status_code=503, detail="Exchange unavailable.")
    # Note: OpenAI client check happens inside perform_single_analysis

    try:
        # Pass parameters using model_dump() for Pydantic v2
        # LSTM parameters are no longer in AnalysisRequest, so they aren't passed
        analysis_params = request.model_dump()
        result = await perform_single_analysis(
            **analysis_params,
            min_requested_rr=None # Pass None as it's not in AnalysisRequest
        )

        # Error Handling (improved status codes)
        if result.error:
            err = result.error # Short alias
            status = 500 # Default internal server error
            log_func = logger.error # Default to error log

            # Check for specific error types or keywords
            if "Warning:" in err or "skipped:" in err:
                status = 200 # Treat warnings/skips as success with info
                log_func = logger.warning
            elif "Insufficient data" in err: status = 422 # Unprocessable Entity
            elif "Invalid symbol" in err: status = 400 # Bad Request
            elif "Network error" in err or "Connection error" in err or "Failed to load markets" in err : status = 504 # Gateway Timeout
            elif "Rate limit exceeded" in err or "ConnectionAbortedError" in err: status = 429 # Too Many Requests
            elif "Authentication Error" in err: status = 401 # Unauthorized
            elif "OpenAI" in err or "GPT" in err: status = 502 # Bad Gateway (issue with external service)
            elif "Exchange error" in err: status = 503 # Service Unavailable (exchange issue)
            # Keep 500 for Indicator Error, Stat Model Error, Backtesting Error, Unexpected

            log_func(f"API Analyze: Request for {request.symbol} finished with status {status}. Detail: {err}")
            if status != 200:
                 raise HTTPException(status_code=status, detail=err)
            # else: return result (for 200 OK with warnings)

        logger.info(f"API Analyze: Request for {request.symbol} completed successfully.")
        return result

    except HTTPException as h:
        # Re-raise HTTPExceptions directly
        raise h
    except Exception as e:
        logger.error(f"API Analyze Endpoint Error for {request.symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected endpoint internal error: {e}")


# Scan (Updated to use modified AnalysisRequest/perform_single_analysis, logging enhanced)
# --- Scan Endpoint (Updated Filtering Logic) ---
@app.post("/api/crypto/scan", response_model=ScanResponse, tags=["Scanning"])
async def scan_market_endpoint(request: ScanRequest):
    logger.critical("--- SCAN ENDPOINT ENTERED ---")
    scan_start_time = time.time()
    # Log the actual request received, including defaults for clarity on what's active
    logger.info(f"API Scan: Starting request: {request.model_dump_json(exclude_defaults=False)}")

    # --- Lookback Warning ---
    if request.timeframe == '1m' and request.lookback > 1500:
         logger.warning(f"API Scan: Lookback {request.lookback} for 1m timeframe might exceed fetch limits (~1500). Analysis might use fewer candles.")
    elif request.lookback > 1500:
         logger.warning(f"API Scan: Lookback {request.lookback} might exceed typical fetch limits (~1500).")

    # --- Basic Setup Checks ---
    if binance_futures is None:
        logger.error("API Scan Abort: Exchange unavailable.")
        raise HTTPException(status_code=503, detail="Exchange unavailable")
    # Note: OpenAI client presence is checked within perform_single_analysis if needed
    if not binance_futures.markets:
        logger.warning("API Scan: Markets not loaded, attempting load...")
        if not await load_exchange_markets(binance_futures):
            logger.error("API Scan Abort: Failed load markets.")
            raise HTTPException(status_code=503, detail="Failed load markets.")

    # --- 1. Get Tickers ---
    try:
        markets = binance_futures.markets
        if not markets:
            logger.error("API Scan Abort: Markets dictionary empty after load attempt.")
            raise ValueError("Markets unavailable post-load.")
        all_tickers = sorted([m['symbol'] for m in markets.values() if m.get('swap') and m.get('quote')=='USDT' and m.get('settle')=='USDT' and m.get('active')])
        logger.info(f"API Scan: Found {len(all_tickers)} active USDT perpetuals.")
        if not all_tickers:
            logger.warning("API Scan: No active USDT perpetuals found. Scan finished.")
            return ScanResponse(scan_parameters=request, total_tickers_attempted=0, total_tickers_succeeded=0,
                                ticker_start_index=request.ticker_start_index, ticker_end_index=request.ticker_end_index,
                                total_opportunities_found=0, top_opportunities=[], errors={})
    except Exception as e:
        logger.error(f"API Scan Tickers Error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Ticker retrieval error: {e}")

    # --- 2. Determine BTC Trend (Market Regime) ---
    btc_trend_state = "UNKNOWN" # Default: unknown/error/filter disabled
    apply_btc_filter = request.filter_by_btc_trend # Check if filter is requested

    if apply_btc_filter:
        logger.info(f"API Scan: BTC Trend filter enabled. Fetching BTC data for timeframe {request.timeframe}...")
        try:
            # Fetch sufficient data for SMAs on the requested timeframe
            btc_symbol = "BTC/USDT:USDT"
            # Ensure enough lookback for SMA 200 + buffer, respecting potential 1m limit
            btc_lookback = max(250, request.lookback)
            if request.timeframe == '1m' and btc_lookback > 1500:
                 logger.warning("Capping BTC lookback fetch for 1m timeframe to 1500")
                 btc_lookback = 1500
            elif btc_lookback > 1500:
                 logger.warning(f"BTC lookback {btc_lookback} might exceed limits, attempting fetch...")


            df_btc_raw = await asyncio.to_thread(get_real_time_data, btc_symbol, request.timeframe, btc_lookback)

            if df_btc_raw.empty or len(df_btc_raw) < 205: # Need enough for SMA200 + calculations
                 logger.warning(f"API Scan: Insufficient BTC data ({len(df_btc_raw)} bars fetched/valid, need >204) for trend analysis on {request.timeframe}. Disabling BTC filter.")
                 apply_btc_filter = False # Disable filter if not enough data
            else:
                df_btc_raw['symbol'] = btc_symbol # Add symbol for indicator logging
                df_btc_indicators = await asyncio.to_thread(apply_technical_indicators, df_btc_raw)

                # Check latest indicators for BTC
                btc_latest = df_btc_indicators.iloc[-1]
                btc_price = btc_latest.get('close')
                btc_sma50 = btc_latest.get('SMA_50')
                btc_sma200 = btc_latest.get('SMA_200')

                if btc_price is not None and btc_sma50 is not None and btc_sma200 is not None and \
                   np.isfinite(btc_price) and np.isfinite(btc_sma50) and np.isfinite(btc_sma200):

                    if btc_price > btc_sma50 > btc_sma200:
                        btc_trend_state = "UPTREND"
                    elif btc_price < btc_sma50 < btc_sma200:
                        btc_trend_state = "DOWNTREND"
                    else:
                        # Neither strictly up nor strictly down based on this definition
                        btc_trend_state = "CHOPPY"

                    logger.info(f"API Scan: Determined BTC Trend ({request.timeframe}): {btc_trend_state} (P: {btc_price:.2f}, S50: {btc_sma50:.2f}, S200: {btc_sma200:.2f})")
                else:
                    logger.warning(f"API Scan: Could not determine BTC trend (missing Price/SMA50/SMA200 values in latest BTC data). Disabling BTC filter. Latest BTC Indicators: {btc_latest.to_dict()}")
                    apply_btc_filter = False
                    btc_trend_state = "UNKNOWN"

        except Exception as e:
            logger.error(f"API Scan: Error fetching or analyzing BTC data: {e}. Disabling BTC filter.", exc_info=True)
            apply_btc_filter = False
            btc_trend_state = "ERROR"
    else:
        logger.info("API Scan: BTC Trend filter is disabled via request.")
    # --- END NEW BTC TREND SECTION ---


    # --- 3. Select Tickers ---
    tickers_to_scan = []
    total_available = len(all_tickers)
    start_index = request.ticker_start_index if request.ticker_start_index is not None else 0
    end_index = request.ticker_end_index
    slice_desc = ""
    actual_end_index_for_response = None

    if start_index >= total_available > 0:
        logger.warning(f"API Scan: Start index {start_index} out of bounds (>= total {total_available}). No tickers selected.")
        return ScanResponse(scan_parameters=request, total_tickers_attempted=0, total_tickers_succeeded=0, ticker_start_index=start_index, ticker_end_index=end_index, total_opportunities_found=0, top_opportunities=[], errors={})
    if start_index < 0:
        logger.warning(f"API Scan: Negative start index {start_index}, adjusted to 0.")
        start_index = 0

    # Determine the slice based on end_index and max_tickers
    if end_index is not None:
        if end_index <= start_index:
            logger.warning(f"API Scan: End index {end_index} <= start index {start_index}. No tickers selected.")
            tickers_to_scan = []
            slice_desc = f"invalid slice request [{start_index}:{end_index}]"
            actual_end_index_for_response = end_index
        else:
            actual_end = min(end_index, total_available)
            tickers_to_scan = all_tickers[start_index:actual_end]
            slice_desc = f"requested slice [{start_index}:{end_index}], actual slice [{start_index}:{actual_end}]"
            actual_end_index_for_response = actual_end
    elif request.max_tickers is not None and request.max_tickers > 0:
        limit = request.max_tickers
        actual_end = min(start_index + limit, total_available)
        tickers_to_scan = all_tickers[start_index:actual_end]
        slice_desc = f"using max_tickers={limit} from index {start_index}, actual slice [{start_index}:{actual_end}]"
        actual_end_index_for_response = actual_end
    elif request.max_tickers == 0:
         logger.warning("API Scan: max_tickers=0 requested. No tickers selected.")
         tickers_to_scan = []
         slice_desc = "max_tickers=0 requested"
         actual_end_index_for_response = start_index
    else: # No end_index, no max_tickers > 0 -> scan all from start_index
         tickers_to_scan = all_tickers[start_index:]
         actual_end = total_available
         slice_desc = f"scanning all from index {start_index}, actual slice [{start_index}:{actual_end}]"
         actual_end_index_for_response = actual_end

    logger.info(f"API Scan: Selected {len(tickers_to_scan)} tickers to analyze ({slice_desc}).")
    total_attempted = len(tickers_to_scan)

    if total_attempted == 0:
        logger.info("API Scan: No tickers selected based on criteria, ending scan early.")
        return ScanResponse(scan_parameters=request, total_tickers_attempted=0, total_tickers_succeeded=0,
                            ticker_start_index=start_index, ticker_end_index=actual_end_index_for_response,
                            total_opportunities_found=0, top_opportunities=[], errors={})


    # --- 4. Run Concurrently ---
    semaphore = asyncio.Semaphore(request.max_concurrent_tasks)
    tasks = []
    processed_count = 0
    progress_lock = asyncio.Lock()
    log_interval = max(1, total_attempted // 20 if total_attempted > 0 else 1) # Log ~5% progress

    async def analyze_with_semaphore_wrapper(ticker):
        nonlocal processed_count
        result = None
        task_start_time = time.time()
        try:
            async with semaphore:
                # logger.debug(f"API Scan: Acquired semaphore for {ticker}")
                result = await perform_single_analysis(
                    symbol=ticker,
                    timeframe=request.timeframe,
                    lookback=request.lookback,
                    account_balance=request.accountBalance,
                    max_leverage=request.maxLeverage,
                    min_requested_rr=request.min_risk_reward_ratio # Pass R/R request
                )
        except Exception as e:
            logger.error(f"API Scan Wrapper Error for {ticker}: {e}", exc_info=True)
            result = AnalysisResponse(symbol=ticker, timeframe=request.timeframe, error=f"Scan Wrapper Exception: {e}")
        finally:
            task_duration = time.time() - task_start_time
            async with progress_lock:
                processed_count += 1
                current_count = processed_count
            if current_count % log_interval == 0 or current_count == total_attempted:
                 logger.info(f"API Scan Progress: {current_count}/{total_attempted} tasks completed (Last: {ticker} took {task_duration:.2f}s).")
        return result if result is not None else AnalysisResponse(symbol=ticker, timeframe=request.timeframe, error="Unknown wrapper state")

    logger.info(f"API Scan: Creating {total_attempted} analysis tasks...")
    for ticker in tickers_to_scan:
        tasks.append(analyze_with_semaphore_wrapper(ticker))

    logger.info(f"API Scan: Gathering results for {total_attempted} tasks (Concurrency: {request.max_concurrent_tasks})...")
    analysis_results_raw: List[Any] = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info(f"API Scan: Finished gathering {len(analysis_results_raw)} task results.")

    # --- 5. Process Results ---
    successful_analyses_count = 0
    analysis_errors = {}
    opportunities_passing_filter = []
    logger.info("API Scan: Starting detailed processing and filtering of results...")

    for i, res_or_exc in enumerate(analysis_results_raw):
        symbol = tickers_to_scan[i] if i < len(tickers_to_scan) else f"UnknownTicker_{i}"
        logger.debug(f"--- Processing item {i+1}/{len(analysis_results_raw)} for symbol [{symbol}] ---")

        # Handle Exceptions from gather
        if isinstance(res_or_exc, Exception):
            logger.error(f"API Scan: Task [{symbol}] UNHANDLED Exception: {res_or_exc}", exc_info=False)
            analysis_errors[symbol] = f"Unhandled Task Exception: {str(res_or_exc)[:200]}"
            continue

        # Handle expected AnalysisResponse
        elif isinstance(res_or_exc, AnalysisResponse):
            result: AnalysisResponse = res_or_exc
            logger.debug(f"API Scan: Result for [{symbol}] is AnalysisResponse.")

            # Check for critical errors vs warnings
            is_warning = result.error and ("Warning:" in result.error or "skipped:" in result.error or "Insufficient data" in result.error)
            is_critical = result.error and not is_warning

            if is_critical:
                logger.error(f"API Scan: Critical error for [{result.symbol}]: {result.error}")
                analysis_errors[result.symbol] = f"Analysis Critical Error: {result.error}"
                continue # Skip further processing
            elif result.error:
                 logger.warning(f"API Scan: Non-critical issue/warning for [{result.symbol}]: {result.error}")
                 successful_analyses_count += 1 # Count as ran even with warnings
            else:
                 successful_analyses_count += 1 # Count error-free runs

            # Extract data safely
            gpt_params = result.gptParams; bt_results = result.backtest; indicators = result.indicators
            bt_analysis = bt_results.trade_analysis if bt_results else None
            direction = gpt_params.trade_direction if gpt_params else None
            gpt_conf = gpt_params.confidence_score if gpt_params and gpt_params.confidence_score is not None else None
            bt_score = bt_results.strategy_score if bt_results and bt_results.strategy_score is not None else None
            current_price = result.currentPrice

            # Log pre-filter state
            logger.info(
                f"[{result.symbol}] Pre-filter: EvalDir='{direction or 'N/A'}', "
                f"GPTConf={f'{gpt_conf:.2f}' if gpt_conf is not None else 'N/A'}, "
                f"BTScore={f'{bt_score:.2f}' if bt_score is not None else 'N/A'}, "
                f"Trades={bt_analysis.total_trades if bt_analysis else 'N/A'}, "
                f"WR={f'{bt_analysis.win_rate:.2%}' if bt_analysis and bt_analysis.win_rate is not None else 'N/A'}, "
                f"PF={'inf' if bt_analysis and bt_analysis.profit_factor == float('inf') else f'{bt_analysis.profit_factor:.2f}' if bt_analysis and bt_analysis.profit_factor is not None else 'N/A'}, "
                f"ADX={f'{indicators.ADX:.2f}' if indicators and indicators.ADX is not None else 'N/A'}"
            )

            # --- Filtering Logic ---
            passes_filters = True
            filter_fail_reason = ""

            # Filter 1: Basic Direction & Confidence/Score
            if direction not in ['long', 'short']: passes_filters = False; filter_fail_reason = f"Eval Direction not tradeable ('{direction}')"
            elif request.trade_direction and direction != request.trade_direction: passes_filters = False; filter_fail_reason = f"Direction mismatch (Req: {request.trade_direction}, Eval: {direction})"
            elif gpt_conf is None or gpt_conf < request.min_gpt_confidence: passes_filters = False; filter_fail_reason = f"GPT Conf too low (Req > {request.min_gpt_confidence:.2f}, Got: {f'{gpt_conf:.2f}' if gpt_conf is not None else 'N/A'})"
            elif bt_score is None or bt_score < request.min_backtest_score: bt_score_str = f'{bt_score:.2f}' if bt_score is not None else 'N/A'; passes_filters = False; filter_fail_reason = f"BT Score too low (Req > {request.min_backtest_score:.2f}, Got: {bt_score_str})"

            # Filter 2: Backtest Stats
            elif passes_filters and bt_analysis is None: passes_filters = False; filter_fail_reason = "Backtest analysis missing"
            elif passes_filters and request.min_backtest_trades is not None and (bt_analysis.total_trades < request.min_backtest_trades): passes_filters = False; filter_fail_reason = f"BT Trades too low (Req >= {request.min_backtest_trades}, Got: {bt_analysis.total_trades})"
            elif passes_filters and request.min_backtest_win_rate is not None and (bt_analysis.win_rate is None or bt_analysis.win_rate < request.min_backtest_win_rate): wr_str = f'{bt_analysis.win_rate:.2%}' if bt_analysis.win_rate is not None else 'N/A'; passes_filters = False; filter_fail_reason = f"BT Win Rate too low (Req >= {request.min_backtest_win_rate:.2%}, Got: {wr_str})"
            elif passes_filters and request.min_backtest_profit_factor is not None and (bt_analysis.profit_factor is None or (bt_analysis.profit_factor != float('inf') and bt_analysis.profit_factor < request.min_backtest_profit_factor)): pf_str = 'inf' if bt_analysis.profit_factor == float('inf') else f'{bt_analysis.profit_factor:.2f}' if bt_analysis.profit_factor is not None else 'N/A'; passes_filters = False; filter_fail_reason = f"BT Profit Factor too low (Req >= {request.min_backtest_profit_factor:.2f}, Got: {pf_str})"

            # Filter 3: Risk/Reward Ratio
            elif passes_filters and request.min_risk_reward_ratio is not None and request.min_risk_reward_ratio > 0:
                entry = gpt_params.optimal_entry if gpt_params else None; sl = gpt_params.stop_loss if gpt_params else None; tp = gpt_params.take_profit if gpt_params else None
                rr_ratio = None
                if entry is not None and sl is not None and tp is not None: risk = abs(entry - sl); reward = abs(tp - entry); rr_ratio = reward / risk if risk > 1e-9 else None
                if rr_ratio is None or rr_ratio < request.min_risk_reward_ratio: rr_str = f'{rr_ratio:.2f}' if rr_ratio is not None else 'N/A'; passes_filters = False; filter_fail_reason = f"R/R Ratio too low (Req >= {request.min_risk_reward_ratio:.2f}, Got: {rr_str})"

            # Filter 4: Indicator Filters (ADX, SMA Alignment)
            elif passes_filters and indicators is None: passes_filters = False; filter_fail_reason = "Indicator data missing"
            elif passes_filters and request.min_adx is not None and request.min_adx > 0:
                 adx = indicators.ADX
                 if adx is None or adx < request.min_adx: adx_str = f'{adx:.1f}' if adx is not None else 'N/A'; passes_filters = False; filter_fail_reason = f"ADX too low (Req >= {request.min_adx:.1f}, Got: {adx_str})"
            elif passes_filters and request.require_sma_alignment:
                 sma50 = indicators.SMA_50; sma200 = indicators.SMA_200; price = current_price; sma_aligned = False
                 price_str = f'{price:.4f}' if price is not None else 'N/A'; sma50_str = f'{sma50:.4f}' if sma50 is not None else 'N/A'; sma200_str = f'{sma200:.4f}' if sma200 is not None else 'N/A'
                 if price is not None and sma50 is not None and sma200 is not None:
                     if direction == 'long' and price > sma50 > sma200: sma_aligned = True
                     elif direction == 'short' and price < sma50 < sma200: sma_aligned = True
                 if not sma_aligned: passes_filters = False; filter_fail_reason = f"SMA alignment failed (Req: {request.require_sma_alignment}, Dir: {direction}, P:{price_str}, S50:{sma50_str}, S200:{sma200_str})"

            # --- Filter 5: BTC Trend Alignment ---
            elif passes_filters and apply_btc_filter: # Only apply if requested AND BTC state is known and not error/unknown
                if btc_trend_state in ["UPTREND", "DOWNTREND", "CHOPPY"]: # Ensure we have a valid state
                    if direction == 'long' and btc_trend_state != 'UPTREND':
                        passes_filters = False; filter_fail_reason = f"BTC Trend not UPTREND (Is: {btc_trend_state})"
                    elif direction == 'short' and btc_trend_state != 'DOWNTREND':
                        passes_filters = False; filter_fail_reason = f"BTC Trend not DOWNTREND (Is: {btc_trend_state})"
                    # else: Allow trade if direction matches trend or if BTC is choppy (optional: could filter out choppy too)
                else: # If BTC state is UNKNOWN or ERROR, the filter effectively does nothing (or could be stricter)
                    logger.debug(f"[{result.symbol}] Skipping BTC trend filter application due to BTC state: {btc_trend_state}")


            # Add Opportunity if Passes All Filters
            if passes_filters:
                logger.info(f"[{result.symbol}] PASSED ALL FILTERS. Adding to opportunities.")
                score_g = float(gpt_conf) if gpt_conf is not None else 0.0
                score_b = float(bt_score) if bt_score is not None else 0.0
                combined_score = round((score_g * 0.6) + (score_b * 0.4), 3) # Adjust weighting as needed

                summary = "Analysis unavailable." # Default
                if result.gptAnalysis:
                    eval_text = result.gptAnalysis.signal_evaluation; tech_text = result.gptAnalysis.technical_analysis
                    if eval_text and isinstance(eval_text, str) and "Error:" not in eval_text: summary = eval_text.split('.')[0] + '.'
                    elif tech_text and isinstance(tech_text, str): summary = tech_text.split('.')[0] + '.'
                    elif result.error: summary = f"Note: {result.error}"
                    else: summary = result.gptAnalysis.raw_text.split('.')[0] + '.' if result.gptAnalysis.raw_text else summary
                    summary = summary[:147] + "..." if len(summary) > 150 else summary

                opportunity = ScanResultItem(
                    rank=0, symbol=result.symbol, timeframe=result.timeframe, currentPrice=current_price,
                    gptConfidence=gpt_conf, backtestScore=bt_score, combinedScore=combined_score,
                    tradeDirection=direction,
                    optimalEntry=gpt_params.optimal_entry if gpt_params else None,
                    stopLoss=gpt_params.stop_loss if gpt_params else None,
                    takeProfit=gpt_params.take_profit if gpt_params else None,
                    gptAnalysisSummary=summary
                )
                opportunities_passing_filter.append(opportunity)
            else:
                 logger.info(f"[{result.symbol}] FAILED FILTERS. Reason: {filter_fail_reason}.")

        # Unexpected type handling
        else:
             logger.error(f"API Scan: UNEXPECTED result type [{symbol}]: {type(res_or_exc)}.")
             analysis_errors[symbol] = f"Unexpected Result Type: {type(res_or_exc).__name__}"

    # --- End of Result Processing Loop ---
    logger.info(f"API Scan: Finished processing results. Succeeded: {successful_analyses_count}, Errors: {len(analysis_errors)}, Passed Filters: {len(opportunities_passing_filter)}.")

    # --- 6. Rank Opportunities ---
    if opportunities_passing_filter:
        logger.info(f"API Scan: Sorting {len(opportunities_passing_filter)} opportunities...")
        opportunities_passing_filter.sort(key=lambda x: x.combinedScore or 0.0, reverse=True)
        top_opportunities = []
        for rank_idx, opp in enumerate(opportunities_passing_filter[:request.top_n]):
            opp.rank = rank_idx + 1; top_opportunities.append(opp)
        logger.info(f"API Scan: Ranking complete. Selected top {len(top_opportunities)}.")
    else:
        logger.info("API Scan: No opportunities passed filters.")
        top_opportunities = []

    # --- 7. Construct Final Response ---
    scan_duration = time.time() - scan_start_time
    logger.info(f"API Scan: Completed in {scan_duration:.2f}s. Returning response.")
    if analysis_errors:
        logger.warning(f"API Scan finished with {len(analysis_errors)} critical errors/exceptions for symbols: {list(analysis_errors.keys())}")

    return ScanResponse(
        scan_parameters=request,
        total_tickers_attempted=total_attempted,
        total_tickers_succeeded=successful_analyses_count,
        ticker_start_index=start_index,
        ticker_end_index=actual_end_index_for_response,
        total_opportunities_found=len(opportunities_passing_filter),
        top_opportunities=top_opportunities,
        errors=analysis_errors
    )


# --- Main Execution (Log Level Handling Updated) ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8029))
    host = os.getenv("HOST", "127.0.0.1")
    reload_flag = os.getenv("UVICORN_RELOAD", "false").lower() in ("true", "1", "yes") # Default reload to false for stability

    # Use the LOG_LEVEL determined by the logging setup earlier
    uvicorn_log_level = LOG_LEVEL.lower()
    logger.info(f"Setting Uvicorn log level to: {uvicorn_log_level}")

    # Print server configuration details
    print("\n" + "="*30)
    print(" --- Starting FastAPI Server ---")
    print(f" Host: {host}")
    print(f" Port: {port}")
    print(f" Auto-Reload: {reload_flag}")
    print(f" Logging Level: {LOG_LEVEL}")
    print(f" OpenAI Client Initialized: {openai_client is not None}")
    print(f" CCXT Client Initialized: {binance_futures is not None}")
    print(f" Default Max Concurrent Tasks: {DEFAULT_MAX_CONCURRENT_TASKS}")
    print(f" LSTM Status: DISABLED") # Indicate LSTM is off
    # --- LSTM DISABLED: Remove GPU check if TF is fully removed ---
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Set memory growth is generally good practice if TF is used elsewhere
    #         # for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    #         print(f" Detected {len(gpus)} Physical GPUs.") # Note: TF not actively used by analysis
    #     except RuntimeError as e: print(f" Error during GPU setup (TF might still be imported): {e}")
    # else: print(" No GPU detected by TensorFlow (or TF not used).")
    # --- END LSTM DISABLED ---
    print("="*30 + "\n")

    # Run Uvicorn with the configured log level
    uvicorn.run(
        "__main__:app",
        host=host,
        port=port,
        reload=reload_flag,
        log_level=uvicorn_log_level # Pass the determined level
        # Optional: Force Uvicorn to use standard loggers if issues persist
        # use_colors=False,
        # log_config=None # Prevents Uvicorn from overriding basicConfig completely
    )
