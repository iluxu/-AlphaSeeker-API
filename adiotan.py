# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import time
import ccxt
# import ccxt.async_support as ccxt_async # Keep commented if not used
import logging
from datetime import datetime
from arch import arch_model
from sklearn.preprocessing import StandardScaler
# Silence TensorFlow warnings BEFORE importing Keras/TF (If TF is still needed elsewhere)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# --- LSTM DISABLED: Imports are already commented out ---
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
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(log_formatter)
logger = logging.getLogger()
logger.setLevel(LOG_LEVEL)
if logger.hasHandlers(): logger.handlers.clear()
logger.addHandler(log_handler)
logger.critical("--- Logging Initialized (Level: %s). Logs should now appear on the console. ---", LOG_LEVEL)

# --- Configuration ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="ConvergenceWarning", category=UserWarning)
load_dotenv()
DEFAULT_MAX_CONCURRENT_TASKS = 5 # Keep default, can be overridden by ScanRequest

# --- CCXT Initialization ---
binance_futures = None
try:
    # Consider adding API Key/Secret from .env if needed for private endpoints later
    # api_key = os.getenv("BINANCE_API_KEY")
    # secret = os.getenv("BINANCE_SECRET")
    binance_futures = ccxt.binanceusdm({
        'enableRateLimit': True,
        'options': { 'adjustForTimeDifference': True },
        'timeout': 30000, # Increased timeout
        'rateLimit': 150 # Slightly reduced to be safer
        # 'apiKey': api_key, # Uncomment if needed
        # 'secret': secret,  # Uncomment if needed
    })
    logger.info("CCXT Binance Futures instance created.")
except Exception as e:
    logger.error(f"Error initializing CCXT: {e}", exc_info=True)
    binance_futures = None

# --- Load Markets Function ---
async def load_exchange_markets(exchange):
    if not exchange: return False
    try:
        logger.info(f"Attempting to load markets for {exchange.id}...")
        markets = await asyncio.to_thread(exchange.load_markets, True) # Force reload might be needed sometimes
        if markets:
             logger.info(f"Successfully loaded {len(markets)} markets for {exchange.id}.")
             return True
        else:
             logger.warning(f"Market loading returned empty for {exchange.id}.")
             return False
    # More specific error handling
    except ccxt.AuthenticationError as e:
        logger.error(f"CCXT Authentication Error for {exchange.id}: {e}", exc_info=False)
        return False
    except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
        logger.error(f"Network/Timeout Error loading markets for {exchange.id}: {e}", exc_info=False)
        return False
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange specific error loading markets for {exchange.id}: {e}", exc_info=False)
        return False
    except Exception as e:
        logger.error(f"Unexpected error loading markets for {exchange.id}: {e}", exc_info=True)
        return False

# --- OpenAI Initialization ---
openai_client = None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable not found. GPT features will be disabled.")
else:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized.")
    except Exception as e:
        logger.error(f"Error initializing OpenAI: {e}", exc_info=True)
        openai_client = None

# --- Test OpenAI Connection ---
async def test_openai_connection(client):
     if not client: return
     try:
         # Use await with async client or asyncio.to_thread for sync client
         await asyncio.to_thread(client.models.list)
         logger.info("OpenAI connection test successful.")
     except openai.AuthenticationError:
         logger.error("OpenAI Authentication Error: Invalid API Key or setup.")
     except openai.RateLimitError:
         logger.warning("OpenAI Rate Limit potentially hit during initial test.")
     except Exception as e:
         logger.error(f"OpenAI connection test failed: {e}")

# --- FastAPI App ---
app = FastAPI(
    title="Crypto Trading Analysis & Scanning API",
    description="API for technical analysis (MACD Signal + Trend), GPT-driven evaluation, backtesting, and market scanning.",
    version="1.4.0_MACD_Signal" # Version bump indicating change
)

# --- Pydantic Models (Largely Unchanged, just confirming fields used) ---

class TickerRequest(BaseModel): pass
class TickersResponse(BaseModel): tickers: List[str]

class AnalysisRequest(BaseModel):
    symbol: str = Field(..., example="BTC/USDT:USDT")
    timeframe: str = Field(default="1h", example="1h")
    lookback: int = Field(default=1000, ge=250)
    accountBalance: float = Field(default=1000.0, ge=0)
    maxLeverage: float = Field(default=10.0, ge=1)
    # No LSTM params

class IndicatorsData(BaseModel):
    RSI: Optional[float] = None; ATR: Optional[float] = None; SMA_50: Optional[float] = None; SMA_200: Optional[float] = None
    EMA_12: Optional[float] = None; EMA_26: Optional[float] = None; MACD: Optional[float] = None; Signal_Line: Optional[float] = None
    Bollinger_Upper: Optional[float] = None; Bollinger_Middle: Optional[float] = None; Bollinger_Lower: Optional[float] = None
    Momentum: Optional[float] = None; Stochastic_K: Optional[float] = None; Stochastic_D: Optional[float] = None
    Williams_R: Optional[float] = Field(None, alias="Williams_%R")
    ADX: Optional[float] = None; CCI: Optional[float] = None; OBV: Optional[float] = None; returns: Optional[float] = None
    model_config = ConfigDict(populate_by_name=True, extra='allow') # Allow extra fields if needed

class ModelOutputData(BaseModel):
    # No LSTM
    garchVolatility: Optional[float] = None
    var95: Optional[float] = None

class GptAnalysisText(BaseModel):
    # No LSTM
    technical_analysis: Optional[str] = None
    risk_assessment: Optional[str] = None
    market_outlook: Optional[str] = None
    raw_text: Optional[str] = None
    signal_evaluation: Optional[str] = None # Keep this for evaluation summary

class GptTradingParams(BaseModel):
    optimal_entry: Optional[float] = None; stop_loss: Optional[float] = None; take_profit: Optional[float] = None
    trade_direction: Optional[str] = None # Can be 'long', 'short', or 'hold'
    leverage: Optional[int] = Field(None, ge=1)
    position_size_usd: Optional[float] = Field(None, ge=0); estimated_profit: Optional[float] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=1)

class BacktestTradeAnalysis(BaseModel):
    total_trades: int = 0; winning_trades: int = 0; losing_trades: int = 0; win_rate: Optional[float] = None
    avg_profit: Optional[float] = None; avg_loss: Optional[float] = None; profit_factor: Optional[float] = None
    total_profit: Optional[float] = None; largest_win: Optional[float] = None; largest_loss: Optional[float] = None
    average_trade_duration: Optional[float] = None

class BacktestResultsData(BaseModel):
    strategy_score: Optional[float] = Field(None, ge=0, le=1); trade_analysis: Optional[BacktestTradeAnalysis] = None
    recommendation: Optional[str] = None; warnings: List[str] = Field([])

class AnalysisResponse(BaseModel):
    symbol: str; timeframe: str; currentPrice: Optional[float] = None
    indicators: Optional[IndicatorsData] = None
    modelOutput: Optional[ModelOutputData] = None
    gptParams: Optional[GptTradingParams] = None
    gptAnalysis: Optional[GptAnalysisText] = None
    backtest: Optional[BacktestResultsData] = None
    error: Optional[str] = None

# ScanRequest - Added explicit min_risk_reward_ratio, potentially relax defaults slightly?
# Keeping user's defaults for now.
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


# --- Helper Functions ---

# --- Data Fetcher (Added more robust error handling and retry logic idea) ---
async def get_real_time_data(symbol: str, timeframe: str = "1d", limit: int = 1000, retries: int = 2) -> pd.DataFrame:
    """Fetch OHLCV data with retries for network issues."""
    logger.debug(f"[{symbol}] Attempting to fetch {limit} candles for timeframe {timeframe}")
    if binance_futures is None: raise ConnectionError("CCXT exchange instance is not available.")
    if not binance_futures.markets:
         logger.warning(f"[{symbol}] Markets not loaded, attempting synchronous load...")
         try: binance_futures.load_markets(True); logger.info(f"[{symbol}] Markets loaded successfully (sync).")
         except Exception as e: raise ConnectionError(f"Failed to load markets synchronously: {e}") from e

    last_exception = None
    for attempt in range(retries + 1):
        try:
            # Use asyncio.to_thread for the synchronous CCXT call
            ohlcv = await asyncio.to_thread(binance_futures.fetch_ohlcv, symbol, timeframe, None, limit)

            if not ohlcv:
                logger.warning(f"[{symbol}] No OHLCV data returned from fetch_ohlcv (Attempt {attempt+1}).")
                # Don't retry if empty list returned, might be no data for symbol/tf
                return pd.DataFrame()

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)
            df = df.apply(pd.to_numeric, errors='coerce')
            df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True) # Keep only valid candles

            if df.empty: logger.warning(f"[{symbol}] DataFrame became empty after type conversion/NaN drop.")
            else: logger.debug(f"[{symbol}] Fetched {len(df)} valid candles.")
            return df # Success

        except ccxt.BadSymbol as e:
            logger.error(f"[{symbol}] Invalid symbol error: {e}")
            raise ValueError(f"Invalid symbol '{symbol}'") from e # No retry for bad symbol
        except ccxt.AuthenticationError as e:
             logger.error(f"[{symbol}] Authentication error: {e}")
             raise PermissionError("CCXT Authentication Error") from e # No retry
        except (ccxt.RateLimitExceeded, ccxt.DDoSProtection, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, ccxt.NetworkError) as e:
            last_exception = e
            wait_time = 2 ** attempt # Exponential backoff
            logger.warning(f"[{symbol}] Network/RateLimit error (Attempt {attempt+1}/{retries+1}): {type(e).__name__}. Retrying in {wait_time}s...")
            if attempt < retries: await asyncio.sleep(wait_time)
            else: logger.error(f"[{symbol}] Max retries reached for network/rate limit errors."); raise ConnectionAbortedError(f"Failed after {retries} retries: {e}") from e
        except Exception as e:
            logger.error(f"[{symbol}] Unexpected error fetching data: {e}", exc_info=True)
            last_exception = e
            # Maybe retry for generic errors too? For now, raise.
            raise RuntimeError(f"Failed to fetch data for {symbol}") from e

    # Should not be reached if logic is correct, but as a safeguard
    raise last_exception if last_exception else RuntimeError(f"Unknown error fetching data for {symbol}")


# --- Indicator Functions (compute_*, apply_technical_indicators - Unchanged) ---
def compute_rsi(series, window=14):
    delta = series.diff(); gain = delta.where(delta > 0, 0.0).fillna(0); loss = -delta.where(delta < 0, 0.0).fillna(0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean(); avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan); rsi = 100.0 - (100.0 / (1.0 + rs)); return rsi.fillna(50) # Fill NaN with 50 (neutral) might be better
def compute_atr(df, window=14):
    high_low = df['high'] - df['low']; high_close = abs(df['high'] - df['close'].shift()); low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False); atr = tr.ewm(com=window - 1, min_periods=window).mean(); return atr
def compute_bollinger_bands(series, window=20):
    sma = series.rolling(window=window, min_periods=window).mean(); std = series.rolling(window=window, min_periods=window).std()
    upper_band = sma + 2 * std; lower_band = sma - 2 * std; return upper_band, sma, lower_band
def compute_stochastic_oscillator(df, window=14, smooth_k=3):
    lowest_low = df['low'].rolling(window=window, min_periods=window).min(); highest_high = df['high'].rolling(window=window, min_periods=window).max()
    range_hh_ll = (highest_high - lowest_low).replace(0, np.nan); k_percent = 100 * ((df['close'] - lowest_low) / range_hh_ll)
    d_percent = k_percent.rolling(window=smooth_k, min_periods=smooth_k).mean(); return k_percent.fillna(50), d_percent.fillna(50) # Fill NaN 50
def compute_williams_r(df, window=14):
    highest_high = df['high'].rolling(window=window, min_periods=window).max(); lowest_low = df['low'].rolling(window=window, min_periods=window).min()
    range_ = (highest_high - lowest_low).replace(0, np.nan); williams_r = -100 * ((highest_high - df['close']) / range_); return williams_r.fillna(-50) # Fill NaN -50
def compute_adx(df, window=14):
    df_adx = df.copy(); df_adx['H-L'] = df_adx['high'] - df_adx['low']; df_adx['H-PC'] = abs(df_adx['high'] - df_adx['close'].shift(1))
    df_adx['L-PC'] = abs(df_adx['low'] - df_adx['close'].shift(1)); df_adx['TR_calc'] = df_adx[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df_adx['DMplus'] = np.where((df_adx['high'] - df_adx['high'].shift(1)) > (df_adx['low'].shift(1) - df_adx['low']), df_adx['high'] - df_adx['high'].shift(1), 0)
    df_adx['DMplus'] = np.where(df_adx['DMplus'] < 0, 0, df_adx['DMplus'])
    df_adx['DMminus'] = np.where((df_adx['low'].shift(1) - df_adx['low']) > (df_adx['high'] - df_adx['high'].shift(1)), df_adx['low'].shift(1) - df_adx['low'], 0)
    df_adx['DMminus'] = np.where(df_adx['DMminus'] < 0, 0, df_adx['DMminus'])
    # Wilder's smoothing (similar to EMA but with alpha = 1/N)
    TR_smooth = df_adx['TR_calc'].ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    DMplus_smooth = df_adx['DMplus'].ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    DMminus_smooth = df_adx['DMminus'].ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    # Calculate DI+ and DI-
    DIplus = 100 * (DMplus_smooth / TR_smooth.replace(0, np.nan)).fillna(0)
    DIminus = 100 * (DMminus_smooth / TR_smooth.replace(0, np.nan)).fillna(0)
    # Calculate DX
    DIsum = (DIplus + DIminus).replace(0, np.nan)
    DX = 100 * (abs(DIplus - DIminus) / DIsum).fillna(0)
    # Calculate ADX
    ADX = DX.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    return ADX.fillna(20) # Fill NaN ADX with 20 (neutral-ish)?
def compute_cci(df, window=20):
    tp = (df['high'] + df['low'] + df['close']) / 3; sma = tp.rolling(window=window, min_periods=window).mean()
    mad = tp.rolling(window=window, min_periods=window).apply(lambda x: np.nanmean(np.abs(x - np.nanmean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad.replace(0, np.nan)); return cci.fillna(0) # Fill NaN CCI 0
def compute_obv(df):
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum(); return obv

def apply_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply technical indicators."""
    df_copy = df.copy()
    df_copy['close'] = df_copy['close'].astype(float)
    df_copy['returns'] = np.log(df_copy['close'] / df_copy['close'].shift(1)).fillna(0)
    df_len = len(df_copy)
    symbol_log = df_copy.get('symbol', 'Unknown')
    logger.debug(f"[{symbol_log}] Applying indicators to {df_len} candles.")

    def assign_if_enough_data(col_name, min_len_needed, calculation_lambda):
        if df_len >= min_len_needed:
             try: df_copy[col_name] = calculation_lambda()
             except Exception as e: logger.error(f"[{symbol_log}] Error calculating {col_name}: {e}", exc_info=False); df_copy[col_name] = np.nan
        else:
             logger.debug(f"[{symbol_log}] Skipping {col_name}, need {min_len_needed}, got {df_len}.")
             df_copy[col_name] = np.nan

    # Calculate base indicators first
    assign_if_enough_data('SMA_50', 50, lambda: df_copy['close'].rolling(window=50, min_periods=50).mean())
    assign_if_enough_data('SMA_200', 200, lambda: df_copy['close'].rolling(window=200, min_periods=200).mean())
    assign_if_enough_data('EMA_12', 26, lambda: df_copy['close'].ewm(span=12, adjust=False, min_periods=12).mean())
    assign_if_enough_data('EMA_26', 26, lambda: df_copy['close'].ewm(span=26, adjust=False, min_periods=26).mean())
    assign_if_enough_data('ATR', 15, lambda: compute_atr(df_copy, window=14))
    assign_if_enough_data('ADX', 28, lambda: compute_adx(df_copy, window=14)) # Need ~2*window for ADX calc

    # Indicators dependent on others
    if df_len >= 26 and 'EMA_12' in df_copy and 'EMA_26' in df_copy and df_copy['EMA_12'].notna().any() and df_copy['EMA_26'].notna().any():
         df_copy['MACD'] = df_copy['EMA_12'] - df_copy['EMA_26']
         if df_len >= 35 and 'MACD' in df_copy and df_copy['MACD'].notna().any(): # Need 9 periods of MACD
             assign_if_enough_data('Signal_Line', 35, lambda: df_copy['MACD'].ewm(span=9, adjust=False, min_periods=9).mean())
         else: df_copy['Signal_Line'] = np.nan
    else: df_copy['MACD'], df_copy['Signal_Line'] = np.nan, np.nan

    # Other indicators
    assign_if_enough_data('RSI', 15, lambda: compute_rsi(df_copy['close'], window=14))
    if df_len >= 21:
        try:
            upper, middle, lower = compute_bollinger_bands(df_copy['close'], window=20)
            df_copy['Bollinger_Upper'], df_copy['Bollinger_Middle'], df_copy['Bollinger_Lower'] = upper, middle, lower
        except Exception as e: logger.error(f"[{symbol_log}] Error calculating Bollinger Bands: {e}", exc_info=False); df_copy['Bollinger_Upper'], df_copy['Bollinger_Middle'], df_copy['Bollinger_Lower'] = np.nan, np.nan, np.nan
    else: df_copy['Bollinger_Upper'], df_copy['Bollinger_Middle'], df_copy['Bollinger_Lower'] = np.nan, np.nan, np.nan

    assign_if_enough_data('Momentum', 11, lambda: df_copy['close'] - df_copy['close'].shift(10))
    if df_len >= 17:
        try:
            k, d = compute_stochastic_oscillator(df_copy, window=14, smooth_k=3)
            df_copy['Stochastic_K'], df_copy['Stochastic_D'] = k, d
        except Exception as e: logger.error(f"[{symbol_log}] Error calculating Stochastic: {e}", exc_info=False); df_copy['Stochastic_K'], df_copy['Stochastic_D'] = np.nan, np.nan
    else: df_copy['Stochastic_K'], df_copy['Stochastic_D'] = np.nan, np.nan

    assign_if_enough_data('Williams_%R', 15, lambda: compute_williams_r(df_copy, window=14))
    assign_if_enough_data('CCI', 21, lambda: compute_cci(df_copy, window=20))
    assign_if_enough_data('OBV', 2, lambda: compute_obv(df_copy))

    logger.debug(f"[{symbol_log}] Finished applying indicators.")
    return df_copy

# --- Statistical Models (GARCH, VaR - Unchanged) ---
def fit_garch_model(returns: pd.Series, symbol_log: str = "Unknown") -> Optional[float]:
    """Fit GARCH(1,1) model. Returns NEXT PERIOD conditional volatility."""
    valid_returns = returns.dropna() * 100
    logger.debug(f"[{symbol_log}] GARCH input len: {len(valid_returns)}")
    if len(valid_returns) < 50:
        logger.warning(f"[{symbol_log}] Skipping GARCH, need 50 returns, got {len(valid_returns)}.")
        return None
    try:
        am = arch_model(valid_returns, vol='Garch', p=1, q=1, dist='Normal') # 'Garch' often preferred over 'GARCH'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(update_freq=0, disp='off', show_warning=False)
        if res.convergence_flag == 0:
            forecasts = res.forecast(horizon=1, reindex=False)
            cond_vol_forecast = np.sqrt(forecasts.variance.iloc[-1, 0]) / 100.0
            logger.debug(f"[{symbol_log}] GARCH fit successful. Forecast Vol: {cond_vol_forecast:.4f}")
            return float(cond_vol_forecast) if np.isfinite(cond_vol_forecast) else None
        else:
            logger.warning(f"[{symbol_log}] GARCH did not converge (Flag: {res.convergence_flag}).")
            return None
    except Exception as e:
        logger.error(f"[{symbol_log}] GARCH fitting error: {e}", exc_info=False)
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
        return float(var_value) if np.isfinite(var_value) else None
    except Exception as e:
        logger.error(f"[{symbol_log}] Error calculating VaR: {e}", exc_info=False)
        return None

# --- GPT Integration (Prompt Heavily Modified for NEW Signal) ---
def gpt_generate_trading_parameters(
    df_with_indicators: pd.DataFrame,
    symbol: str,
    timeframe: str,
    account_balance: float,
    max_leverage: float,
    garch_volatility: Optional[float],
    var95: Optional[float],
    technically_derived_direction: str, # The 'long'/'short'/'hold' derived from MACD/Trend/ADX
    min_requested_rr: Optional[float] # Pass the R/R from ScanRequest here
) -> str:
    """Generate trading parameters using GPT to EVALUATE a MACD/Trend/ADX technical signal."""
    log_prefix = f"[{symbol} ({timeframe}) GPT]"
    if openai_client is None:
        logger.warning(f"{log_prefix} OpenAI client not available.")
        return json.dumps({"error": "OpenAI client not available"})

    # Need MACD, Signal Line, SMA200, ADX, Close, ATR for evaluation
    required_cols_gpt = ['close', 'ATR', 'MACD', 'Signal_Line', 'SMA_200', 'ADX']
    df_valid = df_with_indicators.dropna(subset=required_cols_gpt)
    if df_valid.empty or len(df_valid) < 2: # Need at least current and previous for context
        logger.warning(f"{log_prefix} Insufficient indicator data for GPT evaluation ({required_cols_gpt}).")
        return json.dumps({"error": f"Insufficient indicator data for GPT ({required_cols_gpt})"})

    latest_data = df_valid.iloc[-1].to_dict()
    prev_data = df_valid.iloc[-2].to_dict() # For context like previous MACD relationship
    technical_indicators = {}

    # Extract finite indicators for context - Focus on key ones for the prompt
    key_inds_for_prompt = ['RSI', 'ATR', 'SMA_50', 'SMA_200', 'MACD', 'Signal_Line',
                           'Bollinger_Upper', 'Bollinger_Middle', 'Bollinger_Lower',
                           'Stochastic_K', 'Stochastic_D', 'ADX', 'CCI']
    for key_in_df in key_inds_for_prompt:
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

    # Provide context about the signal trigger
    signal_context = "N/A"
    if technically_derived_direction == 'long':
        signal_context = f"Potential LONG signal triggered by MACD ({latest_data.get('MACD', 'N/A'):.4f}) crossing above Signal Line ({latest_data.get('Signal_Line', 'N/A'):.4f}) while Price ({current_price:.4f}) > SMA200 ({latest_data.get('SMA_200', 'N/A'):.4f}) and ADX ({latest_data.get('ADX', 'N/A'):.2f}) >= 20."
    elif technically_derived_direction == 'short':
        signal_context = f"Potential SHORT signal triggered by MACD ({latest_data.get('MACD', 'N/A'):.4f}) crossing below Signal Line ({latest_data.get('Signal_Line', 'N/A'):.4f}) while Price ({current_price:.4f}) < SMA200 ({latest_data.get('SMA_200', 'N/A'):.4f}) and ADX ({latest_data.get('ADX', 'N/A'):.2f}) >= 20."
    else: # Hold case (shouldn't usually reach GPT if filter applied in perform_single_analysis, but handle defensively)
        signal_context = "No MACD/Trend/ADX signal triggered."
        # If GPT is called even on 'hold', ask it to find *any* setup
        # For now, assume GPT is only called on 'long'/'short'

    market_info = {
        "symbol": symbol,
        "timeframe": timeframe,
        "current_price": current_price,
        "garch_forecast_volatility": garch_vol_str,
        "value_at_risk_95": var95_str,
        "derived_signal_direction": technically_derived_direction,
        "signal_trigger_context": signal_context,
        "key_technical_indicators": technical_indicators,
        "account_balance_usd": account_balance,
        "max_allowable_leverage": max_leverage,
        "minimum_target_rr": min_requested_rr or 1.5 # Use passed value or default
    }
    data_json = json.dumps(market_info, indent=2)
    logger.debug(f"{log_prefix} Data prepared for GPT:\n{data_json}")

    # --- REVISED GPT PROMPT ---
    prompt = f"""You are a cryptocurrency trading analyst evaluating a potential trade setup based on a MACD crossover signal combined with a trend filter (SMA 200) and trend strength (ADX).
The system detected a potential signal: '{technically_derived_direction}'.
Your task is to EVALUATE this signal using the provided market data and technical indicators, and provide actionable parameters if appropriate.

Market Data & Signal Context:
{data_json}

Instructions:
1.  **Evaluate Signal:** Assess the '{technically_derived_direction}' signal based on the `signal_trigger_context`. Look for **confirming** factors (e.g., RSI moving out of overbought/oversold in signal direction, supportive Stochastics, price action respecting SMAs/BBands) and **strong contradicting** factors (e.g., clear divergence on RSI/MACD against the signal, price hitting major resistance/support against the signal, very low ADX < 15 suggesting chop despite signal).
2.  **Determine Action:**
    *   If the initial signal has *some* confirmation OR lacks *strong* contradictions, **confirm the `trade_direction`** as '{technically_derived_direction}'.
    *   Output `trade_direction: 'hold'` ONLY if there are **significant contradictions** from multiple important indicators (e.g., strong divergence + price hitting strong opposing level) OR if the overall context makes the signal very unreliable (e.g., ADX extremely low, price trapped tightly between bands).
3.  **Refine Parameters (if action is 'long' or 'short'):**
    *   `optimal_entry`: Suggest a tactical entry near `current_price` or slightly better (e.g., pullback to EMA/SMA or breakout confirmation). Avoid entries far from `current_price`. Justify briefly.
    *   `stop_loss`: Place logically below recent lows/support (for long) or above recent highs/resistance (for short), potentially using `ATR` (e.g., current_price +/- 1.5 * ATR). Ensure it's not too tight.
    *   `take_profit`: Calculate based on `optimal_entry` and `stop_loss` to meet or exceed the `minimum_target_rr` ({min_requested_rr or 1.5}). Check if this TP level looks reasonable based on chart structure (e.g., next resistance/support level).
    *   `leverage`: Suggest a reasonable leverage (integer >= 1) based on `max_allowable_leverage` and perceived risk/volatility. Be conservative (e.g., 3-10x range usually).
    *   `position_size_usd`: Optional: Suggest a size based on `account_balance_usd`, `leverage`, and risk (e.g., risk 1-2% of account per trade).
4.  **Provide Confidence:** Assign a `confidence_score` (0.0-1.0) based on the *degree of confirmation* and the *absence of strong contradictions*. Score > 0.6 requires decent confirmation and alignment.
5.  **Justify:** Explain your reasoning concisely in the `analysis` sections (`signal_evaluation`, `technical_analysis`, `risk_assessment`).

Respond ONLY with a single, valid JSON object containing the specified fields. Do not include explanations outside the JSON structure.
"""
    # --- END OF REVISED PROMPT ---

    try:
        logger.info(f"{log_prefix} Sending request to GPT (gpt-4o-mini) to evaluate '{technically_derived_direction}' signal...")
        # Consider using a cheaper/faster model if possible, like gpt-4o-mini or gpt-3.5-turbo, if results are acceptable
        response = openai_client.chat.completions.create(
                    model="gpt-4.1", # Use a capable model
                    messages=[
                        {"role": "system", "content": "You are a crypto trading analyst evaluating technical signals. Respond ONLY in JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3, # Lower temperature for more deterministic analysis
                    max_tokens=1800 # Allow slightly more tokens for detailed justification
                )
        gpt_output = response.choices[0].message.content
        print(gpt_output)
        logger.debug(f"{log_prefix} Raw GPT Output received:\n```json\n{gpt_output}\n```")
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


# --- Parse GPT Parameters (Largely unchanged, but logging context helps) ---
def parse_gpt_trading_parameters(gpt_output_str: str, symbol_for_log: str = "") -> Dict[str, Any]:
    """Parse GPT-generated trading parameters (evaluation focused)."""
    log_prefix = f"[{symbol_for_log} Parse]"
    parsed_data = {
        'optimal_entry': None, 'stop_loss': None, 'take_profit': None, 'trade_direction': 'hold', # Default hold
        'leverage': None, 'position_size_usd': None, 'estimated_profit': None, 'confidence_score': 0.0, # Default conf 0
        'analysis': {'signal_evaluation': None, 'technical_analysis': None, 'risk_assessment': None, 'market_outlook': None, 'raw_text': gpt_output_str}
    }
    try:
        # Clean potential markdown ```json ... ``` artifacts
        match = re.search(r'```json\s*(\{.*?\})\s*```', gpt_output_str, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1)
            logger.debug(f"{log_prefix} Extracted JSON block from markdown.")
        else:
            json_str = gpt_output_str # Assume it's already plain JSON

        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise json.JSONDecodeError("GPT output was not a JSON object", json_str, 0)
        logger.debug(f"{log_prefix} Successfully decoded JSON from GPT.")

        # Helper to safely get float/int values
        def get_numeric(key, target_type=float):
            val = data.get(key)
            try:
                num_val = target_type(val)
                return num_val if np.isfinite(num_val) else None
            except (ValueError, TypeError):
                return None

        # Get core trade parameters
        parsed_data['optimal_entry'] = get_numeric('optimal_entry', float)
        parsed_data['stop_loss'] = get_numeric('stop_loss', float)
        parsed_data['take_profit'] = get_numeric('take_profit', float)
        parsed_data['position_size_usd'] = get_numeric('position_size_usd', float)
        parsed_data['estimated_profit'] = get_numeric('estimated_profit', float)
        parsed_data['confidence_score'] = get_numeric('confidence_score', float)
        parsed_data['leverage'] = get_numeric('leverage', int)
        if parsed_data['leverage'] is not None and parsed_data['leverage'] < 1:
            logger.warning(f"{log_prefix} GPT suggested leverage < 1 ({parsed_data['leverage']}), setting to None.")
            parsed_data['leverage'] = None

        # Trade Direction (Crucial - reflects GPT's evaluation)
        direction = data.get('trade_direction')
        if isinstance(direction, str) and direction.lower() in ['long', 'short', 'hold']:
            parsed_data['trade_direction'] = direction.lower()
            logger.info(f"{log_prefix} GPT evaluated signal resulted in direction: '{parsed_data['trade_direction']}'")
        else:
             logger.warning(f"{log_prefix} Invalid or missing trade_direction from GPT: '{direction}'. Defaulting to 'hold'.")
             parsed_data['trade_direction'] = 'hold'

        # Validate parameters *if* GPT suggests a trade
        if parsed_data['trade_direction'] in ['long', 'short']:
             required_params = ['optimal_entry', 'stop_loss', 'take_profit', 'confidence_score']
             missing_params = [p for p in required_params if parsed_data[p] is None]
             if missing_params:
                 logger.warning(f"{log_prefix} GPT suggested '{parsed_data['trade_direction']}' but missing {missing_params}. Forcing 'hold'.")
                 parsed_data['trade_direction'] = 'hold'
             # Basic R/R sanity check
             elif parsed_data['optimal_entry'] and parsed_data['stop_loss'] and parsed_data['take_profit']:
                 entry = parsed_data['optimal_entry']
                 sl = parsed_data['stop_loss']
                 tp = parsed_data['take_profit']
                 if parsed_data['trade_direction'] == 'long' and not (sl < entry < tp):
                     logger.warning(f"{log_prefix} GPT Long levels illogical (SL:{sl}, E:{entry}, TP:{tp}). Forcing 'hold'.")
                     parsed_data['trade_direction'] = 'hold'
                 elif parsed_data['trade_direction'] == 'short' and not (tp < entry < sl):
                     logger.warning(f"{log_prefix} GPT Short levels illogical (TP:{tp}, E:{entry}, SL:{sl}). Forcing 'hold'.")
                     parsed_data['trade_direction'] = 'hold'
                 else:
                     risk = abs(entry - sl); reward = abs(tp - entry)
                     if risk < 1e-9: # Avoid division by zero
                         logger.warning(f"{log_prefix} GPT suggested zero risk (Entry=SL={entry}). Forcing 'hold'.")
                         parsed_data['trade_direction'] = 'hold'
                     # R/R check is now done in the scan filter based on min_risk_reward_ratio

        # Process analysis section
        analysis_dict = data.get('analysis')
        if isinstance(analysis_dict, dict):
            for key in ['signal_evaluation', 'technical_analysis', 'risk_assessment', 'market_outlook']:
                val = analysis_dict.get(key)
                if isinstance(val, str) and val.strip():
                    parsed_data['analysis'][key] = val.strip()
        else:
            logger.warning(f"{log_prefix} 'analysis' section missing or invalid in GPT response.")
            # Try to get any top-level text fields as fallback
            for key in ['evaluation', 'rationale', 'summary', 'technical_analysis', 'risk_assessment']:
                 if key in data and isinstance(data[key], str) and not parsed_data['analysis']['signal_evaluation']:
                      parsed_data['analysis']['signal_evaluation'] = data[key] # Put first found text here
                      break

    except json.JSONDecodeError as e:
        logger.error(f"{log_prefix} Failed to decode JSON from GPT: {e}. Raw: {gpt_output_str[:500]}...")
        parsed_data['trade_direction'] = 'hold'
        parsed_data['analysis']['signal_evaluation'] = f"Error: Failed to parse GPT JSON response. {e}"
    except Exception as e:
        logger.error(f"{log_prefix} Unexpected error parsing GPT response: {e}", exc_info=True)
        parsed_data['trade_direction'] = 'hold'
        parsed_data['analysis']['signal_evaluation'] = f"Error: Unexpected error parsing GPT response. {e}"

    # Clear trade params if final decision is 'hold'
    if parsed_data['trade_direction'] == 'hold':
        logger.info(f"{log_prefix} Final direction is 'hold', clearing trade parameters.")
        parsed_data['optimal_entry'] = None; parsed_data['stop_loss'] = None; parsed_data['take_profit'] = None
        parsed_data['leverage'] = None; parsed_data['position_size_usd'] = None; parsed_data['estimated_profit'] = None
        # Keep confidence score as it might reflect confidence in *not* trading

    logger.debug(f"{log_prefix} Parsed GPT Params: {json.dumps(parsed_data, indent=2)}")
    return parsed_data

# --- Backtesting (CRITICAL: Align with NEW Signal Trigger) ---
def backtest_strategy(
    df_with_indicators: pd.DataFrame,
    # Pass the *initial* technical direction and parameters relevant for backtest
    initial_signal_direction: str, # 'long' or 'short' based on MACD/Trend/ADX
    min_adx_for_trigger: float = 20.0, # Make sure this matches perform_single_analysis
    min_rr_ratio_target: float = 1.5, # Target R:R for simulated trades
    atr_sl_multiplier: float = 1.5, # How many ATRs for stop loss
    max_trade_duration_bars: int = 96, # e.g., 4 days on 1h bars
    min_bars_between_trades: int = 5 # Cooldown period
) -> Dict[str, Any]:
    """Backtest strategy based on *historical occurrences* of the MACD/Trend/ADX signal."""
    symbol_log = df_with_indicators.get('symbol', 'Unknown')
    log_prefix = f"[{symbol_log} Backtest]"
    results = {'strategy_score': 0.0, 'trade_analysis': {'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': None, 'avg_profit': None, 'avg_loss': None, 'profit_factor': None, 'total_profit': None, 'largest_win': None, 'largest_loss': None, 'average_trade_duration': None}, 'recommendation': 'N/A', 'warnings': []}
    trade_analysis = results['trade_analysis']

    # Backtest only makes sense if the initial signal was long or short
    if initial_signal_direction not in ['long', 'short']:
        results['recommendation'] = f"Backtest skipped: Initial signal direction is '{initial_signal_direction}'."
        logger.info(f"{log_prefix} {results['recommendation']}")
        return results

    # --- Requirements for the NEW backtest logic ---
    required_cols = ['open', 'high', 'low', 'close', 'ATR', 'MACD', 'Signal_Line', 'SMA_200', 'ADX']
    # Drop rows where any essential column for signal generation or simulation is missing
    df_clean = df_with_indicators[required_cols].dropna()
    # Shift MACD/Signal to easily check previous bar's condition for crossover
    df_clean['prev_MACD'] = df_clean['MACD'].shift(1)
    df_clean['prev_Signal_Line'] = df_clean['Signal_Line'].shift(1)
    df_clean = df_clean.dropna() # Drop rows where shifted values are NaN (first row)

    if len(df_clean) < 50: # Need sufficient data for meaningful backtest
        results['recommendation'] = "Backtest skipped: Insufficient clean data for simulation."
        results['warnings'].append(f"Insufficient clean data points ({len(df_clean)}) for backtest.")
        logger.warning(f"{log_prefix} {results['recommendation']}")
        return results
    logger.info(f"{log_prefix} Starting backtest for initial signal '{initial_signal_direction}' over {len(df_clean)} potential bars.")

    trades = []; last_entry_index_loc = -1 - min_bars_between_trades

    # --- Simulation Loop (Iterate through cleaned data) ---
    entry_count = 0
    # Use iloc for reliable indexing after dropping NaNs
    for i_loc in range(len(df_clean) - 1): # Iterate up to second to last bar to allow entry on next bar's open
        signal_row = df_clean.iloc[i_loc]
        entry_bar_ts = df_clean.index[i_loc + 1] # Timestamp of the bar we would enter on
        # Need to get the 'open' price from the original df for the entry bar
        try:
            entry_price = df_with_indicators.loc[entry_bar_ts, 'open']
            if pd.isna(entry_price): continue # Skip if entry price is invalid
        except KeyError:
            continue # Skip if timestamp not found in original df (shouldn't happen often)


        # Check cooldown based on iloc index
        if i_loc <= last_entry_index_loc + min_bars_between_trades:
             continue

        # --- Check for MACD/Trend/ADX Signal Conditions on the signal_row ---
        price = signal_row['close']
        sma200 = signal_row['SMA_200']
        adx = signal_row['ADX']
        macd = signal_row['MACD']
        signal = signal_row['Signal_Line']
        prev_macd = signal_row['prev_MACD']
        prev_signal = signal_row['prev_Signal_Line']
        atr = signal_row['ATR'] # ATR from the signal bar for SL calculation

        # Check data validity for this specific bar
        if pd.isna(price) or pd.isna(sma200) or pd.isna(adx) or pd.isna(macd) or \
           pd.isna(signal) or pd.isna(prev_macd) or pd.isna(prev_signal) or pd.isna(atr) or atr <= 1e-9:
            continue # Skip bar if essential data is missing

        setup_found = False
        stop_loss_calc, take_profit_calc = 0.0, 0.0

        # --- Apply the NEW signal logic ---
        is_trending_up = price > sma200
        is_trending_down = price < sma200
        is_adx_strong = adx >= min_adx_for_trigger
        macd_crossed_up = prev_macd <= prev_signal and macd > signal
        macd_crossed_down = prev_macd >= prev_signal and macd < signal

        current_signal_matches_request = False
        if initial_signal_direction == 'long' and is_trending_up and macd_crossed_up and is_adx_strong:
            current_signal_matches_request = True
        elif initial_signal_direction == 'short' and is_trending_down and macd_crossed_down and is_adx_strong:
            current_signal_matches_request = True

        if current_signal_matches_request:
            # Calculate potential SL/TP based on *entry price* and signal bar's ATR
            risk_per_point = atr * atr_sl_multiplier
            if initial_signal_direction == 'long':
                stop_loss_calc = entry_price - risk_per_point
                take_profit_calc = entry_price + risk_per_point * min_rr_ratio_target
            else: # Short
                stop_loss_calc = entry_price + risk_per_point
                take_profit_calc = entry_price - risk_per_point * min_rr_ratio_target

            # Validate levels make sense relative to entry price
            if initial_signal_direction == 'long' and stop_loss_calc < entry_price < take_profit_calc:
                setup_found = True
                logger.debug(f"{log_prefix} Long setup found at {signal_row.name} (MACD X Up, Trend Up, ADX Ok). Entry@{entry_bar_ts}: {entry_price:.4f}, SL: {stop_loss_calc:.4f}, TP: {take_profit_calc:.4f}, ATR: {atr:.4f}")
            elif initial_signal_direction == 'short' and take_profit_calc < entry_price < stop_loss_calc:
                setup_found = True
                logger.debug(f"{log_prefix} Short setup found at {signal_row.name} (MACD X Down, Trend Down, ADX Ok). Entry@{entry_bar_ts}: {entry_price:.4f}, SL: {stop_loss_calc:.4f}, TP: {take_profit_calc:.4f}, ATR: {atr:.4f}")
            else:
                 logger.debug(f"{log_prefix} Setup found at {signal_row.name} but levels invalid (E:{entry_price:.4f}, SL:{stop_loss_calc:.4f}, TP:{take_profit_calc:.4f})")


        # --- Simulate Trade if setup found ---
        if setup_found:
            entry_count += 1
            outcome, exit_price, exit_index_loc_in_df_clean = None, None, -1
            entry_index_loc_in_df_clean = i_loc + 1 # The index in df_clean where the trade starts

            # Simulate bar-by-bar using df_with_indicators for High/Low checks
            max_exit_loc_in_df_clean = min(len(df_clean) - 1, entry_index_loc_in_df_clean + max_trade_duration_bars)

            for k_loc in range(entry_index_loc_in_df_clean, max_exit_loc_in_df_clean + 1):
                current_bar_ts = df_clean.index[k_loc]
                # Get High/Low from the original full df_with_indicators
                try:
                    current_bar_orig = df_with_indicators.loc[current_bar_ts]
                    current_low, current_high = current_bar_orig['low'], current_bar_orig['high']
                    if pd.isna(current_low) or pd.isna(current_high): continue # Skip bar if H/L invalid
                except KeyError:
                    continue # Skip if TS not in original df

                if initial_signal_direction == 'long':
                    if current_low <= stop_loss_calc: # SL Hit
                        outcome, exit_price, exit_index_loc_in_df_clean = 'loss', stop_loss_calc, k_loc
                        logger.debug(f"{log_prefix} Trade {entry_count} (Long) SL hit at {current_bar_ts}")
                        break
                    elif current_high >= take_profit_calc: # TP Hit
                        outcome, exit_price, exit_index_loc_in_df_clean = 'win', take_profit_calc, k_loc
                        logger.debug(f"{log_prefix} Trade {entry_count} (Long) TP hit at {current_bar_ts}")
                        break
                elif initial_signal_direction == 'short':
                    if current_high >= stop_loss_calc: # SL Hit
                        outcome, exit_price, exit_index_loc_in_df_clean = 'loss', stop_loss_calc, k_loc
                        logger.debug(f"{log_prefix} Trade {entry_count} (Short) SL hit at {current_bar_ts}")
                        break
                    elif current_low <= take_profit_calc: # TP Hit
                        outcome, exit_price, exit_index_loc_in_df_clean = 'win', take_profit_calc, k_loc
                        logger.debug(f"{log_prefix} Trade {entry_count} (Short) TP hit at {current_bar_ts}")
                        break

            # If neither SL nor TP hit within duration, exit at close of last bar checked
            if outcome is None:
                exit_index_loc_in_df_clean = max_exit_loc_in_df_clean
                exit_bar_ts = df_clean.index[exit_index_loc_in_df_clean]
                try:
                    exit_price = df_with_indicators.loc[exit_bar_ts, 'close']
                    if pd.isna(exit_price): # Handle rare case exit price is NaN
                         outcome, exit_price = 'error', entry_price # Consider it break even or small loss?
                         logger.warning(f"{log_prefix} Trade {entry_count} exited due to duration but close price was NaN at {exit_bar_ts}. Marked as error.")
                    else:
                         if initial_signal_direction == 'long': outcome = 'win' if exit_price > entry_price else 'loss'
                         else: outcome = 'win' if exit_price < entry_price else 'loss'
                         logger.debug(f"{log_prefix} Trade {entry_count} exited due to duration at {exit_bar_ts} - Outcome: {outcome}")
                except KeyError:
                    outcome, exit_price = 'error', entry_price
                    logger.error(f"{log_prefix} Trade {entry_count} exit bar TS {exit_bar_ts} not found in original df. Marked as error.")


            # Record the trade if valid outcome
            if outcome and outcome != 'error' and exit_price is not None and exit_index_loc_in_df_clean != -1:
                profit_points = (exit_price - entry_price) if initial_signal_direction == 'long' else (entry_price - exit_price)
                trade_duration = exit_index_loc_in_df_clean - entry_index_loc_in_df_clean # Duration in bars relative to df_clean indices
                trades.append({'profit': profit_points, 'duration': trade_duration, 'outcome': outcome})
                last_entry_index_loc = entry_index_loc_in_df_clean # Update cooldown index (use location in df_clean)

    # --- Analyze Backtest Results (Calculations are the same, interpretation changes) ---
    trade_analysis['total_trades'] = len(trades)
    logger.info(f"{log_prefix} Backtest simulation completed. Found {len(trades)} historical trades based on '{initial_signal_direction}' MACD/Trend/ADX signal.")
    if trades:
        # Corrected line
        profits = np.array([t['profit'] for t in trades]); durations = np.array([t['duration'] for t in trades]); outcomes = [t['outcome'] for t in trades] # <--- Change outcomes to trades here
        trade_analysis['winning_trades'] = sum(1 for o in outcomes if o == 'win'); trade_analysis['losing_trades'] = len(trades) - trade_analysis['winning_trades']
        if len(trades) > 0: trade_analysis['win_rate'] = round(trade_analysis['winning_trades'] / len(trades), 3)
        winning_profits = profits[profits > 0]; losing_profits = profits[profits <= 0]; gross_profit = np.sum(winning_profits); gross_loss = abs(np.sum(losing_profits))
        trade_analysis['avg_profit'] = round(np.mean(winning_profits), 4) if len(winning_profits) > 0 else 0.0
        trade_analysis['avg_loss'] = round(abs(np.mean(losing_profits)), 4) if len(losing_profits) > 0 else 0.0
        if gross_loss > 1e-9: trade_analysis['profit_factor'] = round(gross_profit / gross_loss, 2)
        elif gross_profit > 1e-9: trade_analysis['profit_factor'] = float('inf') # Handle case of no losses
        else: trade_analysis['profit_factor'] = 0.0
        trade_analysis['total_profit'] = round(np.sum(profits), 4); trade_analysis['largest_win'] = round(np.max(winning_profits), 4) if len(winning_profits) > 0 else 0.0
        trade_analysis['largest_loss'] = round(np.min(losing_profits), 4) if len(losing_profits) > 0 else 0.0; trade_analysis['average_trade_duration'] = round(np.mean(durations), 1)

        # --- Calculate Strategy Score (Same logic, but now reflects the NEW signal) ---
        score = 0.0; pf = trade_analysis['profit_factor']; wr = trade_analysis['win_rate']; num_trades = len(trades)
        avg_w = trade_analysis['avg_profit']; avg_l = trade_analysis['avg_loss']
        # Adjust scoring weights if needed
        if pf is not None:
            if pf == float('inf') or pf >= 2.5: score += 0.35 # Slightly less weight?
            elif pf >= 1.7: score += 0.25
            elif pf >= 1.3: score += 0.15
            elif pf >= 1.0: score += 0.05
            else: score -= 0.25 # Penalize losing strategies more
        if wr is not None:
            if wr >= 0.6: score += 0.30
            elif wr >= 0.5: score += 0.20
            elif wr >= 0.4: score += 0.10
            else: score -= 0.15
        if num_trades >= 30: score += 0.15 # Slightly less weight for just #trades
        elif num_trades >= 15: score += 0.10
        elif num_trades < 5: score -= 0.20 # Penalize low sample size more
        if avg_w is not None and avg_l is not None and avg_l > 1e-9:
             ratio = avg_w / avg_l # Avg Win / Avg Loss ratio
             if ratio >= 2.0: score += 0.10
             elif ratio >= 1.2: score += 0.05
        if trade_analysis['total_profit'] is not None and trade_analysis['total_profit'] <= 0: score -= 0.30 # Strong penalty for net loss

        results['strategy_score'] = max(0.0, min(1.0, round(score, 2)))
        logger.info(f"{log_prefix} Backtest Score: {results['strategy_score']:.2f} (Trades:{num_trades}, WR:{wr if wr else 'N/A':.1%}, PF:{pf if pf else 'N/A':.2f})")

        # --- Recommendation and Warnings (Reflect the NEW strategy) ---
        strategy_name = "MACD/Trend/ADX"
        if results['strategy_score'] >= 0.70: results['recommendation'] = f"Strong historical performance for {strategy_name} trigger."
        elif results['strategy_score'] >= 0.55: results['recommendation'] = f"Good historical performance for {strategy_name} trigger." # Threshold lowered slightly
        elif results['strategy_score'] >= 0.40: results['recommendation'] = f"Moderate/Mixed historical performance for {strategy_name} trigger."
        else: results['recommendation'] = f"Poor historical performance for {strategy_name} trigger."
        # Warnings remain relevant
        if wr is not None and wr < 0.45: results['warnings'].append(f"Low Win Rate ({wr:.1%})") # Threshold adjusted
        if pf is not None and pf < 1.25: results['warnings'].append(f"Low Profit Factor ({pf:.2f})") # Threshold adjusted
        if num_trades < 10: results['warnings'].append(f"Low Trade Count ({num_trades}).")
        if trade_analysis['total_profit'] is not None and trade_analysis['total_profit'] <= 0: results['warnings'].append("Overall loss in backtest.")
        if avg_l is not None and avg_l > 1e-9 and trade_analysis['largest_loss'] is not None and abs(trade_analysis['largest_loss']) > 3 * avg_l:
             results['warnings'].append("Largest loss significantly exceeds average loss.")
    else:
        results['recommendation'] = f"No historical {strategy_name} setups found to backtest."; results['warnings'].append("No qualifying historical trade setups found."); results['strategy_score'] = 0.0
        logger.info(f"{log_prefix} {results['recommendation']}")

    return results


# --- Core Analysis Function (Updated Signal Trigger and Backtest Call) ---
async def perform_single_analysis(
    symbol: str, timeframe: str, lookback: int, account_balance: float, max_leverage: float,
    min_requested_rr: Optional[float] = None, # Pass this down from ScanRequest
    min_adx_for_trigger: float = 20.0 # Ensure this matches backtester and ScanRequest default/value
) -> AnalysisResponse:
    start_time_ticker = time.time()
    log_prefix = f"[{symbol} ({timeframe})]"
    logger.info(f"{log_prefix} Starting analysis...")
    analysis_result = AnalysisResponse(symbol=symbol, timeframe=timeframe)
    df_indicators = None # Define upfront

    # --- Step 1: Fetch Data ---
    try:
        df_raw = await get_real_time_data(symbol, timeframe, lookback) # Uses new fetcher with retries
        # Minimum required length depends on longest indicator (SMA200 + buffer)
        min_req = 250 # Need enough for SMA200 + MACD calcs + GARCH/VaR
        if df_raw.empty or len(df_raw) < min_req:
            analysis_result.error = f"Insufficient data ({len(df_raw)} fetched/{min_req} required)"; logger.warning(f"{log_prefix} {analysis_result.error}"); return analysis_result
        analysis_result.currentPrice = float(df_raw['close'].iloc[-1])
        logger.info(f"{log_prefix} Data fetched ({len(df_raw)} bars). Price: {analysis_result.currentPrice:.4f}")
    except (ConnectionError, ConnectionAbortedError, ValueError, PermissionError, RuntimeError) as e:
         logger.error(f"{log_prefix} Data fetching failed: {e}")
         analysis_result.error = f"Data Fetch Error: {e}"
         return analysis_result
    except Exception as e:
         logger.error(f"{log_prefix} Unexpected error during Data Fetch: {e}", exc_info=True)
         analysis_result.error = f"Unexpected Data Fetch Error: {e}"
         return analysis_result


    # --- Step 2: Apply Indicators ---
    try:
        df_raw['symbol'] = symbol
        df_indicators = await asyncio.to_thread(apply_technical_indicators, df_raw)
        if df_indicators.empty or len(df_indicators) < 2: # Need at least 2 rows for crossover check
             raise ValueError("Indicator calculation resulted in insufficient data (< 2 rows)")

        latest_indicators_raw = df_indicators.iloc[-1].to_dict()
        indicator_payload = {
            field_name: float(value)
            for field_name, model_field in IndicatorsData.model_fields.items()
            if pd.notna(value := latest_indicators_raw.get(model_field.alias or field_name)) and np.isfinite(value)
        }
        analysis_result.indicators = IndicatorsData(**indicator_payload)
        logger.info(f"{log_prefix} Indicators applied.")
        # Log key indicators for the new signal
        logger.debug(
            f"{log_prefix} Latest Indicators - "
            f"MACD: {analysis_result.indicators.MACD:.4f}, "
            f"Signal: {analysis_result.indicators.Signal_Line:.4f}, "
            f"SMA200: {analysis_result.indicators.SMA_200:.4f}, "
            f"ADX: {analysis_result.indicators.ADX:.2f}, "
            f"Close: {analysis_result.currentPrice:.4f}"
        )
    except Exception as e:
        logger.error(f"{log_prefix} Indicator calculation error: {e}", exc_info=True)
        analysis_result.error = f"Indicator Error: {e}"
        return analysis_result

    # --- Step 2.5: Determine Technical Trigger (NEW: MACD Cross + SMA200 Trend + ADX) ---
    technical_signal_direction = "hold" # Default
    price = analysis_result.currentPrice
    sma200 = analysis_result.indicators.SMA_200 if analysis_result.indicators else None
    latest_adx = analysis_result.indicators.ADX if analysis_result.indicators else None
    macd_line = analysis_result.indicators.MACD if analysis_result.indicators else None
    signal_line = analysis_result.indicators.Signal_Line if analysis_result.indicators else None

    # Get previous bar's MACD values for crossover detection
    prev_macd_line = None
    prev_signal_line = None
    try:
        if df_indicators is not None and len(df_indicators) >= 2:
            # Use .get with default None for safety
            prev_indicators_raw = df_indicators.iloc[-2].to_dict()
            prev_macd_line = prev_indicators_raw.get('MACD')
            prev_signal_line = prev_indicators_raw.get('Signal_Line')
    except IndexError:
        logger.warning(f"{log_prefix} Could not get previous indicator row for crossover check.")

    # Check if all necessary data is present and finite
    has_data = all(v is not None and np.isfinite(v) for v in [
        price, sma200, latest_adx, macd_line, signal_line, prev_macd_line, prev_signal_line
    ])

    if has_data:
        is_trending_up = price > sma200
        is_trending_down = price < sma200
        is_adx_strong = latest_adx >= min_adx_for_trigger
        macd_crossed_up = prev_macd_line <= prev_signal_line and macd_line > signal_line
        macd_crossed_down = prev_macd_line >= prev_signal_line and macd_line < signal_line

        # Determine Signal based on combined conditions
        if is_trending_up and macd_crossed_up and is_adx_strong:
            technical_signal_direction = "long"
        elif is_trending_down and macd_crossed_down and is_adx_strong:
            technical_signal_direction = "short"

        # Log the outcome and conditions clearly
        macd_cross_log = "N/A"
        if macd_crossed_up: macd_cross_log = f"UP ({macd_line:.4f} > {signal_line:.4f} from <= prev)"
        elif macd_crossed_down: macd_cross_log = f"DOWN ({macd_line:.4f} < {signal_line:.4f} from >= prev)"
        else: macd_cross_log = f"None ({macd_line:.4f} vs {signal_line:.4f})"

        trend_log = f"Trend: {'UP' if is_trending_up else 'DOWN' if is_trending_down else 'CHOP/NA'}"
        adx_log = f"ADX: {latest_adx:.2f}"
        conditions_met = f"TrendOK={is_trending_up or is_trending_down}, CrossOK={macd_crossed_up or macd_crossed_down}, ADX_OK={is_adx_strong}"

        logger.info(f"{log_prefix} Technical signal trigger check: {technical_signal_direction.upper()} ({macd_cross_log}, {trend_log}, {adx_log}, {conditions_met})")
    else:
        logger.warning(f"{log_prefix} Cannot determine technical signal: Missing key data (Price, SMA200, ADX, MACD/Signal current/prev).")
        analysis_result.error = (analysis_result.error or "") + "; Warning: Missing data for signal trigger"
        # Keep technical_signal_direction as 'hold'

    # --- Step 3: Statistical Models (GARCH, VaR - Unchanged) ---
    garch_vol, var95 = None, None
    try:
        if df_indicators is not None and 'returns' in df_indicators and not df_indicators['returns'].isnull().all():
            returns = df_indicators['returns']
            garch_task = asyncio.to_thread(fit_garch_model, returns, symbol)
            var_task = asyncio.to_thread(calculate_var, returns, 0.95, symbol)
            garch_vol, var95 = await asyncio.gather(garch_task, var_task)
            analysis_result.modelOutput = ModelOutputData(garchVolatility=garch_vol, var95=var95)
            logger.info(f"{log_prefix} GARCH/VaR calculated (Vol: {garch_vol}, VaR95: {var95}).")
        else:
            logger.warning(f"{log_prefix} Skipping GARCH/VaR (no valid returns data).")
            analysis_result.modelOutput = ModelOutputData() # Ensure modelOutput exists
    except Exception as e:
        logger.error(f"{log_prefix} Stat model error: {e}", exc_info=True)
        analysis_result.error = (analysis_result.error or "") + f"; Stat Model Error: {e}"
        analysis_result.modelOutput = ModelOutputData()


    # --- Step 4: GPT Evaluation (Conditional Call based on NEW signal) ---
    gpt_parsed_output = None
    if openai_client and technical_signal_direction != 'hold':
        logger.info(f"{log_prefix} Technical signal is '{technical_signal_direction}', proceeding with GPT evaluation.")
        try:
            gpt_raw_output = await asyncio.to_thread(
                 gpt_generate_trading_parameters, # Pass the function name
                 # Pass the arguments for the function:
                 df_indicators, symbol, timeframe, account_balance, max_leverage,
                 garch_vol, var95,
                 technical_signal_direction,
                 min_requested_rr
            )
            # gpt_raw_output = await asyncio.to_thread( # If gpt_generate_trading_parameters is sync
            #      gpt_generate_trading_parameters,
            #      df_indicators, symbol, timeframe, account_balance, max_leverage,
            #      garch_vol, var95,
            #      technical_signal_direction,
            #      min_requested_rr
            # )
            logger.debug(f"{log_prefix} RAW GPT Response Received:\n{gpt_raw_output}") # Log raw only if needed
            gpt_parsed_output = parse_gpt_trading_parameters(gpt_raw_output, symbol)

            if gpt_parsed_output:
                 analysis_result.gptParams = GptTradingParams(**{k: v for k, v in gpt_parsed_output.items() if k != 'analysis'})
                 analysis_result.gptAnalysis = GptAnalysisText(**gpt_parsed_output.get('analysis', {}))
                 logger.info(f"{log_prefix} GPT evaluation complete. EvalDir: {analysis_result.gptParams.trade_direction}, Conf: {analysis_result.gptParams.confidence_score}")
                 # Check for errors reported by GPT or parsing
                 gpt_internal_error_msg = gpt_parsed_output.get("error") or (gpt_parsed_output.get('analysis', {}).get('signal_evaluation') or "").startswith("Error:")
                 if gpt_internal_error_msg:
                     err_detail = gpt_parsed_output.get("details", gpt_internal_error_msg)
                     logger.warning(f"{log_prefix} GPT issue: {err_detail}")
                     analysis_result.error = (analysis_result.error or "") + f"; GPT Warning: {str(err_detail)[:100]}"
            else:
                 logger.error(f"{log_prefix} GPT parsing failed unexpectedly.")
                 analysis_result.error = (analysis_result.error or "") + "; GPT parsing failed"
                 analysis_result.gptAnalysis = GptAnalysisText(raw_text="GPT parsing failed.")
                 analysis_result.gptParams = GptTradingParams(trade_direction='hold') # Default to hold

        except Exception as e:
            logger.error(f"{log_prefix} GPT processing error: {e}", exc_info=True)
            analysis_result.error = (analysis_result.error or "") + f"; GPT Processing Error: {e}"
            gpt_parsed_output = None
            analysis_result.gptAnalysis = GptAnalysisText(raw_text=f"Error during GPT step: {e}")
            analysis_result.gptParams = GptTradingParams(trade_direction='hold')

    else:
        reason = "OpenAI client missing" if not openai_client else "Technical signal is 'hold'"
        logger.info(f"{log_prefix} Skipping GPT evaluation: {reason}.")
        analysis_result.gptAnalysis = GptAnalysisText(signal_evaluation=f"GPT skipped: {reason}")
        analysis_result.gptParams = GptTradingParams(trade_direction='hold', confidence_score=None)
        # Ensure gpt_parsed_output is suitable for backtest check below
        gpt_parsed_output = {'trade_direction': 'hold'}


    # --- Step 5: Backtesting (Uses NEW backtest_strategy aligned with MACD signal) ---
    # Backtest is run if the *initial* technical signal was valid ('long' or 'short'),
    # regardless of GPT's final evaluation. The backtest score reflects the raw signal's historical performance.
    if technical_signal_direction in ['long', 'short']:
        logger.info(f"{log_prefix} Running backtest for initial technical signal: {technical_signal_direction}")
        try:
            # Call the REVISED backtest function
            backtest_raw = await asyncio.to_thread(
                backtest_strategy,
                df_indicators.copy(), # Pass df
                technical_signal_direction, # Pass the initial signal
                min_adx_for_trigger=min_adx_for_trigger, # Pass ADX threshold used
                min_rr_ratio_target=(min_requested_rr or 1.5) # Use requested R/R or default for simulation TP
                # Add other backtest params if they become configurable in AnalysisRequest
            )

            analysis_result.backtest = BacktestResultsData(
                strategy_score=backtest_raw.get('strategy_score'),
                trade_analysis=BacktestTradeAnalysis(**backtest_raw.get('trade_analysis', {})),
                recommendation=backtest_raw.get('recommendation'), warnings=backtest_raw.get('warnings', [])
            )
            logger.info(f"{log_prefix} Backtest done. Score: {analysis_result.backtest.strategy_score}, Rec: {analysis_result.backtest.recommendation}")
        except Exception as e:
            logger.error(f"{log_prefix} Backtesting error: {e}", exc_info=True)
            analysis_result.error = (analysis_result.error or "") + f"; Backtesting Error: {e}"
            analysis_result.backtest = BacktestResultsData(recommendation=f"Backtest failed: {e}", warnings=[f"Error: {e}"])
    else:
        logger.info(f"{log_prefix} Skipping backtest (Initial signal: {technical_signal_direction}).")
        analysis_result.backtest = BacktestResultsData(recommendation=f"Backtest skipped: InitialSignal={technical_signal_direction}")


    # Finalization
    duration = time.time() - start_time_ticker
    if not analysis_result.error:
        logger.info(f"{log_prefix} Analysis successful ({duration:.2f}s).")
    else:
        # Log level adjusted based on severity (e.g., is it just a warning?)
        log_level = logging.WARNING if "Warning:" in analysis_result.error or "skipped:" in analysis_result.error else logging.ERROR
        logger.log(log_level, f"{log_prefix} Analysis finished with issues ({duration:.2f}s). Status: {analysis_result.error}")

    return analysis_result


# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI startup: Initializing...")
    # Load markets asynchronously
    asyncio.create_task(load_exchange_markets(binance_futures))
    # Test OpenAI connection in background
    if openai_client: asyncio.create_task(test_openai_connection(openai_client))

# Tickers (Unchanged)
@app.get("/api/crypto/tickers", response_model=TickersResponse, tags=["Utility"])
async def get_crypto_tickers_endpoint():
    # ... (keep existing code) ...
    logger.info("API: Request received for /tickers")
    if binance_futures is None: raise HTTPException(status_code=503, detail="Exchange unavailable")
    if not binance_futures.markets:
        logger.warning("API Tickers: Markets not loaded, attempting load.")
        # Use await here as it's critical for the endpoint
        if not await load_exchange_markets(binance_futures):
             raise HTTPException(status_code=503, detail="Failed load markets.")
    try:
        markets = binance_futures.markets
        if not markets:
            logger.error("API Tickers: Markets dictionary is empty after load attempt.")
            raise HTTPException(status_code=500, detail="Markets loaded empty.")
        # Filter for active USDT-settled perpetual futures
        tickers = sorted([
            m['symbol'] for m in markets.values()
            if m.get('swap') and m.get('linear') and m.get('quote')=='USDT' and m.get('settle')=='USDT' and m.get('active')
        ])
        logger.info(f"API Tickers: Found {len(tickers)} active USDT linear perpetuals.")
        if not tickers: logger.warning("API Tickers: No active USDT linear perpetuals found.")
        return TickersResponse(tickers=tickers)
    except Exception as e:
        logger.error(f"API Tickers Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error getting tickers: {e}")


# Analyze (Uses updated perform_single_analysis)
@app.post("/api/crypto/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_crypto_endpoint(request: AnalysisRequest):
    logger.info(f"API: Analyze request received for {request.symbol} ({request.timeframe})")
    if binance_futures is None: raise HTTPException(status_code=503, detail="Exchange unavailable.")

    try:
        # Pass parameters using model_dump()
        analysis_params = request.model_dump()
        # Add params not in AnalysisRequest but needed by perform_single_analysis
        analysis_params['min_requested_rr'] = 1.5 # Default R/R for single analysis? Or make it a query param?
        analysis_params['min_adx_for_trigger'] = 20.0 # Default ADX for single analysis

        result = await perform_single_analysis(**analysis_params)

        # Error Handling (same logic as before)
        if result.error:
            err = result.error; status = 500; log_func = logger.error
            if "Warning:" in err or "skipped:" in err or "Insufficient data" in err: status = 200; log_func = logger.warning # Treat warnings/skips as OK
            elif "Invalid symbol" in err: status = 400
            elif "Network error" in err or "Connection error" in err or "Failed to load markets" in err or "Data Fetch Error" in err: status = 504
            elif "Rate limit exceeded" in err or "ConnectionAbortedError" in err: status = 429
            elif "Authentication Error" in err: status = 401
            elif "OpenAI" in err or "GPT" in err: status = 502
            elif "Exchange error" in err: status = 503

            log_func(f"API Analyze: Request for {request.symbol} finished with status {status}. Detail: {err}")
            if status != 200: raise HTTPException(status_code=status, detail=err)

        logger.info(f"API Analyze: Request for {request.symbol} completed successfully.")
        return result

    except HTTPException as h: raise h
    except Exception as e:
        logger.error(f"API Analyze Endpoint Error for {request.symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected endpoint internal error: {e}")

# Scan (Uses updated perform_single_analysis, passes R/R and ADX down)
@app.post("/api/crypto/scan", response_model=ScanResponse, tags=["Scanning"])
async def scan_market_endpoint(request: ScanRequest):
    logger.critical("--- SCAN ENDPOINT ENTERED ---")
    scan_start_time = time.time()
    logger.info(f"API Scan: Starting request: {request.model_dump_json(exclude_defaults=False)}")

    # --- Lookback Warning ---
    # Use 1500 as a general limit for Binance 1m fetches
    binance_1m_limit = 1500
    effective_lookback = request.lookback
    if request.timeframe == '1m' and request.lookback > binance_1m_limit:
         logger.warning(f"API Scan: Lookback {request.lookback} for 1m timeframe likely exceeds fetch limits ({binance_1m_limit}). Analysis will use max available.")
         effective_lookback = binance_1m_limit # Use capped value for analysis call estimate
    elif request.lookback > 1500: # General warning for other timeframes too
         logger.warning(f"API Scan: Lookback {request.lookback} might exceed typical fetch limits (~1500). Analysis will use max available.")
         # effective_lookback = 1500 # Could cap here too if needed

    # --- Basic Setup Checks ---
    if binance_futures is None: logger.error("API Scan Abort: Exchange unavailable."); raise HTTPException(status_code=503, detail="Exchange unavailable")
    if not binance_futures.markets:
        logger.warning("API Scan: Markets not loaded, attempting load...")
        if not await load_exchange_markets(binance_futures): logger.error("API Scan Abort: Failed load markets."); raise HTTPException(status_code=503, detail="Failed load markets.")

    # --- 1. Get Tickers ---
    try:
        # Re-fetch or use cached markets? Assume load_markets refreshed if needed.
        markets = binance_futures.markets
        if not markets: raise ValueError("Markets unavailable post-load.")
        # Same filtering as before
        all_tickers = sorted([ m['symbol'] for m in markets.values() if m.get('swap') and m.get('linear') and m.get('quote')=='USDT' and m.get('settle')=='USDT' and m.get('active')])
        logger.info(f"API Scan: Found {len(all_tickers)} active USDT linear perpetuals.")
        if not all_tickers:
            return ScanResponse(scan_parameters=request, total_tickers_attempted=0, total_tickers_succeeded=0, ticker_start_index=request.ticker_start_index, ticker_end_index=request.ticker_end_index, total_opportunities_found=0, top_opportunities=[], errors={})
    except Exception as e: logger.error(f"API Scan Tickers Error: {e}", exc_info=True); raise HTTPException(status_code=502, detail=f"Ticker retrieval error: {e}")

    # --- 2. Determine BTC Trend (Market Regime) ---
    btc_trend_state = "UNKNOWN"; apply_btc_filter = request.filter_by_btc_trend
    if apply_btc_filter:
        logger.info(f"API Scan: BTC Trend filter enabled. Fetching BTC data for timeframe {request.timeframe}...")
        try:
            btc_symbol = "BTC/USDT:USDT"
            # Ensure enough lookback for SMAs on BTC, respecting limits
            btc_lookback = max(250, effective_lookback) # Use the potentially capped lookback
            # Fetch BTC data
            df_btc_raw = await get_real_time_data(btc_symbol, request.timeframe, btc_lookback)
            if df_btc_raw.empty or len(df_btc_raw) < 205: # Need enough for SMA200
                 logger.warning(f"API Scan: Insufficient BTC data ({len(df_btc_raw)} bars) for trend on {request.timeframe}. Disabling BTC filter.")
                 apply_btc_filter = False
            else:
                df_btc_raw['symbol'] = btc_symbol
                df_btc_indicators = await asyncio.to_thread(apply_technical_indicators, df_btc_raw)
                btc_latest = df_btc_indicators.iloc[-1]
                btc_price = btc_latest.get('close'); btc_sma50 = btc_latest.get('SMA_50'); btc_sma200 = btc_latest.get('SMA_200')
                if all(v is not None and np.isfinite(v) for v in [btc_price, btc_sma50, btc_sma200]):
                    if btc_price > btc_sma50 > btc_sma200: btc_trend_state = "UPTREND"
                    elif btc_price < btc_sma50 < btc_sma200: btc_trend_state = "DOWNTREND"
                    else: btc_trend_state = "CHOPPY"
                    logger.info(f"API Scan: Determined BTC Trend ({request.timeframe}): {btc_trend_state} (P: {btc_price:.2f}, S50: {btc_sma50:.2f}, S200: {btc_sma200:.2f})")
                else:
                    logger.warning(f"API Scan: Could not determine BTC trend (missing Price/SMA50/SMA200). Disabling BTC filter.")
                    apply_btc_filter = False; btc_trend_state = "UNKNOWN"
        except Exception as e: logger.error(f"API Scan: Error getting BTC trend: {e}. Disabling BTC filter.", exc_info=True); apply_btc_filter = False; btc_trend_state = "ERROR"
    else: logger.info("API Scan: BTC Trend filter is disabled via request.")

    # --- 3. Select Tickers (Same logic as before) ---
    tickers_to_scan = []; total_available = len(all_tickers); start_index = request.ticker_start_index or 0; end_index = request.ticker_end_index
    actual_end_index_for_response = None; slice_desc = ""
    # ... (keep the slicing logic from the original script) ...
    if start_index >= total_available > 0: logger.warning(f"API Scan: Start index {start_index} out of bounds."); return ScanResponse(...) # Empty response
    if start_index < 0: start_index = 0
    if end_index is not None:
        if end_index <= start_index: tickers_to_scan = []; slice_desc = f"invalid slice [{start_index}:{end_index}]"
        else: actual_end = min(end_index, total_available); tickers_to_scan = all_tickers[start_index:actual_end]; slice_desc = f"requested slice [{start_index}:{end_index}], actual [{start_index}:{actual_end}]"
        actual_end_index_for_response = end_index # Report requested end index
    elif request.max_tickers is not None and request.max_tickers > 0:
        limit = request.max_tickers; actual_end = min(start_index + limit, total_available); tickers_to_scan = all_tickers[start_index:actual_end]; slice_desc = f"max_tickers={limit} from {start_index}, actual [{start_index}:{actual_end}]"
        actual_end_index_for_response = actual_end # Report actual end index
    elif request.max_tickers == 0: tickers_to_scan = []; slice_desc = "max_tickers=0"; actual_end_index_for_response = start_index
    else: actual_end = total_available; tickers_to_scan = all_tickers[start_index:actual_end]; slice_desc = f"all from {start_index}, actual [{start_index}:{actual_end}]"; actual_end_index_for_response = actual_end

    logger.info(f"API Scan: Selected {len(tickers_to_scan)} tickers to analyze ({slice_desc}).")
    total_attempted = len(tickers_to_scan)
    if total_attempted == 0: return ScanResponse(...) # Empty response

    # --- 4. Run Concurrently ---
    semaphore = asyncio.Semaphore(request.max_concurrent_tasks)
    tasks = []
    processed_count = 0; progress_lock = asyncio.Lock()
    log_interval = max(1, total_attempted // 20 if total_attempted > 0 else 1)

    async def analyze_with_semaphore_wrapper(ticker):
        nonlocal processed_count
        result = None; task_start_time = time.time()
        try:
            async with semaphore:
                # Pass relevant ScanRequest parameters to the analysis function
                result = await perform_single_analysis(
                    symbol=ticker,
                    timeframe=request.timeframe,
                    lookback=request.lookback, # Pass original request, fetch func handles limits
                    account_balance=request.accountBalance,
                    max_leverage=request.maxLeverage,
                    min_requested_rr=request.min_risk_reward_ratio, # Pass R/R
                    min_adx_for_trigger=request.min_adx # Pass ADX threshold
                )
        except Exception as e: logger.error(f"API Scan Wrapper Error for {ticker}: {e}", exc_info=True); result = AnalysisResponse(symbol=ticker, timeframe=request.timeframe, error=f"Scan Wrapper Exception: {e}")
        finally:
            task_duration = time.time() - task_start_time
            async with progress_lock: processed_count += 1; current_count = processed_count
            if current_count % log_interval == 0 or current_count == total_attempted: logger.info(f"API Scan Progress: {current_count}/{total_attempted} tasks completed (Last: {ticker} took {task_duration:.2f}s).")
        # Ensure a valid AnalysisResponse is always returned
        return result if result is not None else AnalysisResponse(symbol=ticker, timeframe=request.timeframe, error="Unknown wrapper failure")

    logger.info(f"API Scan: Creating {total_attempted} analysis tasks...")
    tasks = [analyze_with_semaphore_wrapper(ticker) for ticker in tickers_to_scan]

    logger.info(f"API Scan: Gathering results for {total_attempted} tasks (Concurrency: {request.max_concurrent_tasks})...")
    analysis_results_raw: List[Any] = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info(f"API Scan: Finished gathering {len(analysis_results_raw)} task results.")

    # --- 5. Process Results & Apply Filters ---
    successful_analyses_count = 0; analysis_errors = {}; opportunities_passing_filter = []
    logger.info("API Scan: Starting detailed processing and filtering of results...")

    for i, res_or_exc in enumerate(analysis_results_raw):
        symbol = tickers_to_scan[i] if i < len(tickers_to_scan) else f"UnknownTicker_{i}"
        logger.debug(f"--- Processing item {i+1}/{len(analysis_results_raw)} for symbol [{symbol}] ---")

        # Handle Exceptions from gather
        if isinstance(res_or_exc, Exception):
            logger.error(f"API Scan: Task [{symbol}] UNHANDLED Exception: {res_or_exc}", exc_info=False)
            analysis_errors[symbol] = f"Unhandled Task Exception: {str(res_or_exc)[:200]}"
            continue

        # Process AnalysisResponse
        elif isinstance(res_or_exc, AnalysisResponse):
            result: AnalysisResponse = res_or_exc
            # Check for critical errors reported by the analysis function itself
            # Treat insufficient data as a skip, not a hard error for the scan summary
            is_skip = result.error and ("Insufficient data" in result.error or "skipped:" in result.error or "Warning:" in result.error)
            is_critical_error = result.error and not is_skip

            if is_critical_error:
                logger.error(f"API Scan: Critical error for [{result.symbol}]: {result.error}")
                analysis_errors[result.symbol] = f"Analysis Critical Error: {result.error}"
                continue # Skip further processing for this ticker
            elif is_skip:
                 logger.info(f"API Scan: Skipping filter check for [{result.symbol}] due to: {result.error}")
                 successful_analyses_count += 1 # Count as attempted and processed, but skipped filtering
                 continue # Skip filtering for this ticker
            else:
                 # No error or only warnings, proceed to filtering
                 successful_analyses_count += 1

            # --- Extract data safely for filtering ---
            gpt_params = result.gptParams; bt_results = result.backtest; indicators = result.indicators
            bt_analysis = bt_results.trade_analysis if bt_results and bt_results.trade_analysis else None
            # Use GPT's final evaluated direction for filtering trade direction
            eval_direction = gpt_params.trade_direction if gpt_params else None
            gpt_conf = gpt_params.confidence_score if gpt_params and gpt_params.confidence_score is not None else None
            bt_score = bt_results.strategy_score if bt_results and bt_results.strategy_score is not None else None
            current_price = result.currentPrice

            # Log pre-filter state clearly
            log_pre_filter = (
                f"[{result.symbol}] Pre-filter: EvalDir='{eval_direction or 'N/A'}', "
                f"GPTConf={f'{gpt_conf:.2f}' if gpt_conf is not None else 'N/A'}, "
                f"BTScore={f'{bt_score:.2f}' if bt_score is not None else 'N/A'} "
            )
            if bt_analysis:
                 log_pre_filter += (
                     f"(BT Trades={bt_analysis.total_trades}, "
                     f"WR={f'{bt_analysis.win_rate:.1%}' if bt_analysis.win_rate is not None else 'N/A'}, "
                     f"PF={'inf' if bt_analysis.profit_factor == float('inf') else f'{bt_analysis.profit_factor:.2f}' if bt_analysis.profit_factor is not None else 'N/A'}) "
                 )
            if indicators:
                 log_pre_filter += (
                     f"ADX={f'{indicators.ADX:.1f}' if indicators.ADX is not None else 'N/A'}, "
                     f"SMA50={f'{indicators.SMA_50:.2f}' if indicators.SMA_50 is not None else 'N/A'}, "
                     f"SMA200={f'{indicators.SMA_200:.2f}' if indicators.SMA_200 is not None else 'N/A'} "
                 )
            logger.info(log_pre_filter)


            # --- Filtering Logic ---
            passes_filters = True
            filter_fail_reason = ""

            # F1: Basic Direction & Confidence/Score
            if eval_direction not in ['long', 'short']: passes_filters = False; filter_fail_reason = f"Eval Direction not tradeable ('{eval_direction}')"
            elif request.trade_direction and eval_direction != request.trade_direction: passes_filters = False; filter_fail_reason = f"Direction mismatch (Req: {request.trade_direction}, Eval: {eval_direction})"
            elif gpt_conf is None or gpt_conf < request.min_gpt_confidence: passes_filters = False; filter_fail_reason = f"GPT Conf too low (Req >= {request.min_gpt_confidence:.2f}, Got: {f'{gpt_conf:.2f}' if gpt_conf is not None else 'N/A'})"
            elif bt_score is None or bt_score < request.min_backtest_score: passes_filters = False; filter_fail_reason = f"BT Score too low (Req >= {request.min_backtest_score:.2f}, Got: {f'{bt_score:.2f}' if bt_score is not None else 'N/A'})"

            # F2: Backtest Stats (Only if F1 passed)
            if passes_filters:
                if bt_analysis is None: passes_filters = False; filter_fail_reason = "Backtest analysis missing"
                elif request.min_backtest_trades is not None and (bt_analysis.total_trades < request.min_backtest_trades): passes_filters = False; filter_fail_reason = f"BT Trades too low (Req >= {request.min_backtest_trades}, Got: {bt_analysis.total_trades})"
                elif request.min_backtest_win_rate is not None and (bt_analysis.win_rate is None or bt_analysis.win_rate < request.min_backtest_win_rate): passes_filters = False; filter_fail_reason = f"BT Win Rate too low (Req >= {request.min_backtest_win_rate:.1%}, Got: {f'{bt_analysis.win_rate:.1%}' if bt_analysis.win_rate is not None else 'N/A'})"
                elif request.min_backtest_profit_factor is not None and (bt_analysis.profit_factor is None or (bt_analysis.profit_factor != float('inf') and bt_analysis.profit_factor < request.min_backtest_profit_factor)): passes_filters = False; filter_fail_reason = f"BT Profit Factor too low (Req >= {request.min_backtest_profit_factor:.2f}, Got: {'inf' if bt_analysis.profit_factor == float('inf') else f'{bt_analysis.profit_factor:.2f}' if bt_analysis.profit_factor is not None else 'N/A'})"

            # F3: Risk/Reward Ratio (Only if F1, F2 passed)
            if passes_filters and request.min_risk_reward_ratio is not None and request.min_risk_reward_ratio > 0:
                entry = gpt_params.optimal_entry if gpt_params else None; sl = gpt_params.stop_loss if gpt_params else None; tp = gpt_params.take_profit if gpt_params else None
                rr_ratio = None
                if entry is not None and sl is not None and tp is not None:
                     risk = abs(entry - sl); reward = abs(tp - entry)
                     if risk > 1e-9: rr_ratio = reward / risk
                     else: rr_ratio = None # Avoid division by zero if entry=sl

                if rr_ratio is None or rr_ratio < request.min_risk_reward_ratio:
                     passes_filters = False; filter_fail_reason = f"R/R Ratio too low (Req >= {request.min_risk_reward_ratio:.2f}, Got: {f'{rr_ratio:.2f}' if rr_ratio is not None else 'N/A'})"

            # F4: Indicator Filters (ADX, SMA Alignment) (Only if F1, F2, F3 passed)
            if passes_filters:
                if indicators is None: passes_filters = False; filter_fail_reason = "Indicator data missing"
                # ADX filter applied during signal generation, but can double-check here if needed
                # elif request.min_adx is not None and request.min_adx > 0:
                #      adx = indicators.ADX
                #      if adx is None or adx < request.min_adx: passes_filters = False; filter_fail_reason = f"ADX too low (Req >= {request.min_adx:.1f}, Got: {f'{adx:.1f}' if adx is not None else 'N/A'})"
                elif request.require_sma_alignment:
                     sma50 = indicators.SMA_50; sma200 = indicators.SMA_200; price = current_price; sma_aligned = False
                     if price is not None and sma50 is not None and sma200 is not None:
                         if eval_direction == 'long' and price > sma50 > sma200: sma_aligned = True
                         elif eval_direction == 'short' and price < sma50 < sma200: sma_aligned = True
                     if not sma_aligned: passes_filters = False; filter_fail_reason = f"SMA alignment failed (Req: {request.require_sma_alignment}, Dir: {eval_direction}, P:{price:.2f}, S50:{sma50:.2f}, S200:{sma200:.2f})"

            # F5: BTC Trend Alignment (Only if F1-F4 passed and filter enabled)
            if passes_filters and apply_btc_filter and btc_trend_state not in ["UNKNOWN", "ERROR"]:
                if eval_direction == 'long' and btc_trend_state != 'UPTREND':
                    passes_filters = False; filter_fail_reason = f"BTC Trend not UPTREND (Is: {btc_trend_state})"
                elif eval_direction == 'short' and btc_trend_state != 'DOWNTREND':
                    passes_filters = False; filter_fail_reason = f"BTC Trend not DOWNTREND (Is: {btc_trend_state})"
                # else: Allow if trend matches or BTC is choppy

            # --- Add Opportunity if Passes All Filters ---
            if passes_filters:
                logger.info(f"[{result.symbol}] PASSED ALL FILTERS. Adding to opportunities.")
                score_g = float(gpt_conf) if gpt_conf is not None else 0.0
                score_b = float(bt_score) if bt_score is not None else 0.0
                # Weighting: GPT Conf (prediction quality) 60%, BT Score (historical signal performance) 40%
                combined_score = round((score_g * 0.6) + (score_b * 0.4), 3)

                # Extract a concise summary from GPT analysis
                summary = "Analysis details unavailable."
                if result.gptAnalysis:
                    # Prioritize signal_evaluation, then technical_analysis, then raw text if needed
                    primary_summary = result.gptAnalysis.signal_evaluation or result.gptAnalysis.technical_analysis
                    if primary_summary and isinstance(primary_summary, str) and "Error:" not in primary_summary:
                         summary = (primary_summary.split('.')[0] + '.').strip() # First sentence
                    elif result.gptAnalysis.raw_text and isinstance(result.gptAnalysis.raw_text, str):
                         summary = (result.gptAnalysis.raw_text.split('.')[0] + '.').strip() # Fallback to raw
                    summary = summary[:147] + "..." if len(summary) > 150 else summary # Truncate

                opportunity = ScanResultItem(
                    rank=0, symbol=result.symbol, timeframe=result.timeframe, currentPrice=current_price,
                    gptConfidence=gpt_conf, backtestScore=bt_score, combinedScore=combined_score,
                    tradeDirection=eval_direction, # Use the evaluated direction
                    optimalEntry=gpt_params.optimal_entry if gpt_params else None,
                    stopLoss=gpt_params.stop_loss if gpt_params else None,
                    takeProfit=gpt_params.take_profit if gpt_params else None,
                    gptAnalysisSummary=summary
                )
                opportunities_passing_filter.append(opportunity)
            else:
                 # Log the failure reason only if it wasn't a planned skip
                 if not is_skip:
                     logger.info(f"[{result.symbol}] FAILED FILTERS. Reason: {filter_fail_reason}.")

        # Unexpected type handling
        else:
             logger.error(f"API Scan: UNEXPECTED result type [{symbol}]: {type(res_or_exc)}.")
             analysis_errors[symbol] = f"Unexpected Result Type: {type(res_or_exc).__name__}"

    # --- End of Result Processing Loop ---
    logger.info(f"API Scan: Finished processing results. Succeeded/Filterable: {successful_analyses_count}, Errors: {len(analysis_errors)}, Passed Filters: {len(opportunities_passing_filter)}.")

    # --- 6. Rank Opportunities ---
    if opportunities_passing_filter:
        logger.info(f"API Scan: Sorting {len(opportunities_passing_filter)} opportunities by Combined Score...")
        # Sort by combined score descending
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
        total_tickers_succeeded=successful_analyses_count, # Reflects analyses run without critical errors
        ticker_start_index=start_index,
        ticker_end_index=actual_end_index_for_response, # Report the actual end index used
        total_opportunities_found=len(opportunities_passing_filter),
        top_opportunities=top_opportunities,
        errors=analysis_errors
    )


# --- Main Execution (Unchanged) ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8029))
    host = os.getenv("HOST", "127.0.0.1")
    reload_flag = os.getenv("UVICORN_RELOAD", "false").lower() in ("true", "1", "yes")
    uvicorn_log_level = LOG_LEVEL.lower()
    logger.info(f"Setting Uvicorn log level to: {uvicorn_log_level}")
    print("\n" + "="*30); print(" --- Starting FastAPI Server ---")
    print(f" Host: {host}"); print(f" Port: {port}"); print(f" Auto-Reload: {reload_flag}")
    print(f" Logging Level: {LOG_LEVEL}"); print(f" OpenAI Client Initialized: {openai_client is not None}")
    print(f" CCXT Client Initialized: {binance_futures is not None}")
    print(f" Default Max Concurrent Scan Tasks: {DEFAULT_MAX_CONCURRENT_TASKS}") # Default, overridden by request
    print(f" Initial Signal Trigger: MACD Cross + SMA200 Trend + ADX >= 20") # State the new signal
    print("="*30 + "\n")
    uvicorn.run(
        "__main__:app", host=host, port=port, reload=reload_flag, log_level=uvicorn_log_level
    )

