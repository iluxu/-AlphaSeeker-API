# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import time
import ccxt
import logging
from datetime import datetime
from arch import arch_model
# from sklearn.preprocessing import StandardScaler # Pas utilisé ici
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

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # INFO par défaut, DEBUG pour plus de détails
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s') # Ajout numéro ligne
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(log_formatter)
logger = logging.getLogger()
logger.setLevel(LOG_LEVEL)
if logger.hasHandlers(): logger.handlers.clear()
logger.addHandler(log_handler)
logger.critical("--- Logging Initialized (Level: %s) ---", LOG_LEVEL)

# --- Configuration ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="ConvergenceWarning", category=UserWarning)
load_dotenv()
DEFAULT_MAX_CONCURRENT_TASKS = 5 

# --- CCXT Initialization ---
binance_futures = None
try:
    binance_futures = ccxt.binanceusdm({
        'enableRateLimit': True,
        'options': { 'adjustForTimeDifference': True },
        'timeout': 30000, 
        'rateLimit': 150 
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
        markets = await asyncio.to_thread(exchange.load_markets, True) 
        if markets:
             logger.info(f"Successfully loaded {len(markets)} markets for {exchange.id}.")
             return True
        else:
             logger.warning(f"Market loading returned empty for {exchange.id}.")
             return False
    except ccxt.AuthenticationError as e: logger.error(f"CCXT Auth Error: {e}", exc_info=False); return False
    except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e: logger.error(f"Network/Timeout Error loading markets: {e}", exc_info=False); return False
    except ccxt.ExchangeError as e: logger.error(f"Exchange error loading markets: {e}", exc_info=False); return False
    except Exception as e: logger.error(f"Unexpected error loading markets: {e}", exc_info=True); return False

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
         await asyncio.to_thread(client.models.list) 
         logger.info("OpenAI connection test successful.")
     except openai.AuthenticationError: logger.error("OpenAI Authentication Error.")
     except openai.RateLimitError: logger.warning("OpenAI Rate Limit potentially hit during test.")
     except Exception as e: logger.error(f"OpenAI connection test failed: {e}")

# --- FastAPI App ---
app = FastAPI(
    title="Crypto Trading Analysis & Scanning API",
    description="API for technical analysis (Breakout Strategy), GPT-driven evaluation, backtesting, and market scanning.",
    version="1.5.2_Breakout_Full" # Version mise à jour
)

# --- Pydantic Models ---

class TickerRequest(BaseModel): pass
class TickersResponse(BaseModel): tickers: List[str]

class AnalysisRequest(BaseModel): 
    symbol: str = Field(..., example="BTC/USDT:USDT")
    timeframe: str = Field(default="1h", example="1h")
    lookback: int = Field(default=1000, ge=50) 
    accountBalance: float = Field(default=1000.0, ge=0)
    maxLeverage: float = Field(default=10.0, ge=1)

class IndicatorsData(BaseModel):
    # Indicateurs clés pour Breakout et simulation
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None # Garder pour référence
    volume: Optional[float] = None
    ATR: Optional[float] = None 
    RSI: Optional[float] = None 
    Rolling_High_20: Optional[float] = None # Exemple, sera dynamique
    Rolling_Low_20: Optional[float] = None  # Exemple, sera dynamique
    Avg_Volume_20: Optional[float] = None # Exemple, sera dynamique
    # Indicateurs optionnels pour contexte
    SMA_50: Optional[float] = None 
    SMA_200: Optional[float] = None
    ADX: Optional[float] = None; 
    Bollinger_Upper: Optional[float] = None 
    Bollinger_Middle: Optional[float] = None 
    Bollinger_Lower: Optional[float] = None 
    returns: Optional[float] = None
    
    model_config = ConfigDict(populate_by_name=True, extra='allow') # IMPORTANT pour les noms dynamiques

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
    optimal_entry: Optional[float] = None; stop_loss: Optional[float] = None; take_profit: Optional[float] = None
    trade_direction: Optional[str] = None
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

# --- ScanRequest --- 
class ScanRequest(BaseModel):
    ticker_start_index: Optional[int] = Field(default=0, ge=0)
    ticker_end_index: Optional[int] = Field(default=None, ge=0)
    timeframe: str = Field(default="1h", description="Candle timeframe (e.g., '1m', '5m', '1h').")
    max_tickers: Optional[int] = Field(default=100, description="Limit tickers if end_index is null.")
    top_n: int = Field(default=10, ge=1)

    # Core Filters
    min_gpt_confidence: float = Field(default=0.60, ge=0, le=1)
    min_backtest_score: float = Field(default=0.50, ge=0, le=1) 
    trade_direction: Optional[str] = Field(default=None, pattern="^(long|short)$")
    filter_by_btc_trend: Optional[bool] = Field(default=False) 

    # Backtest Filters
    min_backtest_trades: Optional[int] = Field(default=10, ge=0)
    min_backtest_win_rate: Optional[float] = Field(default=0.40, ge=0, le=1) 
    min_backtest_profit_factor: Optional[float] = Field(default=1.4, ge=0)  

    # GPT/Risk Filter
    min_risk_reward_ratio: Optional[float] = Field(default=1.8, ge=0) 

    # Indicator Filters (Non pertinents pour Breakout pur, gardés désactivés par défaut)
    # min_adx: Optional[float] = Field(default=None, ge=0) 
    require_sma_alignment: Optional[bool] = Field(default=False) 

    # --- Breakout Strategy Specific Params (Optionnel - Ajouter ici si configurable via API) ---
    # breakout_period: int = Field(default=20, ge=5)
    # volume_multiplier: float = Field(default=1.5, ge=1.0)
    # rsi_threshold_long: float = Field(default=55.0, ge=50, le=100)
    # rsi_threshold_short: float = Field(default=45.0, ge=0, le=50) 

    # Analysis Config
    lookback: int = Field(default=1000, ge=50) 
    accountBalance: float = Field(default=5000.0, ge=0)
    maxLeverage: float = Field(default=5.0, ge=1) 
    max_concurrent_tasks: int = Field(default=10, ge=1) 

# ScanResultItem (Inchangé)
class ScanResultItem(BaseModel):
    rank: int; symbol: str; timeframe: str; currentPrice: Optional[float] = None
    gptConfidence: Optional[float] = None; backtestScore: Optional[float] = None; combinedScore: Optional[float] = None
    tradeDirection: Optional[str] = None; optimalEntry: Optional[float] = None; stopLoss: Optional[float] = None
    takeProfit: Optional[float] = None; gptAnalysisSummary: Optional[str] = None

# ScanResponse (Inchangé)
class ScanResponse(BaseModel):
    scan_parameters: ScanRequest; total_tickers_attempted: int; total_tickers_succeeded: int
    ticker_start_index: Optional[int]; ticker_end_index: Optional[int] 
    total_opportunities_found: int; top_opportunities: List[ScanResultItem]
    errors: Dict[str, str] = Field(default={})


# --- Helper Functions ---

# --- Data Fetcher --- 
async def get_real_time_data(symbol: str, timeframe: str = "1d", limit: int = 1000, retries: int = 2) -> pd.DataFrame:
    """Fetch OHLCV data with retries for network issues."""
    logger.debug(f"[{symbol}] Attempting to fetch {limit} candles for timeframe {timeframe}")
    global binance_futures
    if binance_futures is None: raise ConnectionError("CCXT exchange instance is not available.")
    if not binance_futures.markets:
         logger.warning(f"[{symbol}] Markets not loaded, attempting synchronous load...")
         try: await asyncio.to_thread(binance_futures.load_markets, True); logger.info(f"[{symbol}] Markets loaded successfully (sync).")
         except Exception as e: raise ConnectionError(f"Failed to load markets synchronously: {e}") from e

    last_exception = None
    for attempt in range(retries + 1):
        try:
            # Use asyncio.to_thread for the synchronous CCXT call
            ohlcv = await asyncio.to_thread(binance_futures.fetch_ohlcv, symbol, timeframe, None, limit)

            if not ohlcv:
                logger.warning(f"[{symbol}] No OHLCV data returned from fetch_ohlcv (Attempt {attempt+1}).")
                return pd.DataFrame() # Return empty DF, don't retry if exchange returns nothing

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)
            # Convert to numeric, coercing errors. Crucial step.
            df = df.apply(pd.to_numeric, errors='coerce') 
            # Drop rows where ANY of the core values are NaN after coercion
            df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True) 

            if df.empty: logger.warning(f"[{symbol}] DataFrame became empty after type conversion/NaN drop.")
            else: logger.debug(f"[{symbol}] Fetched {len(df)} valid candles.")
            return df # Success

        except ccxt.BadSymbol as e: raise ValueError(f"Invalid symbol '{symbol}'") from e 
        except ccxt.AuthenticationError as e: raise PermissionError("CCXT Authentication Error") from e 
        except (ccxt.RateLimitExceeded, ccxt.DDoSProtection, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, ccxt.NetworkError) as e:
            last_exception = e; wait_time = 2 ** attempt 
            logger.warning(f"[{symbol}] Network/RateLimit error (Attempt {attempt+1}/{retries+1}): {type(e).__name__}. Retrying in {wait_time}s...")
            if attempt < retries: await asyncio.sleep(wait_time)
            else: logger.error(f"[{symbol}] Max retries reached."); raise ConnectionAbortedError(f"Failed after {retries} retries: {e}") from e
        except Exception as e:
            logger.error(f"[{symbol}] Unexpected error fetching data: {e}", exc_info=True)
            raise RuntimeError(f"Failed to fetch data for {symbol}") from e
            
    # Should only be reached if loop finishes without success or exception (unlikely)
    raise last_exception if last_exception else RuntimeError(f"Unknown error fetching data for {symbol}")

# --- Indicator Calculation Functions (compute_*) ---
# (ASSUREZ-VOUS QUE CES FONCTIONS SONT DÉFINIES ICI OU IMPORTÉES CORRECTEMENT)
def compute_rsi(series, window=14):
    if not isinstance(series, pd.Series) or series.empty: return pd.Series(dtype=float)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).fillna(0)
    loss = -delta.where(delta < 0, 0.0).fillna(0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan) # Avoid division by zero
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50) # Fill initial NaNs with 50

def compute_atr(df, window=14):
    if not isinstance(df, pd.DataFrame) or not {'high', 'low', 'close'}.issubset(df.columns): return pd.Series(index=df.index, dtype=float)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    atr = tr.ewm(com=window - 1, min_periods=window).mean()
    return atr

def compute_adx(df, window=14):
    # --- Placeholder ADX - REMPLACEZ PAR VOTRE CODE COMPLET ---
    if not isinstance(df, pd.DataFrame) or not {'high', 'low', 'close'}.issubset(df.columns): return pd.Series(index=df.index, dtype=float)
    # Simuler un calcul ADX pour l'exemple
    adx_series = pd.Series(np.random.rand(len(df)) * 40 + 10, index=df.index) # Valeurs aléatoires entre 10 et 50
    logger.warning("Using PLACEHOLDER ADX calculation!") 
    return adx_series.fillna(20)
    # --- FIN PLACEHOLDER ---

def compute_bollinger_bands(series, window=20):
    if not isinstance(series, pd.Series) or series.empty: return (pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float))
    sma = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, sma, lower

# --- Apply Indicators (Breakout Version) ---
def apply_technical_indicators(df: pd.DataFrame, breakout_period: int = 20) -> pd.DataFrame:
    """
    Apply technical indicators, including breakout levels and volume average.
    Ensures required columns exist and handles potential calculation errors.
    """
    df_copy = df.copy()
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Convert core columns to numeric, coercing errors
    for col in required_cols:
        if col in df_copy:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        else:
            logger.warning(f"Column '{col}' missing for indicator calculation. Assigning NaN.")
            df_copy[col] = np.nan 
            
    # Drop rows where any core value is NaN AFTER coercion - vital for calculations
    df_copy.dropna(subset=required_cols, inplace=True)
    
    if df_copy.empty:
         logger.warning("DataFrame empty after dropping rows with NaN in core columns. Cannot calculate indicators.")
         return df_copy # Retourner vide

    # Calculate returns safely after ensuring 'close' is numeric and non-zero
    safe_close = df_copy['close'].replace(0, np.nan)
    df_copy['returns'] = np.log(safe_close / safe_close.shift(1)).fillna(0)
    
    df_len = len(df_copy)
    symbol_log = df_copy.get('symbol', 'UnknownSymbol') 
    logger.debug(f"[{symbol_log}] Applying indicators to {df_len} valid candles for breakout period {breakout_period}.")

    # --- Breakout Calculations ---
    high_col = f'Rolling_High_{breakout_period}'
    low_col = f'Rolling_Low_{breakout_period}'
    vol_col = f'Avg_Volume_{breakout_period}'
    min_periods_breakout = max(1, int(breakout_period * 0.8)) # Exiger au moins 80% de la période

    # Appliquer rolling seulement si assez de données NON-NAN existent
    if df_copy['high'].notna().sum() >= min_periods_breakout: 
        df_copy[high_col] = df_copy['high'].shift(1).rolling(window=breakout_period, min_periods=min_periods_breakout).max()
    else: df_copy[high_col] = np.nan
        
    if df_copy['low'].notna().sum() >= min_periods_breakout:
        df_copy[low_col] = df_copy['low'].shift(1).rolling(window=breakout_period, min_periods=min_periods_breakout).min()
    else: df_copy[low_col] = np.nan
        
    if df_copy['volume'].notna().sum() >= min_periods_breakout:
        df_copy[vol_col] = df_copy['volume'].shift(1).rolling(window=breakout_period, min_periods=min_periods_breakout).mean()
        df_copy[vol_col] = df_copy[vol_col].replace(0, np.nan) 
    else: df_copy[vol_col] = np.nan
    # --- End Breakout Calculations ---

    # Helper function 
    def assign_if_enough_data(col_name, min_len_needed, calculation_lambda, required_source_cols=None):
        source_ok = True
        if required_source_cols:
            for src_col in required_source_cols:
                # Vérifier si la colonne existe et a ASSEZ de points valides
                if src_col not in df_copy or df_copy[src_col].notna().sum() < min_len_needed :
                    source_ok = False
                    logger.debug(f"[{symbol_log}] Skipping {col_name}, required source '{src_col}' missing or insufficient valid data.")
                    break
                    
        if source_ok and df_len >= min_len_needed:
             try: 
                 result = calculation_lambda()
                 df_copy[col_name] = pd.to_numeric(result, errors='coerce')
             except Exception as e: 
                 logger.error(f"[{symbol_log}] Error calculating {col_name}: {e}", exc_info=False)
                 df_copy[col_name] = np.nan
        else:
             if col_name not in df_copy: df_copy[col_name] = np.nan
             if source_ok: logger.debug(f"[{symbol_log}] Skipping {col_name}, need {min_len_needed} valid source data points.")

    # --- Other Necessary/Useful Indicators ---
    assign_if_enough_data('ATR', 15, lambda: compute_atr(df_copy, window=14), ['high', 'low', 'close'])
    assign_if_enough_data('RSI', 15, lambda: compute_rsi(df_copy['close'], window=14), ['close'])
    assign_if_enough_data('SMA_50', 50, lambda: df_copy['close'].rolling(window=50, min_periods=50).mean(), ['close'])
    assign_if_enough_data('SMA_200', 200, lambda: df_copy['close'].rolling(window=200, min_periods=200).mean(), ['close'])
    assign_if_enough_data('ADX', 28, lambda: compute_adx(df_copy, window=14), ['high', 'low', 'close'])

    # Bollinger Bands Example
    min_len_bollinger = 21
    if df_len >= min_len_bollinger and 'close' in df_copy and df_copy['close'].notna().sum() >= min_len_bollinger:
        try:
            upper, middle, lower = compute_bollinger_bands(df_copy['close'], window=20)
            df_copy['Bollinger_Upper']=pd.to_numeric(upper, errors='coerce')
            df_copy['Bollinger_Middle']=pd.to_numeric(middle, errors='coerce')
            df_copy['Bollinger_Lower']=pd.to_numeric(lower, errors='coerce')
        except Exception as e: 
            logger.error(f"[{symbol_log}] Error calculating Bollinger Bands: {e}", exc_info=False)
            df_copy['Bollinger_Upper']=np.nan; df_copy['Bollinger_Middle']=np.nan; df_copy['Bollinger_Lower']=np.nan
    else: 
        df_copy['Bollinger_Upper']=np.nan; df_copy['Bollinger_Middle']=np.nan; df_copy['Bollinger_Lower']=np.nan
        logger.debug(f"[{symbol_log}] Skipping Bollinger Bands (req {min_len_bollinger} valid close points).")

    logger.debug(f"[{symbol_log}] Finished applying indicators.")
    return df_copy

# --- Statistical Models (GARCH, VaR - Inchangé) ---
# ... (Gardez les fonctions fit_garch_model et calculate_var précédentes) ...

# --- GPT Integration (Breakout Prompt) ---
# Helper function pour le contexte GPT
def _recheck_breakout_conditions(latest_row: pd.Series, breakout_period: int, volume_confirmation_multiplier: float, rsi_momentum_threshold: float) -> Dict[str, Any]:
    # ... (Gardez la fonction _recheck_breakout_conditions précédente) ...
    conditions = {'breakout_long': False,'breakout_short': False,'volume_ok': False,'rsi_ok_long': False,'rsi_ok_short': False,'error': None}
    try:
        roll_high_col = f'Rolling_High_{breakout_period}'; roll_low_col = f'Rolling_Low_{breakout_period}'; avg_vol_col = f'Avg_Volume_{breakout_period}'
        current_high=latest_row.get('high'); current_low=latest_row.get('low'); current_volume=latest_row.get('volume')
        rolling_high=latest_row.get(roll_high_col); rolling_low=latest_row.get(roll_low_col); avg_volume=latest_row.get(avg_vol_col); current_rsi=latest_row.get('RSI')
        required=[current_high, current_low, current_volume, rolling_high, rolling_low, avg_volume, current_rsi]
        if not all(v is not None and pd.notna(v) and np.isfinite(v) for v in required): conditions['error'] = "Missing indicator data."; return conditions
        if avg_volume <= 1e-9: conditions['error'] = "Average volume near zero."; return conditions
        conditions['breakout_long'] = current_high > rolling_high; conditions['breakout_short'] = current_low < rolling_low
        conditions['volume_ok'] = current_volume > (avg_volume * volume_confirmation_multiplier)
        rsi_short_threshold = 100.0 - rsi_momentum_threshold
        conditions['rsi_ok_long'] = current_rsi > rsi_momentum_threshold; conditions['rsi_ok_short'] = current_rsi < rsi_short_threshold
    except Exception as e: conditions['error'] = f"Error rechecking: {e}"
    return conditions

def gpt_generate_trading_parameters(
    df_with_indicators: pd.DataFrame, symbol: str, timeframe: str, account_balance: float, max_leverage: float,
    garch_volatility: Optional[float], var95: Optional[float], technically_derived_direction: str, 
    min_requested_rr: Optional[float], breakout_period: int = 20,  
    volume_confirmation_multiplier: float = 1.5, rsi_momentum_threshold: float = 55.0 
) -> str:
    # ... (Gardez la fonction gpt_generate_trading_parameters COMPLÈTE de la réponse précédente, avec le prompt Breakout) ...
    log_prefix = f"[{symbol} ({timeframe}) GPT Breakout]"
    global openai_client 
    if openai_client is None: return json.dumps({"error": "OpenAI client not available"})
    if df_with_indicators.empty or len(df_with_indicators) < 2: return json.dumps({"error": "Insufficient indicator data rows for GPT."})
    latest_row_series = df_with_indicators.iloc[-1]; latest_data = latest_row_series.to_dict() 
    roll_high_col=f'Rolling_High_{breakout_period}'; roll_low_col=f'Rolling_Low_{breakout_period}'; avg_vol_col=f'Avg_Volume_{breakout_period}'
    key_inds_for_prompt = ['ATR', 'RSI', 'volume', roll_high_col, roll_low_col, avg_vol_col]
    optional_inds = ['SMA_50', 'SMA_200', 'ADX', 'Bollinger_Upper', 'Bollinger_Lower']
    key_inds_for_prompt.extend(optional_inds)
    # --- DANS la fonction gpt_generate_trading_parameters ---

    # ... (code avant la boucle) ...
    
    technical_indicators = {}
    missing_keys_critical = [] # Renommée de missing_keys pour clarté
    
    for key in key_inds_for_prompt:
        # Indentation niveau 1 (sous le for)
        value = latest_data.get(key)
        # Déterminer si la clé est critique AVANT de vérifier sa valeur
        is_critical = key in ['ATR', 'RSI', 'volume', roll_high_col, roll_low_col, avg_vol_col] 
        
        if pd.notna(value) and np.isfinite(value):
            # Indentation niveau 2 (sous le if)
             # --- Logique de formatage ---
             if abs(value) >= 1e4 or (abs(value) < 1e-4 and value != 0): 
                 # Indentation niveau 3
                 tech_val = f"{value:.3e}"
             elif key in ['volume', avg_vol_col]: 
                 # Indentation niveau 3
                 tech_val = f"{value:.0f}" 
             else: 
                 # Indentation niveau 3
                 tech_val = f"{float(value):.4f}"
             # --- Fin logique formatage ---
             technical_indicators[key] = tech_val # Indentation niveau 2
        else:
            # Indentation niveau 2 (sous le else, aligné avec le if)
             technical_indicators[key] = "N/A" 
             # --- CORRECTION ICI ---
             # Cet 'if' doit être indenté au même niveau que la ligne précédente (sous le else)
             if is_critical: 
                 # Cette ligne doit être indentée davantage (sous le 'if is_critical')
                 missing_keys_critical.append(key)
             # --- FIN CORRECTION ---

    # Ce 'if' est au même niveau que le 'for' (niveau 0 d'indentation par rapport au for)
    if missing_keys_critical:
        logger.error(f"{log_prefix} Cannot generate prompt, critical indicators missing: {missing_keys_critical}")
        return json.dumps({"error": f"Core indicator data missing for GPT: {missing_keys_critical}"})

    # ... (reste de la fonction gpt_generate_trading_parameters) ...
    current_price = latest_data.get('close')
    if pd.isna(current_price) or not np.isfinite(current_price): return json.dumps({"error": "Missing current price for GPT context"})
    current_price = round(float(current_price), 4)
    garch_vol_str = f"{garch_volatility:.4%}" if garch_volatility is not None and np.isfinite(garch_volatility) else "N/A"
    var95_str = f"{var95:.4%}" if var95 is not None and np.isfinite(var95) else "N/A"
    signal_context = "Signal Context Error"; conditions = _recheck_breakout_conditions(latest_row_series, breakout_period, volume_confirmation_multiplier, rsi_momentum_threshold)
    if conditions['error']: signal_context = f"Context Error: {conditions['error']}"
    elif technically_derived_direction == 'long': signal_context = (f"Potential LONG: Breakout above {breakout_period}-bar high ({technical_indicators.get(roll_high_col, 'N/A')}). Bar High: {latest_data.get('high'):.4f}. Vol OK: {conditions['volume_ok']}. RSI OK (> {rsi_momentum_threshold:.1f}): {conditions['rsi_ok_long']}.")
    elif technically_derived_direction == 'short': rsi_short_threshold = 100.0 - rsi_momentum_threshold; signal_context = (f"Potential SHORT: Breakdown below {breakout_period}-bar low ({technical_indicators.get(roll_low_col, 'N/A')}). Bar Low: {latest_data.get('low'):.4f}. Vol OK: {conditions['volume_ok']}. RSI OK (< {rsi_short_threshold:.1f}): {conditions['rsi_ok_short']}.")
    else: signal_context = "No valid signal direction provided."
    market_info = {"symbol": symbol, "timeframe": timeframe, "current_price": current_price,"garch_forecast_volatility": garch_vol_str, "value_at_risk_95": var95_str,"derived_signal_direction": technically_derived_direction,"signal_trigger_context": signal_context,"key_technical_indicators": technical_indicators,"account_balance_usd": account_balance, "max_allowable_leverage": max_leverage,"minimum_target_rr": min_requested_rr or 1.5 }
    data_json = json.dumps(market_info, indent=2, default=str) 
    prompt = f"""You are a cryptocurrency trading analyst evaluating a potential trade setup based on a price Breakout strategy ({breakout_period}-period High/Low).
The system detected a potential signal direction: '{technically_derived_direction}'. Your task is to EVALUATE this signal using the provided market data, indicators, and signal context, then provide actionable parameters if appropriate.
Market Data & Signal Context:\n{data_json}\nInstructions:
1. Evaluate Signal: Assess the '{technically_derived_direction}' signal based on the `signal_trigger_context`. Consider: Quality of Breakout, Confirmation (Volume, RSI), Contradictions (nearby S/R, divergences, low ADX if avail.).
2. Determine Action: Confirm `trade_direction` if breakout is valid and lacks strong contradictions. Output `trade_direction: 'hold'` if weak or faces immediate obstacles.
3. Refine Parameters (if action is 'long' or 'short'): `optimal_entry` (near breakout level/retest), `stop_loss` (invalidate breakout, use ATR guide: {technical_indicators.get('ATR', 'N/A')}), `take_profit` (based on entry/SL/RR: {min_requested_rr or 1.5}), `leverage` (1-10x respecting max).
4. Provide Confidence: `confidence_score` (0.0-1.0) based on quality, confirmation, lack of contradictions.
5. Justify: Explain concisely in `analysis` sections.
Respond ONLY with a single, valid JSON object.
"""
    try:
        logger.info(f"{log_prefix} Sending request to GPT...")
        response = openai_client.chat.completions.create(model="gpt-4.1", messages=[{"role": "system", "content": "You are a crypto trading analyst. Respond ONLY in JSON."},{"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0.3, max_tokens=1500 )
        gpt_output = response.choices[0].message.content
        return gpt_output or "{}"
    except openai.RateLimitError as e: logger.error(f"{log_prefix} OpenAI Rate Limit Error: {e}"); return json.dumps({"error": "OpenAI API rate limit exceeded", "details": str(e)})
    except openai.APIError as e: logger.error(f"{log_prefix} OpenAI API Error: {e}", exc_info=False); return json.dumps({"error": "OpenAI API error", "details": str(e)})
    except Exception as e: logger.error(f"{log_prefix} Error querying OpenAI: {e}", exc_info=False); return json.dumps({"error": "Failed to query OpenAI", "details": str(e)})


# --- Parse GPT Parameters ---
def parse_gpt_trading_parameters(gpt_output_str: str, symbol_for_log: str = "") -> Dict[str, Any]:
    # ... (Code inchangé - version précédente) ...
    log_prefix = f"[{symbol_for_log} Parse]"; parsed_data = {'optimal_entry': None,'stop_loss': None,'take_profit': None,'trade_direction': 'hold','leverage': None,'position_size_usd': None,'estimated_profit': None,'confidence_score': 0.0,'analysis': {'signal_evaluation': None,'technical_analysis': None,'risk_assessment': None,'market_outlook': None,'raw_text': gpt_output_str}}
    try:
        match = re.search(r'```json\s*(\{.*?\})\s*```', gpt_output_str, re.DOTALL | re.IGNORECASE)
        json_str = match.group(1) if match else gpt_output_str
        data = json.loads(json_str)
        if not isinstance(data, dict): raise json.JSONDecodeError("Not a JSON object", json_str, 0)
        def get_numeric(key, target_type=float):
            val = data.get(key); 
            try: num_val = target_type(val); return num_val if np.isfinite(num_val) else None
            except (ValueError, TypeError): return None
        parsed_data['optimal_entry']=get_numeric('optimal_entry', float); parsed_data['stop_loss']=get_numeric('stop_loss', float); parsed_data['take_profit']=get_numeric('take_profit', float)
        parsed_data['position_size_usd']=get_numeric('position_size_usd', float); parsed_data['estimated_profit']=get_numeric('estimated_profit', float)
        parsed_data['confidence_score']=get_numeric('confidence_score', float); parsed_data['leverage']=get_numeric('leverage', int)
        if parsed_data['leverage'] is not None and parsed_data['leverage'] < 1: parsed_data['leverage'] = None
        direction = data.get('trade_direction')
        if isinstance(direction, str) and direction.lower() in ['long', 'short', 'hold']: parsed_data['trade_direction'] = direction.lower()
        else: parsed_data['trade_direction'] = 'hold'; logger.warning(f"{log_prefix} Invalid trade_direction '{direction}'. Defaulting hold.")
        if parsed_data['trade_direction'] in ['long', 'short']:
             required = ['optimal_entry', 'stop_loss', 'take_profit', 'confidence_score']; missing = [p for p in required if parsed_data[p] is None]
             if missing: parsed_data['trade_direction'] = 'hold'; logger.warning(f"{log_prefix} GPT suggested {parsed_data['trade_direction']} but missing {missing}. Forcing hold.")
             elif parsed_data['optimal_entry'] and parsed_data['stop_loss'] and parsed_data['take_profit']:
                 entry=parsed_data['optimal_entry']; sl=parsed_data['stop_loss']; tp=parsed_data['take_profit']
                 if (parsed_data['trade_direction'] == 'long' and not (sl < entry < tp)) or \
                    (parsed_data['trade_direction'] == 'short' and not (tp < entry < sl)): parsed_data['trade_direction'] = 'hold'; logger.warning(f"{log_prefix} GPT levels illogical. Forcing hold.")
                 elif abs(entry - sl) < 1e-9: parsed_data['trade_direction'] = 'hold'; logger.warning(f"{log_prefix} GPT zero risk. Forcing hold.")
        analysis_dict = data.get('analysis')
        if isinstance(analysis_dict, dict):
            for key in ['signal_evaluation', 'technical_analysis', 'risk_assessment', 'market_outlook']:
                val = analysis_dict.get(key); 
                if isinstance(val, str) and val.strip(): parsed_data['analysis'][key] = val.strip()
    except json.JSONDecodeError as e: logger.error(f"{log_prefix} Failed JSON decode: {e}. Raw: {gpt_output_str[:200]}..."); parsed_data['trade_direction'] = 'hold'; parsed_data['analysis']['signal_evaluation'] = f"Error: Parse Fail {e}"
    except Exception as e: logger.error(f"{log_prefix} Unexpected parse error: {e}", exc_info=True); parsed_data['trade_direction'] = 'hold'; parsed_data['analysis']['signal_evaluation'] = f"Error: Unexpected Parse {e}"
    if parsed_data['trade_direction'] == 'hold':
        parsed_data['optimal_entry']=None; parsed_data['stop_loss']=None; parsed_data['take_profit']=None; parsed_data['leverage']=None; parsed_data['position_size_usd']=None; parsed_data['estimated_profit']=None
    return parsed_data

# --- Backtesting (Breakout Version) ---
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Assume other necessary imports and definitions exist (df_with_indicators, etc.)

# --- Backtesting (BREAKOUT STRATEGY) ---
def backtest_breakout_strategy( 
    df_with_indicators: pd.DataFrame,
    initial_signal_direction: str, # 'long' or 'short'
    # Strategy Parameters (Must match perform_single_analysis)
    breakout_period: int = 20,
    volume_confirmation_multiplier: float = 1.5,
    rsi_momentum_threshold: float = 55.0,
    # Simulation Parameters
    min_rr_ratio_target: float = 1.5,
    atr_sl_multiplier: float = 1.5,
    max_trade_duration_bars: int = 96, # Example: 4 days on 1h, 1 day on 15m
    min_bars_between_trades: int = 5 
) -> Dict[str, Any]:
    """
    Backtests a strategy based on historical occurrences of breakouts 
    of N-period high/low, confirmed by volume and RSI momentum.
    """
    
    symbol_log = "UnknownSymbol"
    if 'symbol' in df_with_indicators.columns:
        first_valid_symbol = df_with_indicators['symbol'].dropna()
        if not first_valid_symbol.empty:
            symbol_log = str(first_valid_symbol.iloc[0])
            
    log_prefix = f"[{symbol_log} Backtest Breakout N={breakout_period}]" 
    
    results = {
        'strategy_score': 0.0, 
        'trade_analysis': {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 
            'win_rate': None, 'avg_profit': None, 'avg_loss': None, 
            'profit_factor': None, 'total_profit': None, 'largest_win': None, 
            'largest_loss': None, 'average_trade_duration': None
        }, 
        'recommendation': 'N/A', 
        'warnings': []
    }
    trade_analysis = results['trade_analysis']

    if initial_signal_direction not in ['long', 'short']: 
        results['recommendation'] = f"Backtest skipped: Invalid initial signal direction '{initial_signal_direction}'."
        logger.info(f"{log_prefix} {results['recommendation']}")
        return results

    # --- Requirements ---
    roll_high_col = f'Rolling_High_{breakout_period}'
    roll_low_col = f'Rolling_Low_{breakout_period}'
    avg_vol_col = f'Avg_Volume_{breakout_period}'
    
    required_cols = [
        'open', 'high', 'low', 'close', 'volume', 'ATR', 'RSI', 
        roll_high_col, roll_low_col, avg_vol_col
    ]
    
    missing_cols = [col for col in required_cols if col not in df_with_indicators.columns]
    if missing_cols:
        results['recommendation'] = f"Backtest skipped: Missing required columns: {missing_cols}."
        logger.warning(f"{log_prefix} {results['recommendation']}")
        results['warnings'].append(f"Missing columns: {missing_cols}")
        return results
        
    # --- Data Cleaning ---
    df_clean = df_with_indicators[required_cols].dropna().copy() 

    min_data_points_needed = breakout_period + 10 
    if len(df_clean) < min_data_points_needed: 
        results['recommendation'] = f"Backtest skipped: Insufficient clean data ({len(df_clean)} < {min_data_points_needed})."
        results['warnings'].append(f"Insufficient clean data ({len(df_clean)})")
        logger.warning(f"{log_prefix} {results['recommendation']}")
        return results

    logger.info(f"{log_prefix} Starting backtest for '{initial_signal_direction}' signals over {len(df_clean)} potential bars.")

    # --- Simulation Setup ---
    trades = [] 
    last_entry_iloc = -1 - min_bars_between_trades 
    rsi_short_threshold = 100.0 - rsi_momentum_threshold 
    entry_count = 0

    # --- Simulation Loop ---
    for i_loc in range(len(df_clean) - 1): 
        signal_row = df_clean.iloc[i_loc]
        entry_bar_iloc = i_loc + 1
        entry_bar_timestamp = df_clean.index[entry_bar_iloc] 
        
        # --- Get Entry Price ---
        try: 
            entry_price = df_with_indicators.loc[entry_bar_timestamp, 'open']
            if pd.isna(entry_price) or not np.isfinite(entry_price): continue 
        except KeyError: continue 
            
        # --- Cooldown Check ---
        if i_loc <= last_entry_iloc + min_bars_between_trades:
             continue

        # --- Check Signal Conditions ---
        current_high = signal_row['high']; current_low = signal_row['low']; current_volume = signal_row['volume']
        rolling_high = signal_row[roll_high_col]; rolling_low = signal_row[roll_low_col]
        avg_volume = signal_row[avg_vol_col]; current_rsi = signal_row['RSI']; atr = signal_row['ATR'] 

        if atr <= 1e-9 or avg_volume <= 1e-9: continue 

        setup_found = False
        stop_loss_calc, take_profit_calc = 0.0, 0.0

        # Apply Breakout signal logic
        breakout_long_condition = current_high > rolling_high
        breakout_short_condition = current_low < rolling_low
        volume_confirmed = current_volume > (avg_volume * volume_confirmation_multiplier)
        rsi_confirms_long = current_rsi > rsi_momentum_threshold
        rsi_confirms_short = current_rsi < rsi_short_threshold

        signal_matches_request = False
        if initial_signal_direction == 'long' and breakout_long_condition and volume_confirmed and rsi_confirms_long:
            signal_matches_request = True
        elif initial_signal_direction == 'short' and breakout_short_condition and volume_confirmed and rsi_confirms_short:
            signal_matches_request = True

        if signal_matches_request:
            # Calculate SL/TP 
            risk_per_point = atr * atr_sl_multiplier
            if initial_signal_direction == 'long':
                stop_loss_calc = entry_price - risk_per_point
                take_profit_calc = entry_price + risk_per_point * min_rr_ratio_target
            else: # Short
                stop_loss_calc = entry_price + risk_per_point
                take_profit_calc = entry_price - risk_per_point * min_rr_ratio_target

            # Validate levels
            if (initial_signal_direction == 'long' and np.isfinite(stop_loss_calc) and np.isfinite(take_profit_calc) and stop_loss_calc < entry_price < take_profit_calc) or \
               (initial_signal_direction == 'short' and np.isfinite(stop_loss_calc) and np.isfinite(take_profit_calc) and take_profit_calc < entry_price < stop_loss_calc):
                setup_found = True
                logger.debug(f"{log_prefix} {initial_signal_direction.capitalize()} Breakout setup found at {signal_row.name}. Entry@{entry_bar_timestamp}: {entry_price:.4f}, SL: {stop_loss_calc:.4f}, TP: {take_profit_calc:.4f}")
            else:
                 logger.debug(f"{log_prefix} Setup found at {signal_row.name} but SL/TP levels invalid/illogical.")

        # --- Simulate Trade ---
        if setup_found:
            entry_count += 1
            outcome, exit_price, exit_iloc = None, None, -1
            entry_iloc = entry_bar_iloc 
            max_exit_iloc = min(len(df_clean) - 1, entry_iloc + max_trade_duration_bars) 

            for k_iloc in range(entry_iloc, max_exit_iloc + 1):
                current_bar_timestamp = df_clean.index[k_iloc]
                try:
                    current_bar_orig = df_with_indicators.loc[current_bar_timestamp]
                    current_low = current_bar_orig.get('low'); current_high = current_bar_orig.get('high')
                    if pd.isna(current_low) or pd.isna(current_high) or not np.isfinite(current_low) or not np.isfinite(current_high): continue 
                except KeyError: continue 

                # Check SL/TP
                if initial_signal_direction == 'long':
                    if current_low <= stop_loss_calc: outcome, exit_price, exit_iloc = 'loss', stop_loss_calc, k_iloc; break
                    elif current_high >= take_profit_calc: outcome, exit_price, exit_iloc = 'win', take_profit_calc, k_iloc; break
                elif initial_signal_direction == 'short':
                    if current_high >= stop_loss_calc: outcome, exit_price, exit_iloc = 'loss', stop_loss_calc, k_iloc; break
                    elif current_low <= take_profit_calc: outcome, exit_price, exit_iloc = 'win', take_profit_calc, k_iloc; break
            
            # Handle duration exit
            if outcome is None:
                exit_iloc = max_exit_iloc; exit_bar_timestamp = df_clean.index[exit_iloc]
                try:
                    exit_price = df_with_indicators.loc[exit_bar_timestamp, 'close']
                    if pd.isna(exit_price) or not np.isfinite(exit_price): outcome, exit_price = 'error', entry_price 
                    else: outcome = 'win' if (initial_signal_direction == 'long' and exit_price > entry_price) or (initial_signal_direction == 'short' and exit_price < entry_price) else 'loss'
                except KeyError: outcome, exit_price = 'error', entry_price
                logger.debug(f"{log_prefix} Trade {entry_count} exited by duration at {exit_bar_timestamp}. Outcome: {outcome}")

            # Record trade
            if outcome and outcome != 'error' and exit_price is not None and np.isfinite(exit_price) and exit_iloc != -1:
                profit_points = (exit_price - entry_price) if initial_signal_direction == 'long' else (entry_price - exit_price)
                trade_duration = exit_iloc - entry_iloc 
                trades.append({'profit': profit_points, 'duration': trade_duration, 'outcome': outcome})
                last_entry_iloc = entry_iloc 

    # --- Analyze Backtest Results --- 
    trade_analysis['total_trades'] = len(trades) 
    logger.info(f"{log_prefix} BT finished. Found {len(trades)} trades.")
    if trades:
        profits=np.array([t['profit'] for t in trades]); durations=np.array([t['duration'] for t in trades]); outcomes=[t['outcome'] for t in trades] 
        trade_analysis['winning_trades'] = sum(1 for o in outcomes if o == 'win'); trade_analysis['losing_trades'] = len(trades) - trade_analysis['winning_trades']
        if len(trades) > 0: trade_analysis['win_rate'] = round(trade_analysis['winning_trades'] / len(trades), 3)
        winning_profits = profits[profits > 0]; losing_profits = profits[profits <= 0]; gross_profit = np.sum(winning_profits); gross_loss = abs(np.sum(losing_profits))
        trade_analysis['avg_profit'] = round(np.mean(winning_profits), 4) if len(winning_profits) > 0 else 0.0; trade_analysis['avg_loss'] = round(abs(np.mean(losing_profits)), 4) if len(losing_profits) > 0 else 0.0
        if gross_loss > 1e-9: trade_analysis['profit_factor'] = round(gross_profit / gross_loss, 2)
        elif gross_profit > 1e-9: trade_analysis['profit_factor'] = float('inf') 
        else: trade_analysis['profit_factor'] = 0.0 
        trade_analysis['total_profit'] = round(np.sum(profits), 4); trade_analysis['largest_win'] = round(np.max(winning_profits), 4) if len(winning_profits) > 0 else 0.0
        trade_analysis['largest_loss'] = round(np.min(losing_profits), 4) if len(losing_profits) > 0 else 0.0; trade_analysis['average_trade_duration'] = round(np.mean(durations), 1) if len(durations) > 0 else 0.0
        
        # --- Calculate Strategy Score --- 
        score = 0.0; pf = trade_analysis['profit_factor']; wr = trade_analysis['win_rate']; num_trades = len(trades); avg_w = trade_analysis['avg_profit']; avg_l = trade_analysis['avg_loss']
        
        # PF Score
        if pf is not None:
            if pf == float('inf') or pf >= 2.5: 
                score += 0.35 
            elif pf >= 1.7: 
                score += 0.25
            elif pf >= 1.4: 
                score += 0.15 
            elif pf >= 1.0: 
                score += 0.05
            else: # pf < 1.0
                score -= 0.30 
                
        # WR Score        
        if wr is not None:
            if wr >= 0.6: 
                score += 0.20
            elif wr >= 0.5: 
                score += 0.15
            elif wr >= 0.4: 
                score += 0.10
            else: # wr < 0.4
                score -= 0.15 
                
        # Num Trades Score - CORRECTION ICI
        if num_trades >= 30: 
            score += 0.15 
        elif num_trades >= 15: # Doit être elif
            score += 0.10
        elif num_trades < 10: # Doit être elif (ou else si c'est la dernière condition)
            score -= 0.15 
        # --- FIN CORRECTION ---
            
        # Avg Win/Loss Ratio Score
        if avg_w is not None and avg_l is not None and avg_l > 1e-9: 
             ratio = avg_w / avg_l 
             if ratio >= 2.0: 
                 score += 0.10
             elif ratio >= 1.5: 
                 score += 0.05
                 
        # Net Profit Penalty        
        if trade_analysis['total_profit'] is not None and trade_analysis['total_profit'] <= 0: 
             score -= 0.30 

        results['strategy_score'] = max(0.0, min(1.0, round(score, 2)))
        logger.info(f"{log_prefix} Backtest Score: {results['strategy_score']:.2f} (Trades:{num_trades}, WR:{wr*100 if wr else 'N/A':.1f}%, PF:{pf if pf else 'N/A':.2f})")

        # Recommendations / Warnings
        strategy_name = "Breakout"
        if results['strategy_score'] >= 0.65: results['recommendation'] = f"Good historical performance for {strategy_name}." 
        elif results['strategy_score'] >= 0.50: results['recommendation'] = f"Moderate historical performance for {strategy_name}."
        else: results['recommendation'] = f"Poor historical performance for {strategy_name}."
        if wr is not None and wr < 0.40: results['warnings'].append(f"Low Win Rate ({wr:.1%})") 
        if pf is not None and pf < 1.4: results['warnings'].append(f"Low Profit Factor ({pf:.2f})") 
        if num_trades < 10: results['warnings'].append(f"Low Trade Count ({num_trades}).")
        if trade_analysis['total_profit'] is not None and trade_analysis['total_profit'] <= 0: results['warnings'].append("Overall loss in backtest.")
        if avg_l is not None and avg_l > 1e-9 and trade_analysis['largest_loss'] is not None and abs(trade_analysis['largest_loss']) > 3 * avg_l: results['warnings'].append("Largest loss >> avg loss.")
            
    else: # No trades
        strategy_name = "Breakout"; results['recommendation'] = f"No historical {strategy_name} setups found."; results['warnings'].append("No qualifying historical trades."); results['strategy_score'] = 0.0; logger.info(f"{log_prefix} {results['recommendation']}")
        
    return results


# --- Core Analysis Function (Breakout Version) ---
async def perform_single_analysis(
    symbol: str, timeframe: str, lookback: int, account_balance: float, max_leverage: float,
    min_requested_rr: Optional[float] = 1.5, breakout_period: int = 20,  
    volume_confirmation_multiplier: float = 1.5, rsi_momentum_threshold: float = 55.0
) -> AnalysisResponse:
    # --- Helper function ---
    def is_valid_numeric(value): return value is not None and isinstance(value, (int, float, np.number)) and np.isfinite(value)
    def format_or_na(value, fmt): return f"{value:{fmt}}" if is_valid_numeric(value) else "N/A"

    start_time_ticker = time.time()
    log_prefix = f"[{symbol} ({timeframe})]"
    logger.info(f"{log_prefix} Starting analysis (Breakout Strategy N={breakout_period})...")
    analysis_result = AnalysisResponse(symbol=symbol, timeframe=timeframe)
    df_indicators = None 
    indicator_payload = {} # Définir ici pour être sûr qu'il existe
    roll_high_col = f'Rolling_High_{breakout_period}' # Définir les noms ici
    roll_low_col = f'Rolling_Low_{breakout_period}'
    avg_vol_col = f'Avg_Volume_{breakout_period}'

    # --- Step 1: Fetch Data ---
    try:
        min_req = breakout_period + 50 
        if lookback < min_req: logger.warning(f"{log_prefix} Adjusting lookback {lookback} to {min_req}."); lookback = min_req
        df_raw = await get_real_time_data(symbol, timeframe, lookback) 
        if df_raw.empty or len(df_raw) < min_req: analysis_result.error = f"Insufficient data ({len(df_raw)}/{min_req})"; return analysis_result
        current_price_val = df_raw['close'].iloc[-1]
        if pd.isna(current_price_val) or not np.isfinite(current_price_val): analysis_result.error = "Invalid current price"; return analysis_result
        analysis_result.currentPrice = float(current_price_val)
        logger.info(f"{log_prefix} Data fetched ({len(df_raw)} bars). Price: {analysis_result.currentPrice:.4f}")
    except (ConnectionError, ConnectionAbortedError, ValueError, PermissionError, RuntimeError) as e: analysis_result.error = f"Data Fetch Error: {e}"; return analysis_result
    except Exception as e: analysis_result.error = f"Unexpected Data Fetch Error: {e}"; logger.error(f"{log_prefix} Unexpected Fetch Error", exc_info=True); return analysis_result

    # --- Step 2: Apply Indicators & Create Payload --- 
    try:
        if 'symbol' not in df_raw.columns: df_raw['symbol'] = symbol 
        df_indicators = await asyncio.to_thread(apply_technical_indicators, df_raw, breakout_period)
        
        if df_indicators is None or df_indicators.empty or len(df_indicators) < 2: raise ValueError("Indicator calc resulted in insufficient data.")

        latest_indicators_raw = df_indicators.iloc[-1].to_dict()
        indicator_payload.clear() # Vider avant de remplir
        invalid_indicator_details = {}

        # Peupler le payload avec validation
        expected_fields_and_aliases = {f.alias or k: k for k, f in IndicatorsData.model_fields.items()}
        expected_fields_and_aliases.update({ # Ajouter les clés dynamiques
             roll_high_col: roll_high_col, roll_low_col: roll_low_col, avg_vol_col: avg_vol_col 
        })
        
        for key_in_df, field_name in expected_fields_and_aliases.items():
            value = latest_indicators_raw.get(key_in_df) 
            if value is not None and pd.notna(value):
                if is_valid_numeric(value):
                    indicator_payload[field_name] = float(value) 
                else:
                    invalid_indicator_details[key_in_df] = f"Invalid Type/Value ({type(value).__name__}: {value})"

        critical_keys_for_signal = ['high', 'low', 'volume', 'ATR', 'RSI', roll_high_col, roll_low_col, avg_vol_col]
        missing_critical = [k for k in critical_keys_for_signal if k not in indicator_payload]
        
        if invalid_indicator_details: logger.warning(f"{log_prefix} Invalid indicators found: {invalid_indicator_details}")
        if missing_critical: raise ValueError(f"Critical indicators missing/invalid after validation: {missing_critical}")
        
        analysis_result.indicators = IndicatorsData(**indicator_payload)
        logger.info(f"{log_prefix} Indicators applied and payload created.")
        logger.debug(
            f"{log_prefix} Latest Validated Indicators Payload - "
            f"H:{format_or_na(indicator_payload.get('high'), '.4f')} L:{format_or_na(indicator_payload.get('low'), '.4f')} "
            f"RollH:{format_or_na(indicator_payload.get(roll_high_col), '.4f')} RollL:{format_or_na(indicator_payload.get(roll_low_col), '.4f')} "
            f"Vol:{format_or_na(indicator_payload.get('volume'), '.0f')} AvgVol:{format_or_na(indicator_payload.get(avg_vol_col), '.0f')} " 
            f"RSI:{format_or_na(indicator_payload.get('RSI'), '.1f')}"
        )
            
    except Exception as e:
        logger.error(f"{log_prefix} Indicator application or payload creation error: {e}", exc_info=True)
        analysis_result.error = f"Indicator Error: {e}"
        return analysis_result 

    # --- Step 2.5: Determine Technical Trigger ---
    technical_signal_direction = "hold" 
    try:
        current_high = indicator_payload.get('high'); current_low = indicator_payload.get('low'); current_volume = indicator_payload.get('volume')
        rolling_high = indicator_payload.get(roll_high_col); rolling_low = indicator_payload.get(roll_low_col)
        avg_volume = indicator_payload.get(avg_vol_col); current_rsi = indicator_payload.get('RSI')

        # Vérifier si toutes les valeurs requises sont présentes dans le payload validé
        required_values_present = all(k in indicator_payload for k in critical_keys_for_signal)

        if not required_values_present:
             logger.warning(f"{log_prefix} Required values missing in validated payload for signal check.")
             analysis_result.error = (analysis_result.error or "") + "; Warning: Data missing in payload for signal"
        # avg_volume est déjà vérifié > 1e-9 implicitement car les NaN/Inf/0 sont filtrés avant
        else: 
             breakout_long_condition = current_high > rolling_high
             breakout_short_condition = current_low < rolling_low
             volume_confirmed = current_volume > (avg_volume * volume_confirmation_multiplier)
             rsi_short_threshold = 100.0 - rsi_momentum_threshold 
             rsi_confirms_long = current_rsi > rsi_momentum_threshold
             rsi_confirms_short = current_rsi < rsi_short_threshold

             if breakout_long_condition and volume_confirmed and rsi_confirms_long: technical_signal_direction = "long"
             elif breakout_short_condition and volume_confirmed and rsi_confirms_short: technical_signal_direction = "short"

             logger.info(
                 f"{log_prefix} Technical signal trigger check (Breakout): {technical_signal_direction.upper()} "
                 f"(BO_L={breakout_long_condition} "
                     f"[H:{format_or_na(current_high, '.4f')} > RollH:{format_or_na(rolling_high, '.4f')}], "
                 f"BO_S={breakout_short_condition} "
                     f"[L:{format_or_na(current_low, '.4f')} < RollL:{format_or_na(rolling_low, '.4f')}], "
                 f"Vol_OK={volume_confirmed} "
                     f"[V:{format_or_na(current_volume, '.0f')} > AvgV*Mult:{format_or_na(avg_volume * volume_confirmation_multiplier if avg_volume else None, '.0f')}], "
                 f"RSI_L_OK={rsi_confirms_long} "
                     f"[RSI:{format_or_na(current_rsi, '.1f')} > {rsi_momentum_threshold:.1f}], "
                 f"RSI_S_OK={rsi_confirms_short} "
                     f"[RSI:{format_or_na(current_rsi, '.1f')} < {rsi_short_threshold:.1f}])"
             )

    except Exception as e:
         logger.error(f"{log_prefix} Error during breakout signal check logic: {e}", exc_info=True)
         analysis_result.error = (analysis_result.error or "") + f"; Error in breakout signal logic: {e}"
         technical_signal_direction = "hold" 
         
    # --- Step 3: Statistical Models (GARCH, VaR) ---
    # ... (Code inchangé, mais s'assure qu'il gère le cas où df_indicators est None) ...
    garch_vol, var95 = None, None
    try:
        if df_indicators is not None and 'returns' in df_indicators and df_indicators['returns'].notna().sum() >= 50:
            returns = df_indicators['returns'].dropna() 
            if len(returns) >= 50: 
                 garch_task = asyncio.to_thread(fit_garch_model, returns, symbol)
                 var_task = asyncio.to_thread(calculate_var, returns, 0.95, symbol)
                 garch_vol, var95 = await asyncio.gather(garch_task, var_task)
                 analysis_result.modelOutput = ModelOutputData(garchVolatility=garch_vol, var95=var95)
                 logger.info(f"{log_prefix} GARCH/VaR calculated (Vol: {format_or_na(garch_vol, '.4f')}, VaR95: {format_or_na(var95, '.4f')}).")
            else: logger.warning(f"{log_prefix} Skipping GARCH/VaR (< 50 valid returns)."); analysis_result.modelOutput = ModelOutputData()
        else: logger.warning(f"{log_prefix} Skipping GARCH/VaR (no valid 'returns' data or < 50 points)."); analysis_result.modelOutput = ModelOutputData() 
    except Exception as e: logger.error(f"{log_prefix} Stat model error: {e}", exc_info=True); analysis_result.error = (analysis_result.error or "") + f"; Stat Model Error: {e}"; analysis_result.modelOutput = ModelOutputData()

    # --- Step 4: GPT Evaluation ---
    gpt_parsed_output = None
    if openai_client and technical_signal_direction != 'hold':
        logger.info(f"{log_prefix} Tech signal '{technical_signal_direction}', proceeding with GPT eval.")
        try:
            gpt_raw_output = await asyncio.to_thread(
                 gpt_generate_trading_parameters, df_indicators, symbol, timeframe, account_balance, max_leverage,
                 garch_vol, var95, technical_signal_direction, min_requested_rr,
                 breakout_period=breakout_period, volume_confirmation_multiplier=volume_confirmation_multiplier,
                 rsi_momentum_threshold=rsi_momentum_threshold 
            )
            gpt_parsed_output = parse_gpt_trading_parameters(gpt_raw_output, symbol)
            if gpt_parsed_output:
                 analysis_result.gptParams = GptTradingParams(**{k: v for k, v in gpt_parsed_output.items() if k != 'analysis'})
                 analysis_result.gptAnalysis = GptAnalysisText(**gpt_parsed_output.get('analysis', {}))
                 logger.info(f"{log_prefix} GPT eval complete. EvalDir: {analysis_result.gptParams.trade_direction}, Conf: {analysis_result.gptParams.confidence_score}")
                 gpt_internal_error_msg = gpt_parsed_output.get("error") or (gpt_parsed_output.get('analysis', {}).get('signal_evaluation') or "").startswith("Error:")
                 if gpt_internal_error_msg:
                     err_detail = gpt_parsed_output.get("details", gpt_internal_error_msg); logger.warning(f"{log_prefix} GPT issue: {err_detail}"); analysis_result.error = (analysis_result.error or "") + f"; GPT Warning: {str(err_detail)[:100]}"
            else:
                 logger.error(f"{log_prefix} GPT parsing failed."); analysis_result.error = (analysis_result.error or "") + "; GPT parsing failed"
                 analysis_result.gptAnalysis = GptAnalysisText(raw_text="GPT parsing failed."); analysis_result.gptParams = GptTradingParams(trade_direction='hold') 
        except Exception as e:
            logger.error(f"{log_prefix} GPT processing error: {e}", exc_info=True); analysis_result.error = (analysis_result.error or "") + f"; GPT Processing Error: {e}"
            gpt_parsed_output = None; analysis_result.gptAnalysis = GptAnalysisText(raw_text=f"Error GPT step: {e}"); analysis_result.gptParams = GptTradingParams(trade_direction='hold') # Assigner défaut
    else:
        reason = "OpenAI client missing" if not openai_client else "Technical signal is 'hold'"
        logger.info(f"{log_prefix} Skipping GPT evaluation: {reason}.")
        analysis_result.gptAnalysis = GptAnalysisText(signal_evaluation=f"GPT skipped: {reason}")
        analysis_result.gptParams = GptTradingParams(trade_direction='hold', confidence_score=None)
        gpt_parsed_output = {'trade_direction': 'hold'}

    # --- Step 5: Backtesting ---
    if technical_signal_direction in ['long', 'short'] and df_indicators is not None: # S'assurer que df_indicators existe
        logger.info(f"{log_prefix} Running backtest for initial signal (Breakout): {technical_signal_direction}")
        try:
            backtest_raw = await asyncio.to_thread(
                backtest_breakout_strategy, df_indicators.copy(), technical_signal_direction,
                breakout_period=breakout_period, volume_confirmation_multiplier=volume_confirmation_multiplier,
                rsi_momentum_threshold=rsi_momentum_threshold, min_rr_ratio_target=(min_requested_rr or 1.5) 
            )
            analysis_result.backtest = BacktestResultsData(
                strategy_score=backtest_raw.get('strategy_score'),
                trade_analysis=BacktestTradeAnalysis(**backtest_raw.get('trade_analysis', {})),
                recommendation=backtest_raw.get('recommendation'), warnings=backtest_raw.get('warnings', [])
            )
            logger.info(f"{log_prefix} Backtest done. Score: {analysis_result.backtest.strategy_score}, Rec: {analysis_result.backtest.recommendation}")
        except Exception as e:
            logger.error(f"{log_prefix} Backtesting error: {e}", exc_info=True); analysis_result.error = (analysis_result.error or "") + f"; Backtesting Error: {e}"
            analysis_result.backtest = BacktestResultsData(recommendation=f"Backtest failed: {e}", warnings=[f"Error: {e}"])
    else:
        reason_skip = "Initial signal is 'hold'" if technical_signal_direction == 'hold' else "Indicator data unavailable"
        logger.info(f"{log_prefix} Skipping backtest ({reason_skip}).")
        analysis_result.backtest = BacktestResultsData(recommendation=f"Backtest skipped: {reason_skip}")

    # --- Finalization ---
    duration = time.time() - start_time_ticker
    if not analysis_result.error: logger.info(f"{log_prefix} Analysis successful ({duration:.2f}s).")
    else:
        log_level = logging.WARNING if "Warning:" in analysis_result.error or "skipped:" in analysis_result.error or "Insufficient data" in analysis_result.error else logging.ERROR
        logger.log(log_level, f"{log_prefix} Analysis finished with issues ({duration:.2f}s). Status: {analysis_result.error}")
    return analysis_result



async def analyze_with_semaphore_wrapper(
    ticker: str,                   
    request: ScanRequest,          
    semaphore: asyncio.Semaphore,  
    progress_lock: asyncio.Lock,   
    processed_count_ref: List[int], 
    total_attempted: int,          
    log_interval: int              
    ) -> AnalysisResponse:         
    """
    Wrapper for perform_single_analysis with semaphore and progress tracking.
    Calls the Breakout version of perform_single_analysis.
    """
    result = None
    task_start_time = time.time()
    log_prefix_wrapper = f"[{ticker} ({request.timeframe}) Wrapper]" 

    try:
        async with semaphore:
            logger.debug(f"{log_prefix_wrapper} Acquired semaphore. Calling analysis...")
            result = await perform_single_analysis(
                symbol=ticker,
                timeframe=request.timeframe,
                lookback=request.lookback, 
                account_balance=request.accountBalance,
                max_leverage=request.maxLeverage,
                min_requested_rr=request.min_risk_reward_ratio, 
                # Params Breakout utilisent les défauts de perform_single_analysis
            )
    except Exception as e:
        logger.error(f"{log_prefix_wrapper} UNEXPECTED Exception during analysis call: {e}", exc_info=True)
        result = AnalysisResponse(
            symbol=ticker, timeframe=request.timeframe, 
            error=f"Scan Wrapper/Analysis Exception: {str(e)[:200]}" 
        )
    finally:
        task_duration = time.time() - task_start_time
        async with progress_lock:
            processed_count_ref[0] += 1 
            current_count = processed_count_ref[0]
        if current_count % log_interval == 0 or current_count == total_attempted:
             logger.info(f"API Scan Progress: {current_count}/{total_attempted} tasks completed (Last: {ticker} took {task_duration:.2f}s).")
        else: logger.debug(f"API Scan Progress: {current_count}/{total_attempted} (Task {ticker} took {task_duration:.2f}s).")
             
    return result if result is not None else AnalysisResponse(
        symbol=ticker, timeframe=request.timeframe, 
        error="Unknown wrapper state or analysis failed silently"
    )
# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    # ... (Inchangé) ...
    logger.info("FastAPI startup: Initializing...")
    asyncio.create_task(load_exchange_markets(binance_futures))
    if openai_client: asyncio.create_task(test_openai_connection(openai_client))

@app.get("/api/crypto/tickers", response_model=TickersResponse, tags=["Utility"])
async def get_crypto_tickers_endpoint():
    # ... (Inchangé) ...
    logger.info("API: Request received for /tickers")
    global binance_futures
    if binance_futures is None: raise HTTPException(status_code=503, detail="Exchange unavailable")
    if not binance_futures.markets:
        if not await load_exchange_markets(binance_futures): raise HTTPException(status_code=503, detail="Failed load markets.")
    try:
        markets = binance_futures.markets
        if not markets: raise HTTPException(status_code=500, detail="Markets loaded empty.")
        tickers = sorted([m['symbol'] for m in markets.values() if m.get('swap') and m.get('linear') and m.get('quote')=='USDT' and m.get('settle')=='USDT' and m.get('active')])
        logger.info(f"API Tickers: Found {len(tickers)} active USDT linear perpetuals.")
        return TickersResponse(tickers=tickers)
    except Exception as e: logger.error(f"API Tickers Error: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.post("/api/crypto/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_crypto_endpoint(request: AnalysisRequest):
    # ... (Inchangé, appelle perform_single_analysis qui utilise maintenant Breakout par défaut) ...
    logger.info(f"API: Analyze request received for {request.symbol} ({request.timeframe})")
    global binance_futures
    if binance_futures is None: raise HTTPException(status_code=503, detail="Exchange unavailable.")
    try:
        analysis_params = request.model_dump()
        # Passer les paramètres de la requête à la fonction d'analyse
        # Les paramètres spécifiques à la stratégie (breakout_period, etc.) utiliseront 
        # les défauts de perform_single_analysis si non passés explicitement ici.
        result = await perform_single_analysis(
             symbol=analysis_params['symbol'],
             timeframe=analysis_params['timeframe'],
             lookback=analysis_params['lookback'],
             account_balance=analysis_params['accountBalance'],
             max_leverage=analysis_params['maxLeverage'],
             min_requested_rr=1.5 # Ou lire depuis request si ajouté
             # breakout_period=20 # Ou lire depuis request si ajouté
             # ... etc
        )
        if result.error: # Gestion d'erreur standard
            err=result.error; status=500; log_func=logger.error
            if "Warning:" in err or "skipped:" in err or "Insufficient data" in err: status=200; log_func=logger.warning
            elif "Invalid symbol" in err: status=400
            elif "Fetch Error" in err or "Network" in err or "Connection" in err: status=504
            elif "Rate limit" in err or "ConnectionAbortedError" in err: status=429
            elif "Authentication Error" in err: status=401
            elif "OpenAI" in err or "GPT" in err: status=502
            elif "Exchange error" in err: status=503
            elif "Indicator Error" in err: status=422 # Erreur de calcul indicateur -> données probablement invalides
            log_func(f"API Analyze: Request for {request.symbol} status {status}. Detail: {err}")
            if status != 200: raise HTTPException(status_code=status, detail=err)
        logger.info(f"API Analyze: Request for {request.symbol} completed successfully.")
        return result
    except HTTPException as h: raise h
    except Exception as e: logger.error(f"API Analyze Endpoint Error for {request.symbol}: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Unexpected endpoint error: {e}")


@app.post("/api/crypto/scan", response_model=ScanResponse, tags=["Scanning"]) 
async def scan_market_endpoint(request: ScanRequest):
    # ... (Code complet de scan_market_endpoint de la réponse précédente, 
    #      qui utilise analyze_with_semaphore_wrapper, qui appelle maintenant 
    #      la version corrigée de perform_single_analysis) ...
    global binance_futures 
    if binance_futures is None: raise HTTPException(status_code=503, detail="Exchange unavailable")
    if not binance_futures.markets:
        if not await load_exchange_markets(binance_futures): raise HTTPException(status_code=503, detail="Failed load markets.")
    scan_start_time = time.time()
    logger.critical("--- SCAN ENDPOINT ENTERED (Strategy: Breakout) ---") 
    logger.info(f"API Scan: Received Request Object Timeframe: '{request.timeframe}'") 
    logger.info(f"API Scan: Starting request (Breakout): {request.model_dump_json(exclude_defaults=False)}")
    binance_1m_limit = 1500; effective_lookback = request.lookback
    if request.timeframe == '1m' and request.lookback > binance_1m_limit: logger.warning(f"API Scan: 1m Lookback {request.lookback} > {binance_1m_limit} limit.")
    elif request.lookback > 1500: logger.warning(f"API Scan: Lookback {request.lookback} > 1500 limit.")
    try: # Get Tickers
        markets = binance_futures.markets; 
        if not markets: raise ValueError("Markets dict empty.")
        all_tickers = sorted([m['symbol'] for m in markets.values() if m.get('swap') and m.get('linear') and m.get('quote')=='USDT' and m.get('settle')=='USDT' and m.get('active')])
        logger.info(f"API Scan: Found {len(all_tickers)} active USDT perpetuals.")
        if not all_tickers: return ScanResponse(scan_parameters=request, total_tickers_attempted=0, total_tickers_succeeded=0, ticker_start_index=request.ticker_start_index, ticker_end_index=request.ticker_end_index, total_opportunities_found=0, top_opportunities=[], errors={})
    except Exception as e: raise HTTPException(status_code=502, detail=f"Ticker retrieval error: {e}")
    # --- BTC Trend (Optional) ---
    btc_trend_state = "UNKNOWN"; apply_btc_filter = request.filter_by_btc_trend
    if apply_btc_filter:
        logger.info(f"API Scan: BTC Trend filter enabled. Fetching BTC data ({request.timeframe})...")
        try:
            btc_lookback = max(250, effective_lookback); df_btc_raw = await get_real_time_data("BTC/USDT:USDT", request.timeframe, btc_lookback)
            if df_btc_raw.empty or len(df_btc_raw) < 205: apply_btc_filter = False; logger.warning(f"API Scan: Insufficient BTC data ({len(df_btc_raw)}). Disabling BTC filter.")
            else:
                if 'symbol' not in df_btc_raw.columns: df_btc_raw['symbol'] = "BTC/USDT:USDT"
                df_btc_indicators = await asyncio.to_thread(apply_technical_indicators, df_btc_raw); btc_latest = df_btc_indicators.iloc[-1]
                btc_price=btc_latest.get('close'); btc_sma50=btc_latest.get('SMA_50'); btc_sma200=btc_latest.get('SMA_200')
                if all(v is not None and pd.notna(v) and np.isfinite(v) for v in [btc_price, btc_sma50, btc_sma200]):
                    if btc_price > btc_sma50 > btc_sma200: btc_trend_state = "UPTREND"
                    elif btc_price < btc_sma50 < btc_sma200: btc_trend_state = "DOWNTREND"
                    else: btc_trend_state = "CHOPPY"
                    logger.info(f"API Scan: BTC Trend ({request.timeframe}): {btc_trend_state}")
                else: apply_btc_filter = False; btc_trend_state = "UNKNOWN"; logger.warning("API Scan: Could not determine BTC trend. Disabling filter.")
        except Exception as e: apply_btc_filter = False; btc_trend_state = "ERROR"; logger.error(f"API Scan: Error getting BTC trend: {e}", exc_info=True)
    else: logger.info("API Scan: BTC Trend filter disabled.")
    # --- Select Tickers ---
    tickers_to_scan=[]; total_available=len(all_tickers); start_index = request.ticker_start_index or 0; end_index = request.ticker_end_index; actual_end_index_for_response=None; slice_desc=""
    if start_index < 0: start_index = 0
    if start_index >= total_available > 0: return ScanResponse(scan_parameters=request, total_tickers_attempted=0, total_tickers_succeeded=0, ticker_start_index=start_index, ticker_end_index=end_index, total_opportunities_found=0, top_opportunities=[], errors={})
    if end_index is not None: # Slice logic
        if end_index <= start_index: actual_end_index_for_response=end_index; slice_desc=f"invalid slice [{start_index}:{end_index}]"
        else: actual_end=min(end_index, total_available); tickers_to_scan=all_tickers[start_index:actual_end]; slice_desc=f"slice [{start_index}:{actual_end}]"; actual_end_index_for_response=end_index
    elif request.max_tickers is not None and request.max_tickers > 0: # Max tickers logic
        actual_end=min(start_index + request.max_tickers, total_available); tickers_to_scan=all_tickers[start_index:actual_end]; slice_desc=f"max_tickers={request.max_tickers} slice [{start_index}:{actual_end}]"; actual_end_index_for_response=actual_end
    elif request.max_tickers == 0: actual_end_index_for_response=start_index; slice_desc="max_tickers=0"
    else: actual_end=total_available; tickers_to_scan=all_tickers[start_index:]; slice_desc=f"all from {start_index} [{start_index}:{actual_end}]"; actual_end_index_for_response=actual_end
    logger.info(f"API Scan: Selected {len(tickers_to_scan)} tickers ({slice_desc}).")
    total_attempted = len(tickers_to_scan); 
    if total_attempted == 0: return ScanResponse(scan_parameters=request, total_tickers_attempted=0, total_tickers_succeeded=0, ticker_start_index=start_index, ticker_end_index=actual_end_index_for_response, total_opportunities_found=0, top_opportunities=[], errors={})
    # --- Run Concurrently ---
    semaphore=asyncio.Semaphore(request.max_concurrent_tasks); tasks=[]; processed_count_ref=[0]; progress_lock=asyncio.Lock(); log_interval=max(1,total_attempted//20 if total_attempted>=20 else 1)
    logger.info(f"API Scan: Creating {total_attempted} analysis tasks...")
    for ticker in tickers_to_scan: tasks.append(analyze_with_semaphore_wrapper(ticker, request, semaphore, progress_lock, processed_count_ref, total_attempted, log_interval))
    logger.info(f"API Scan: Gathering results (Concurrency: {request.max_concurrent_tasks})...")
    analysis_results_raw: List[Any] = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info(f"API Scan: Finished gathering {len(analysis_results_raw)} results.")
    # --- Process Results & Filters ---
    successful_analyses_count=0; analysis_errors={}; opportunities_passing_filter=[]
    logger.info("API Scan: Processing and filtering results...")
    # --- Helper for safe formatting in filter logs ---
    def format_or_na(value, fmt): return f"{value:{fmt}}" if value is not None and isinstance(value, (int, float, np.number)) and np.isfinite(value) else "N/A"
    for i, res_or_exc in enumerate(analysis_results_raw):
        symbol = tickers_to_scan[i] if i < len(tickers_to_scan) else f"Unknown_{i}"
        if isinstance(res_or_exc, Exception): logger.error(f"API Scan: Task [{symbol}] UNHANDLED Exception: {res_or_exc}", exc_info=True); analysis_errors[symbol] = f"Unhandled Task Exc: {str(res_or_exc)[:200]}"; continue
        elif isinstance(res_or_exc, AnalysisResponse):
            result: AnalysisResponse = res_or_exc
            is_skip = result.error and ("Insufficient data" in result.error or "skipped:" in result.error or "Warning:" in result.error); is_critical = result.error and not is_skip
            if is_critical: logger.error(f"API Scan: Critical error for [{result.symbol}]: {result.error}"); analysis_errors[result.symbol] = f"Analysis Error: {result.error}"; continue
            elif is_skip: logger.info(f"API Scan: Skipping filter for [{result.symbol}]: {result.error}"); successful_analyses_count += 1; continue
            else: successful_analyses_count += 1
            gpt_params=result.gptParams; bt_results=result.backtest; indicators=result.indicators; bt_analysis=bt_results.trade_analysis if bt_results and bt_results.trade_analysis else None
            eval_direction=gpt_params.trade_direction if gpt_params else None; gpt_conf=gpt_params.confidence_score if gpt_params and gpt_params.confidence_score is not None else None
            bt_score=bt_results.strategy_score if bt_results and bt_results.strategy_score is not None else None; current_price=result.currentPrice
            log_pre = f"[{result.symbol}] Pre-filter: EvalDir='{eval_direction or 'N/A'}', GPTConf={format_or_na(gpt_conf, '.2f')}, BTScore={format_or_na(bt_score, '.2f')}"
            if bt_analysis: log_pre+=f" (BT Trades={bt_analysis.total_trades}, WR={format_or_na(bt_analysis.win_rate, '.1%')}, PF={format_or_na(bt_analysis.profit_factor if bt_analysis.profit_factor != float('inf') else 999.9, '.2f')})" # Format PF
            logger.info(log_pre)
            passes_filters=True; filter_fail_reason=""
            # Filter logic (F1-F5) ...
            if eval_direction not in ['long', 'short']: passes_filters=False; filter_fail_reason=f"EvalDir not tradeable ('{eval_direction}')"
            elif request.trade_direction and eval_direction != request.trade_direction: passes_filters=False; filter_fail_reason=f"Direction mismatch (Req:{request.trade_direction})"
            elif gpt_conf is None or gpt_conf < request.min_gpt_confidence: passes_filters=False; filter_fail_reason=f"GPT Conf low (Got:{format_or_na(gpt_conf,'.2f')}<Req:{request.min_gpt_confidence:.2f})"
            elif bt_score is None or bt_score < request.min_backtest_score: passes_filters=False; filter_fail_reason=f"BT Score low (Got:{format_or_na(bt_score,'.2f')}<Req:{request.min_backtest_score:.2f})"
            if passes_filters: # F2 Backtest Stats
                if bt_analysis is None: passes_filters=False; filter_fail_reason="BT analysis missing"
                elif request.min_backtest_trades is not None and bt_analysis.total_trades < request.min_backtest_trades: logger.debug(f"[{symbol}] Filter Fail: Trades={bt_analysis.total_trades} < {request.min_backtest_trades}"); passes_filters=False; filter_fail_reason=f"BT Trades low ({bt_analysis.total_trades})"
                elif request.min_backtest_win_rate is not None and (bt_analysis.win_rate is None or bt_analysis.win_rate < request.min_backtest_win_rate): logger.debug(f"[{symbol}] Filter Fail: WR={format_or_na(bt_analysis.win_rate,'.1%')} < {request.min_backtest_win_rate:.1%}"); passes_filters=False; filter_fail_reason=f"BT WR low ({format_or_na(bt_analysis.win_rate,'.1%')})"
                elif request.min_backtest_profit_factor is not None and (bt_analysis.profit_factor is None or (bt_analysis.profit_factor!=float('inf') and bt_analysis.profit_factor < request.min_backtest_profit_factor)): logger.debug(f"[{symbol}] Filter Fail: PF={format_or_na(bt_analysis.profit_factor if bt_analysis.profit_factor != float('inf') else 999.9, '.2f')} < {request.min_backtest_profit_factor:.2f}"); passes_filters=False; filter_fail_reason=f"BT PF low ({format_or_na(bt_analysis.profit_factor if bt_analysis.profit_factor != float('inf') else 999.9, '.2f')})"
            if passes_filters and request.min_risk_reward_ratio is not None and request.min_risk_reward_ratio > 0: # F3 R:R
                entry=gpt_params.optimal_entry if gpt_params else None; sl=gpt_params.stop_loss if gpt_params else None; tp=gpt_params.take_profit if gpt_params else None; rr_ratio = None
                if (
                    entry is not None and
                    sl is not None and
                    tp is not None and
                    np.isfinite(entry) and
                    np.isfinite(sl) and
                    np.isfinite(tp)
                ):
                    risk = abs(entry - sl)
                    reward = abs(tp - entry)
                    if risk > 1e-9:
                        rr_ratio = reward / risk


                logger.debug(f"[{symbol}] Filter Check R/R: Got={format_or_na(rr_ratio,'.2f')}, Req>={request.min_risk_reward_ratio:.2f}")
                if rr_ratio is None or rr_ratio < request.min_risk_reward_ratio: passes_filters=False; filter_fail_reason=f"R/R Ratio low ({format_or_na(rr_ratio,'.2f')})"
            if passes_filters and request.require_sma_alignment: # F4 SMA Align (Optional for Breakout)
                 if indicators is None: passes_filters=False; filter_fail_reason="Indicators missing for SMA align"
                 else: # ... (garder la logique SMA align check précédente avec log debug) ...
                      sma50=indicators.SMA_50; sma200=indicators.SMA_200; price=current_price; sma_aligned=False
                      if all(v is not None and np.isfinite(v) for v in [price, sma50, sma200]):
                          if eval_direction=='long' and price > sma50 > sma200: sma_aligned=True
                          elif eval_direction=='short' and price < sma50 < sma200: sma_aligned=True
                      logger.debug(f"[{symbol}] Filter Check SMA Align (Req={request.require_sma_alignment}): Aligned={sma_aligned}")
                      if not sma_aligned: passes_filters=False; filter_fail_reason="SMA alignment failed"
            if passes_filters and apply_btc_filter and btc_trend_state not in ["UNKNOWN", "ERROR"]: # F5 BTC Trend
                btc_alignment_ok = (eval_direction=='long' and btc_trend_state=='UPTREND') or (eval_direction=='short' and btc_trend_state=='DOWNTREND') or (btc_trend_state=='CHOPPY') # Allow choppy?
                logger.debug(f"[{symbol}] Filter Check BTC Trend (Filter On={apply_btc_filter}): BTC_State='{btc_trend_state}', EvalDir='{eval_direction}', AlignOK={btc_alignment_ok}")
                if not btc_alignment_ok: passes_filters=False; filter_fail_reason=f"BTC Trend filter failed (Dir:{eval_direction}, BTC:{btc_trend_state})"
            if passes_filters: # Add Opportunity
                logger.info(f"[{result.symbol}] PASSED ALL FILTERS. Adding to opportunities.")
                score_g = float(gpt_conf) if gpt_conf is not None else 0.0; score_b = float(bt_score) if bt_score is not None else 0.0
                combined_score = round((score_g * 0.6) + (score_b * 0.4), 3) 
                summary="Analysis unavailable."; # ... (logique extraction summary précédente) ...
                if result.gptAnalysis:
                    primary_summary = (
                        result.gptAnalysis.signal_evaluation
                        or result.gptAnalysis.technical_analysis
                    )
                    if (
                        primary_summary
                        and isinstance(primary_summary, str)
                        and "Error:" not in primary_summary
                    ):
                        summary = (primary_summary.split('.')[0] + '.').strip()
                    elif result.gptAnalysis.raw_text and isinstance(result.gptAnalysis.raw_text, str):
                        summary = (result.gptAnalysis.raw_text.split('.')[0] + '.').strip()

                if len(summary) > 150:
                    summary = summary[:147] + "..."

                opportunity = ScanResultItem(
                    rank=0,
                    symbol=result.symbol,
                    timeframe=result.timeframe,
                    currentPrice=current_price,
                    gptConfidence=gpt_conf,
                    backtestScore=bt_score,
                    combinedScore=combined_score,
                    tradeDirection=eval_direction,
                    optimalEntry=gpt_params.optimal_entry if gpt_params else None,
                    stopLoss=gpt_params.stop_loss if gpt_params else None,
                    takeProfit=gpt_params.take_profit if gpt_params else None,
                    gptAnalysisSummary=summary
                )

                opportunities_passing_filter.append(opportunity)

            else: 
                 if not is_skip: logger.info(f"[{result.symbol}] FAILED FILTERS. Reason: {filter_fail_reason}.")
        else: logger.error(f"API Scan: UNEXPECTED result type for {symbol}: {type(res_or_exc)}."); analysis_errors[symbol]=f"Unexpected Result Type: {type(res_or_exc).__name__}"
    # --- Rank Opportunities ---
    if opportunities_passing_filter:
        logger.info(f"API Scan: Sorting {len(opportunities_passing_filter)} opportunities...")
        opportunities_passing_filter.sort(key=lambda x: x.combinedScore or 0.0, reverse=True)
        top_opportunities = []
        for rank_idx, opp in enumerate(opportunities_passing_filter[:request.top_n]): opp.rank=rank_idx+1; top_opportunities.append(opp)
        logger.info(f"API Scan: Ranking complete. Top {len(top_opportunities)} selected.")
    else: logger.info("API Scan: No opportunities passed filters."); top_opportunities = []
    # --- Construct Final Response ---
    scan_duration=time.time()-scan_start_time; logger.info(f"API Scan: Completed in {scan_duration:.2f}s.")
    if analysis_errors: logger.warning(f"API Scan finished with {len(analysis_errors)} errors for symbols: {list(analysis_errors.keys())}")
    return ScanResponse(scan_parameters=request, total_tickers_attempted=total_attempted, total_tickers_succeeded=successful_analyses_count, ticker_start_index=start_index, ticker_end_index=actual_end_index_for_response, total_opportunities_found=len(opportunities_passing_filter), top_opportunities=top_opportunities, errors=analysis_errors)

# --- Main Execution ---
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
    # --- MISE À JOUR : Indiquer la stratégie Breakout ---
    print(f" Initial Signal Trigger: Breakout Strategy (Default N=20, Vol>1.5x, RSI>55/<45)") 
    print("="*30 + "\n")
    uvicorn.run(
        # Assurez-vous que 'main_runner' est le nom de votre fichier .py ou utilisez "__main__"
        "__main__:app",  # Ou "adiotan:app" si le fichier s'appelle adiotan.py
        host=host, 
        port=port, 
        reload=reload_flag, 
        log_level=uvicorn_log_level
    )