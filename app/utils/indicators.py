# Technical indicator calculations
# app/utils/indicators.py
import pandas as pd
import numpy as np
import logging

# Get a logger for this module
logger = logging.getLogger(__name__)

# --- Indicator Functions (compute_*, apply_technical_indicators) ---
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).fillna(0)
    loss = -delta.where(delta < 0, 0.0).fillna(0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan) # Avoid division by zero
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50) # Fill initial NaNs with 50 (neutral)

def compute_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    atr = tr.ewm(com=window - 1, min_periods=window).mean()
    return atr

def compute_bollinger_bands(series, window=20):
    sma = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    return upper_band, sma, lower_band # Return middle band (SMA) too

def compute_stochastic_oscillator(df, window=14, smooth_k=3):
    lowest_low = df['low'].rolling(window=window, min_periods=window).min()
    highest_high = df['high'].rolling(window=window, min_periods=window).max()
    range_hh_ll = (highest_high - lowest_low).replace(0, np.nan) # Avoid division by zero
    k_percent = 100 * ((df['close'] - lowest_low) / range_hh_ll)
    d_percent = k_percent.rolling(window=smooth_k, min_periods=smooth_k).mean()
    return k_percent.fillna(50), d_percent.fillna(50) # Fill NaN 50

def compute_williams_r(df, window=14):
    highest_high = df['high'].rolling(window=window, min_periods=window).max()
    lowest_low = df['low'].rolling(window=window, min_periods=window).min()
    range_ = (highest_high - lowest_low).replace(0, np.nan) # Avoid division by zero
    williams_r = -100 * ((highest_high - df['close']) / range_)
    return williams_r.fillna(-50) # Fill NaN -50 (middle)

def compute_adx(df, window=14):
    # Ensure input df has required columns
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        raise ValueError("DataFrame must contain 'high', 'low', 'close' columns for ADX calculation")

    df_adx = df.copy() # Work on a copy

    # Calculate True Range (TR)
    df_adx['H-L'] = df_adx['high'] - df_adx['low']
    df_adx['H-PC'] = abs(df_adx['high'] - df_adx['close'].shift(1))
    df_adx['L-PC'] = abs(df_adx['low'] - df_adx['close'].shift(1))
    # Ensure TR is non-negative and handle potential NaNs from shift
    df_adx['TR_calc'] = df_adx[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False).fillna(0) # Fill initial NaN TR with 0

    # Calculate Directional Movement (+DM, -DM)
    df_adx['delta_high'] = df_adx['high'].diff()
    df_adx['delta_low'] = df_adx['low'].diff()

    df_adx['DMplus_raw'] = np.where((df_adx['delta_high'] > df_adx['delta_low']) & (df_adx['delta_high'] > 0), df_adx['delta_high'], 0)
    df_adx['DMminus_raw'] = np.where((df_adx['delta_low'] > df_adx['delta_high']) & (df_adx['delta_low'] > 0), df_adx['delta_low'], 0)

    # Wilder's Smoothing (similar to EMA with alpha = 1/N)
    # Use adjust=False for compatibility with traditional Wilder's method
    # Add min_periods to avoid initial calculations producing only NaN
    TR_smooth = df_adx['TR_calc'].ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    DMplus_smooth = df_adx['DMplus_raw'].ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    DMminus_smooth = df_adx['DMminus_raw'].ewm(alpha=1/window, adjust=False, min_periods=window).mean()

    # Calculate Directional Indicators (+DI, -DI)
    # Handle division by zero if TR_smooth is zero
    DIplus = 100 * (DMplus_smooth / TR_smooth.replace(0, np.nan)).fillna(0)
    DIminus = 100 * (DMminus_smooth / TR_smooth.replace(0, np.nan)).fillna(0)

    # Calculate Directional Movement Index (DX)
    DIsum = (DIplus + DIminus).replace(0, np.nan) # Avoid division by zero
    DX = 100 * (abs(DIplus - DIminus) / DIsum).fillna(0) # Fill initial NaN DX with 0

    # Calculate Average Directional Index (ADX)
    ADX = DX.ewm(alpha=1/window, adjust=False, min_periods=window * 2 -1 ).mean() # ADX needs more periods to stabilize

    # Handle potential NaNs in the final ADX result, especially at the beginning
    # Filling with 20 might be reasonable assumption for low-trend start
    return ADX.fillna(20)

def compute_cci(df, window=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=window, min_periods=window).mean()
    # Use pandas std() for Mean Absolute Deviation calculation for simplicity and robustness
    mad = tp.rolling(window=window, min_periods=window).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    # mad = tp.rolling(window=window, min_periods=window).apply(lambda x: pd.Series(x).mad(), raw=True) # Deprecated mad()
    cci = (tp - sma) / (0.015 * mad.replace(0, np.nan)) # Avoid division by zero
    return cci.fillna(0) # Fill NaN CCI 0 (center line)

def compute_obv(df):
    # Ensure volume is numeric
    volume = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    close_diff = df['close'].diff()
    # Calculate OBV changes: volume if close increases, -volume if close decreases, 0 otherwise
    obv_changes = np.where(close_diff > 0, volume, np.where(close_diff < 0, -volume, 0))
    # Calculate cumulative sum, starting from 0
    obv = np.cumsum(obv_changes)
    # Convert to Series with original index
    return pd.Series(obv, index=df.index).fillna(0)

# <<< --- THIS IS THE FUNCTION YOU NEED TO IMPORT --- >>>
def apply_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply technical indicators."""
    if df.empty:
        logger.warning("Input DataFrame is empty, cannot apply indicators.")
        return df

    df_copy = df.copy()
    # Ensure numeric types where expected, coerce errors to NaN
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaN in essential OHLCV columns after conversion
    df_copy.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
    if df_copy.empty:
        logger.warning("DataFrame became empty after coercing OHLCV to numeric and dropping NaNs.")
        return df_copy

    df_copy['returns'] = np.log(df_copy['close'] / df_copy['close'].shift(1)).fillna(0)
    df_len = len(df_copy)
    symbol_log = df_copy.get('symbol', 'Unknown') # Get symbol if passed in df
    logger.debug(f"[{symbol_log}] Applying indicators to {df_len} candles.")

    # Helper to safely assign calculated indicator
    def assign_if_enough_data(col_name, min_len_needed, calculation_lambda):
        if df_len >= min_len_needed:
            try:
                result = calculation_lambda()
                # Ensure result has same index as df_copy for proper alignment
                if isinstance(result, pd.Series):
                    df_copy[col_name] = result.reindex(df_copy.index)
                elif isinstance(result, tuple) and len(result) > 0: # Handle functions returning tuples (e.g., BBands)
                    # Assuming the tuple elements correspond to column names expected
                    # This part needs to be handled specifically where called if func returns tuple
                     pass # Handled specifically below
                else:
                     # Handle scalar or unexpected types if necessary
                     df_copy[col_name] = result # This might broadcast if result is scalar
            except Exception as e:
                logger.error(f"[{symbol_log}] Error calculating {col_name}: {e}", exc_info=False)
                df_copy[col_name] = np.nan
        else:
            logger.debug(f"[{symbol_log}] Skipping {col_name}, need {min_len_needed} valid candles, got {df_len}.")
            df_copy[col_name] = np.nan # Assign NaN column if not enough data

    # --- Calculate Indicators ---
    # Moving Averages (require window size)
    assign_if_enough_data('SMA_50', 50, lambda: df_copy['close'].rolling(window=50, min_periods=50).mean())
    assign_if_enough_data('SMA_200', 200, lambda: df_copy['close'].rolling(window=200, min_periods=200).mean())
    assign_if_enough_data('EMA_12', 12, lambda: df_copy['close'].ewm(span=12, adjust=False, min_periods=12).mean()) # Need 12 periods for EMA12
    assign_if_enough_data('EMA_26', 26, lambda: df_copy['close'].ewm(span=26, adjust=False, min_periods=26).mean()) # Need 26 periods for EMA26

    # MACD (depends on EMA_12, EMA_26) - Requires 26 periods minimum
    if 'EMA_12' in df_copy and 'EMA_26' in df_copy and df_copy['EMA_12'].notna().any() and df_copy['EMA_26'].notna().any():
        df_copy['MACD'] = df_copy['EMA_12'] - df_copy['EMA_26']
        # Signal Line (depends on MACD) - Requires 9 periods of MACD, so 26 + 9 - 1 = 34 periods total
        assign_if_enough_data('Signal_Line', 34, lambda: df_copy['MACD'].ewm(span=9, adjust=False, min_periods=9).mean())
    else:
        df_copy['MACD'], df_copy['Signal_Line'] = np.nan, np.nan

    # Other Indicators
    assign_if_enough_data('RSI', 15, lambda: compute_rsi(df_copy['close'], window=14)) # RSI needs window + 1
    assign_if_enough_data('ATR', 15, lambda: compute_atr(df_copy, window=14)) # ATR needs window + 1

    # Bollinger Bands (needs window)
    if df_len >= 20:
        try:
            # compute_bollinger_bands returns a tuple (upper, middle, lower)
            upper, middle, lower = compute_bollinger_bands(df_copy['close'], window=20)
            df_copy['Bollinger_Upper'] = upper.reindex(df_copy.index)
            df_copy['Bollinger_Middle'] = middle.reindex(df_copy.index) # Also known as SMA 20
            df_copy['Bollinger_Lower'] = lower.reindex(df_copy.index)
        except Exception as e:
            logger.error(f"[{symbol_log}] Error calculating Bollinger Bands: {e}", exc_info=False)
            df_copy['Bollinger_Upper'], df_copy['Bollinger_Middle'], df_copy['Bollinger_Lower'] = np.nan, np.nan, np.nan
    else:
        logger.debug(f"[{symbol_log}] Skipping Bollinger Bands, need 20, got {df_len}.")
        df_copy['Bollinger_Upper'], df_copy['Bollinger_Middle'], df_copy['Bollinger_Lower'] = np.nan, np.nan, np.nan


    # Momentum (requires lookback period + 1)
    assign_if_enough_data('Momentum', 11, lambda: df_copy['close'].diff(10))

    # Stochastic Oscillator (needs window + smooth_k) -> 14 + 3 = 17
    if df_len >= 17: # 14 for %K window, 3 for %D smoothing
        try:
            # compute_stochastic_oscillator returns a tuple (k, d)
            k, d = compute_stochastic_oscillator(df_copy, window=14, smooth_k=3)
            df_copy['Stochastic_K'] = k.reindex(df_copy.index)
            df_copy['Stochastic_D'] = d.reindex(df_copy.index)
        except Exception as e:
            logger.error(f"[{symbol_log}] Error calculating Stochastic: {e}", exc_info=False)
            df_copy['Stochastic_K'], df_copy['Stochastic_D'] = np.nan, np.nan
    else:
        logger.debug(f"[{symbol_log}] Skipping Stochastic Oscillator, need 17, got {df_len}.")
        df_copy['Stochastic_K'], df_copy['Stochastic_D'] = np.nan, np.nan

    # Williams %R (needs window + 1)
    assign_if_enough_data('Williams_%R', 15, lambda: compute_williams_r(df_copy, window=14)) # Alias handled by Pydantic model

    # ADX (requires ~2*window) -> ~28 for stable calculation
    assign_if_enough_data('ADX', 28, lambda: compute_adx(df_copy, window=14))

    # CCI (requires window + 1)
    assign_if_enough_data('CCI', 21, lambda: compute_cci(df_copy, window=20))

    # OBV (requires just 2 periods for diff)
    assign_if_enough_data('OBV', 2, lambda: compute_obv(df_copy))

    logger.debug(f"[{symbol_log}] Finished applying indicators.")
    return df_copy