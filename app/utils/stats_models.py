# app/utils/stats_models.py
import pandas as pd
import numpy as np
import logging
import warnings
from arch import arch_model
from typing import Optional

# Get a logger for this module
logger = logging.getLogger(__name__)

# --- Statistical Models (GARCH, VaR) ---
def fit_garch_model(returns: pd.Series, symbol_log: str = "Unknown") -> Optional[float]:
    """Fit GARCH(1,1) model. Returns NEXT PERIOD conditional volatility."""
    # Ensure returns are numeric and drop NaNs
    valid_returns = pd.to_numeric(returns, errors='coerce').dropna() * 100 # Scale for GARCH
    logger.debug(f"[{symbol_log}] GARCH input len after dropna/scaling: {len(valid_returns)}")

    # Check if sufficient data points remain
    if len(valid_returns) < 50: # Need a reasonable number of observations
        logger.warning(f"[{symbol_log}] Skipping GARCH, need at least 50 valid returns, got {len(valid_returns)}.")
        return None

    try:
        # Define the GARCH(1,1) model
        # Use 'Garch' (or 'ARCH', 'EGARCH' etc.) for vol='...'
        # Common distributions: 'Normal', 't', 'skewt'
        am = arch_model(valid_returns, vol='Garch', p=1, q=1, dist='Normal')

        # Fit the model, suppress output and convergence warnings during fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Ignore convergence warnings here
            # disp='off' prevents printing fitting output
            res = am.fit(update_freq=0, disp='off', show_warning=False)

        # Check convergence AFTER fitting
        if res.convergence_flag == 0:
            # Forecast 1 step ahead
            forecasts = res.forecast(horizon=1, reindex=False) # reindex=False uses the last index for forecast
            # Extract forecasted variance, take sqrt for volatility, unscale (/100)
            cond_vol_forecast = np.sqrt(forecasts.variance.iloc[-1, 0]) / 100.0
            logger.debug(f"[{symbol_log}] GARCH fit successful. Forecast Vol: {cond_vol_forecast:.6f}")
            # Return forecast only if it's a finite number
            return float(cond_vol_forecast) if np.isfinite(cond_vol_forecast) else None
        else:
            logger.warning(f"[{symbol_log}] GARCH model did not converge (Flag: {res.convergence_flag}). Result: {res.summary()}")
            return None
    except Exception as e:
        # Catch potential errors during model fitting (e.g., singular matrix)
        logger.error(f"[{symbol_log}] GARCH fitting error: {e}", exc_info=False)
        return None

def calculate_var(returns: pd.Series, confidence_level: float = 0.95, symbol_log: str = "Unknown") -> Optional[float]:
    """Calculate Historical Value at Risk (VaR) at specified confidence level."""
    # Ensure returns are numeric and drop NaNs
    valid_returns = pd.to_numeric(returns, errors='coerce').dropna()
    logger.debug(f"[{symbol_log}] VaR input len after dropna: {len(valid_returns)}")

    # Check if sufficient data points remain
    if len(valid_returns) < 20: # Need some data for percentile calculation
        logger.warning(f"[{symbol_log}] Skipping VaR, need at least 20 valid returns, got {len(valid_returns)}.")
        return None

    try:
        # Calculate the percentile corresponding to the confidence level
        # For VaR (potential loss), we look at the lower tail percentile
        var_percentile = (1.0 - confidence_level) * 100.0
        var_value = np.percentile(valid_returns, var_percentile)
        logger.debug(f"[{symbol_log}] VaR calculated: {var_value:.6f} at {confidence_level*100:.0f}% confidence (Percentile: {var_percentile:.2f}).")
        # Return VaR only if it's a finite number
        return float(var_value) if np.isfinite(var_value) else None
    except Exception as e:
        logger.error(f"[{symbol_log}] Error calculating VaR: {e}", exc_info=False)
        return None