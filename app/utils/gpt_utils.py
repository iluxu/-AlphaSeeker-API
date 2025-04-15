# app/utils/gpt_utils.py
import openai
import json
import re
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

# Import the client and settings - assumes clients.py and config.py are set up
# Make sure these imports are correct based on your project structure
try:
    # If running within the structured app, use relative imports if needed
    # or ensure PYTHONPATH includes the project root.
    from app.core.clients import openai_client
    from app.core.config import settings
except ImportError as e:
    # Provide a more helpful error message if imports fail during development
    # This might happen if running this file directly or if structure is wrong
    print(f"Error importing from app.core in gpt_utils.py: {e}")
    print("Ensure app/core/clients.py and app/core/config.py exist and are correct,")
    print("and that the script is run in a context where 'app' is importable (e.g., from project root).")
    # Depending on requirements, you might raise the error or set defaults for testing
    openai_client = None
    settings = None # Or a default settings object if applicable

# Get a logger for this module
# Assumes logging is configured elsewhere (e.g., in config.py or main entry point)
logger = logging.getLogger(__name__) # Get logger specific to this module

# --- GPT Integration ---
def gpt_generate_trading_parameters(
    df_with_indicators: pd.DataFrame,
    symbol: str,
    timeframe: str,
    account_balance: float,
    max_leverage: float,
    garch_volatility: Optional[float],
    var95: Optional[float],
    technically_derived_direction: str, # The 'long'/'short'/'hold' derived from signal logic
    min_requested_rr: Optional[float] # Pass the R/R from ScanRequest here
) -> str:
    """Generate trading parameters using GPT to EVALUATE a technical signal."""
    log_prefix = f"[{symbol} ({timeframe}) GPT]"

    # Check if openai_client was initialized correctly
    if openai_client is None:
        logger.warning(f"{log_prefix} OpenAI client not available. Skipping GPT evaluation.")
        # Return a JSON string indicating the error and defaulting to hold
        return json.dumps({"error": "OpenAI client not available", "trade_direction": "hold"})

    # Check for necessary columns in the DataFrame
    required_cols_gpt = ['close', 'ATR', 'MACD', 'Signal_Line', 'SMA_200', 'ADX'] # Key cols for prompt context
    if df_with_indicators is None or df_with_indicators.empty:
         logger.warning(f"{log_prefix} DataFrame is empty. Cannot evaluate.")
         return json.dumps({"error": "Input DataFrame is empty", "trade_direction": "hold"})

    # Check if latest row has the data needed - avoid dropping entire rows if only latest needed
    if len(df_with_indicators) == 0: # Double check after potential empty check
         logger.warning(f"{log_prefix} DataFrame is empty after initial check. Cannot evaluate.")
         return json.dumps({"error": "Input DataFrame is empty", "trade_direction": "hold"})

    latest_row = df_with_indicators.iloc[-1]
    # prev_row = df_with_indicators.iloc[-2] if len(df_with_indicators) >= 2 else None # Not strictly needed for prompt context here

    # Check specifically if required columns exist and are not null in the latest row
    missing_or_null_cols = []
    for col in required_cols_gpt:
        if col not in latest_row.index: # Check existence first
            missing_or_null_cols.append(f"{col} (missing)")
        elif pd.isnull(latest_row[col]): # Then check for null
            missing_or_null_cols.append(f"{col} (null)")

    if missing_or_null_cols:
        logger.warning(f"{log_prefix} Insufficient latest indicator data for GPT evaluation. Issues: {missing_or_null_cols}.")
        return json.dumps({"error": f"Insufficient latest indicator data for GPT ({', '.join(missing_or_null_cols)})", "trade_direction": "hold"})

    # Extract latest data, format for readability
    latest_data = latest_row.to_dict()
    technical_indicators = {}
    # Define keys expected by the prompt, ensure consistency
    key_inds_for_prompt = ['RSI', 'ATR', 'SMA_50', 'SMA_200', 'MACD', 'Signal_Line',
                           'Bollinger_Upper', 'Bollinger_Middle', 'Bollinger_Lower',
                           'Stochastic_K', 'Stochastic_D', 'ADX', 'CCI', 'Williams_%R'] # Use the DataFrame column names

    for key_in_df in key_inds_for_prompt:
        value = latest_data.get(key_in_df)
        # Use a prompt-friendly key name if needed, but stick to original if possible
        prompt_key = key_in_df.replace('%', '_pct') # Example sanitization for keys in JSON prompt if needed

        if pd.notna(value) and np.isfinite(value):
            # Basic formatting for readability in the prompt
            if abs(value) >= 1e5 or (abs(value) < 1e-4 and value != 0):
                technical_indicators[prompt_key] = f"{value:.3e}" # Scientific notation
            else:
                technical_indicators[prompt_key] = round(float(value), 4) # Standard rounding
        else:
            technical_indicators[prompt_key] = "N/A" # Indicate missing/invalid data clearly

    current_price = latest_data.get('close')
    # Ensure current_price is valid before proceeding
    if current_price is None or not np.isfinite(current_price):
        logger.error(f"{log_prefix} Missing or invalid current price for GPT context.")
        return json.dumps({"error": "Missing or invalid current price for GPT context", "trade_direction": "hold"})
    current_price = round(float(current_price), 4)

    # Format GARCH/VaR nicely
    garch_vol_str = f"{garch_volatility:.4%}" if garch_volatility is not None and np.isfinite(garch_volatility) else "N/A"
    var95_str = f"{var95:.4%}" if var95 is not None and np.isfinite(var95) else "N/A"

    # Provide context about the signal trigger more clearly
    signal_context = "N/A"
    macd_val = latest_data.get('MACD')
    signal_val = latest_data.get('Signal_Line')
    sma200_val = latest_data.get('SMA_200')
    adx_val = latest_data.get('ADX')

    # Format values safely for the string, checking for None and NaN/inf
    def format_value(val, precision):
        if pd.notna(val) and np.isfinite(val):
             return f"{val:.{precision}f}"
        return 'N/A'

    macd_val_str = format_value(macd_val, 4)
    signal_val_str = format_value(signal_val, 4)
    sma200_val_str = format_value(sma200_val, 4)
    adx_val_str = format_value(adx_val, 2)

    if technically_derived_direction == 'long':
        signal_context = f"Potential LONG signal triggered: MACD({macd_val_str}) crossed above Signal({signal_val_str}) while Price({current_price:.4f}) > SMA200({sma200_val_str}) and ADX({adx_val_str}) >= 20."
    elif technically_derived_direction == 'short':
        signal_context = f"Potential SHORT signal triggered: MACD({macd_val_str}) crossed below Signal({signal_val_str}) while Price({current_price:.4f}) < SMA200({sma200_val_str}) and ADX({adx_val_str}) >= 20."
    else:
        # This case might occur if called despite no technical signal
        signal_context = "No specific MACD/Trend/ADX signal triggered for evaluation (Hold state)."

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
        "minimum_target_rr": min_requested_rr or 1.5 # Use passed value or default R:R
    }

    # Convert dict to JSON string for the prompt
    try:
        # Use default=str as a fallback for non-serializable types, though values should be cleaned above
        data_json = json.dumps(market_info, indent=2, default=str)
        logger.debug(f"{log_prefix} Data prepared for GPT:\n{data_json}")
    except TypeError as e:
        logger.error(f"{log_prefix} Failed to serialize market info to JSON (check types): {e}")
        return json.dumps({"error": "Failed to serialize market data for GPT", "trade_direction": "hold"})
    except Exception as e:
        logger.error(f"{log_prefix} Unexpected error serializing market info: {e}")
        return json.dumps({"error": "Unexpected error serializing market data for GPT", "trade_direction": "hold"})

    # --- REVISED GPT PROMPT ---
    prompt = f"""You are a cryptocurrency trading analyst evaluating a potential trade setup based on a MACD crossover signal combined with a trend filter (SMA 200) and trend strength (ADX).
The system detected a potential signal: '{technically_derived_direction}'.
Your task is to EVALUATE this signal using the provided market data and technical indicators, and provide actionable parameters if appropriate."""

    # --- END OF GPT PROMPT ---
    try:
        logger.info(f"{log_prefix} Sending request to GPT (gpt-4o-mini) to evaluate '{technically_derived_direction}' signal...")
        # Use a capable model, consider cost/speed trade-offs (e.g., gpt-4o-mini, gpt-4-turbo)
        response = openai_client.chat.completions.create(
                    model="gpt-4o-mini", # Balance of capability, speed, cost
                    messages=[
                        {"role": "system", "content": "You are a crypto trading analyst. Respond ONLY in valid JSON format as specified in the user prompt."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}, # Enforce JSON output
                    temperature=0.2, # Low temperature for more deterministic analysis
                    max_tokens=1200 # Adjust based on expected response length and model limits
                )
        gpt_output = response.choices[0].message.content
        # Basic check if response seems like valid JSON before returning
        if gpt_output and isinstance(gpt_output, str) and gpt_output.strip().startswith('{') and gpt_output.strip().endswith('}'):
            logger.debug(f"{log_prefix} Raw valid-looking GPT Output received (length: {len(gpt_output)}).")
            return gpt_output
        else:
             logger.warning(f"{log_prefix} GPT returned empty or invalid-looking response: {str(gpt_output)[:100]}...")
             return json.dumps({"error": "GPT returned empty or invalid response structure", "trade_direction": "hold"})

    except openai.RateLimitError as e:
        logger.error(f"{log_prefix} OpenAI Rate Limit Error: {e}")
        return json.dumps({"error": "OpenAI API rate limit exceeded", "details": str(e), "trade_direction": "hold"})
    except openai.APIConnectionError as e:
        logger.error(f"{log_prefix} OpenAI API Connection Error: {e}")
        return json.dumps({"error": "OpenAI API connection error", "details": str(e), "trade_direction": "hold"})
    except openai.AuthenticationError as e:
         logger.error(f"{log_prefix} OpenAI Authentication Error: {e}")
         return json.dumps({"error": "OpenAI authentication error (check API key)", "details": str(e), "trade_direction": "hold"})
    except openai.APIStatusError as e: # Catch broader API errors (e.g., 500s, 400s)
        logger.error(f"{log_prefix} OpenAI API Status Error: {e.status_code} - {e.response}")
        error_detail = str(e)
        try: # Try to extract detail from response body if possible
             error_detail = e.response.json().get('error', {}).get('message', str(e))
        except: pass
        return json.dumps({"error": f"OpenAI API status error: {e.status_code}", "details": error_detail, "trade_direction": "hold"})
    except Exception as e:
        logger.error(f"{log_prefix} Unexpected error querying OpenAI: {e}", exc_info=True) # Log full traceback
        return json.dumps({"error": "Unexpected error querying OpenAI", "details": str(e), "trade_direction": "hold"})

def parse_gpt_trading_parameters(gpt_output_str: str, symbol_for_log: str = "") -> Dict[str, Any]:
    log_prefix = f"[{symbol_for_log} Parse]"
    # Initialize with defaults, especially 'hold' and 0 confidence
    parsed_data = {
    'optimal_entry': None, 'stop_loss': None, 'take_profit': None,
    'trade_direction': 'hold', # Default to hold unless explicitly overridden by valid GPT response
    'leverage': None, 'position_size_usd': None, 'estimated_profit': None,
    'confidence_score': 0.0, # Default confidence to 0.0
    # Store raw text for debugging, analysis dict gets populated
    'analysis': {'signal_evaluation': None, 'technical_analysis': None, 'risk_assessment': None, 'market_outlook': None, 'raw_text': gpt_output_str}
    }
    if not gpt_output_str or not isinstance(gpt_output_str, str):
        logger.error(f"{log_prefix} Received empty or non-string input for parsing.")
        parsed_data['analysis']['signal_evaluation'] = "Error: Received invalid input for GPT parsing."
        return parsed_data # Return defaults with error
    try:
        # Clean potential markdown ```json ... ``` artifacts more robustly
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', gpt_output_str, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1)
            logger.debug(f"{log_prefix} Extracted JSON block from markdown.")
        else:
            # Attempt to find JSON object even without markdown fences - check if string IS a JSON object
            json_match = re.search(r'^\s*\{.*\}\s*$', gpt_output_str, re.DOTALL)
            if json_match:
                 json_str = json_match.group(0).strip() # Use the whole string if it looks like JSON
                 logger.debug(f"{log_prefix} Assuming the entire string is JSON.")
            else:
                 # If no clear JSON structure found, parsing will likely fail
                 json_str = gpt_output_str
                 logger.warning(f"{log_prefix} Could not clearly identify JSON structure. Attempting parse anyway.")

        # Attempt to load the JSON
        data = json.loads(json_str)
        if not isinstance(data, dict):
            # This should ideally not happen if json.loads succeeds, but check defensively
            raise json.JSONDecodeError("GPT output parsed but was not a dictionary", json_str, 0)
        logger.debug(f"{log_prefix} Successfully decoded JSON from GPT.")

        # Helper to safely get numeric values, returning None if conversion fails or not finite
        def get_numeric(key, target_type=float):
            val = data.get(key)
            if val is None: return None # Key not present
            # Handle case where GPT might return string "N/A" or similar despite prompt instructions
            if isinstance(val, str) and val.strip().lower() in ['n/a', 'none', 'null', '']:
                 return None
            try:
                num_val = target_type(val) # Attempt conversion (e.g., float(), int())
                # Check for NaN or infinity
                return num_val if np.isfinite(num_val) else None
            except (ValueError, TypeError):
                logger.warning(f"{log_prefix} Could not convert key '{key}' value '{val}' to {target_type.__name__}. Type: {type(val)}")
                return None

        # Get core trade parameters using the safe helper
        parsed_data['optimal_entry'] = get_numeric('optimal_entry', float)
        parsed_data['stop_loss'] = get_numeric('stop_loss', float)
        parsed_data['take_profit'] = get_numeric('take_profit', float)
        parsed_data['position_size_usd'] = get_numeric('position_size_usd', float)
        parsed_data['estimated_profit'] = get_numeric('estimated_profit', float)
        parsed_data['confidence_score'] = get_numeric('confidence_score', float)
        parsed_data['leverage'] = get_numeric('leverage', int)

        # Clamp confidence score between 0.0 and 1.0 AFTER getting it
        if parsed_data['confidence_score'] is not None:
            parsed_data['confidence_score'] = max(0.0, min(1.0, parsed_data['confidence_score']))
        else:
            # If get_numeric returned None, default confidence should be 0.0
            parsed_data['confidence_score'] = 0.0

        # Validate leverage (must be >= 1)
        if parsed_data['leverage'] is not None and parsed_data['leverage'] < 1:
            logger.warning(f"{log_prefix} GPT suggested leverage < 1 ({parsed_data['leverage']}), setting to None.")
            parsed_data['leverage'] = None

        # --- Trade Direction (Crucial - determines if params are relevant) ---
        direction = data.get('trade_direction')
        # Validate direction before assigning
        if isinstance(direction, str) and direction.lower() in ['long', 'short', 'hold']:
            parsed_data['trade_direction'] = direction.lower()
            logger.info(f"{log_prefix} GPT evaluation resulted in direction: '{parsed_data['trade_direction']}' with Conf: {parsed_data['confidence_score']:.2f}")
        else:
            logger.warning(f"{log_prefix} Invalid or missing 'trade_direction' from GPT: '{direction}'. Defaulting to 'hold'.")
            parsed_data['trade_direction'] = 'hold' # Ensure it defaults to hold if invalid

        # --- Validate parameters *only if* GPT suggests a trade ('long' or 'short') ---
        if parsed_data['trade_direction'] in ['long', 'short']:
            required_params = ['optimal_entry', 'stop_loss', 'take_profit'] # Core params for a trade
            missing_params = [p for p in required_params if parsed_data[p] is None]
            if missing_params:
                logger.warning(f"{log_prefix} GPT suggested '{parsed_data['trade_direction']}' but missing required params: {missing_params}. Forcing 'hold'.")
                parsed_data['trade_direction'] = 'hold'
            else:
                # Basic sanity check on SL/TP relative to entry
                entry = parsed_data['optimal_entry']
                sl = parsed_data['stop_loss']
                tp = parsed_data['take_profit']
                logical_levels = False
                if parsed_data['trade_direction'] == 'long' and sl < entry < tp:
                    logical_levels = True
                elif parsed_data['trade_direction'] == 'short' and tp < entry < sl:
                    logical_levels = True

                if not logical_levels:
                    logger.warning(f"{log_prefix} GPT levels illogical for '{parsed_data['trade_direction']}' (E:{entry:.4f}, SL:{sl:.4f}, TP:{tp:.4f}). Forcing 'hold'.")
                    parsed_data['trade_direction'] = 'hold'
                else:
                    # Check for zero risk only if levels are logical otherwise
                    risk = abs(entry - sl)
                    # Use a small epsilon for float comparison
                    if risk < 1e-9: # Avoid division by zero or nonsensical trade
                        logger.warning(f"{log_prefix} GPT suggested zero or negligible risk (Entry ~= SL = {entry:.6f}). Forcing 'hold'.")
                        parsed_data['trade_direction'] = 'hold'

        # --- Process Analysis Text ---
        analysis_dict = data.get('analysis')
        if isinstance(analysis_dict, dict):
            for key in ['signal_evaluation', 'technical_analysis', 'risk_assessment', 'market_outlook']:
                val = analysis_dict.get(key)
                # Ensure value is a non-empty string before assigning
                if isinstance(val, str) and val.strip():
                    parsed_data['analysis'][key] = val.strip()
                else:
                    # Keep as None if missing or empty
                    parsed_data['analysis'][key] = None # Explicitly set to None
        else:
            logger.warning(f"{log_prefix} 'analysis' section missing or not a dict in GPT response.")
            # Keep default None values for analysis fields

    except json.JSONDecodeError as e:
        logger.error(f"{log_prefix} Failed to decode JSON from GPT: {e}. Raw response start: '{gpt_output_str[:200]}...'")
        # Ensure defaults are set correctly on error
        parsed_data['trade_direction'] = 'hold'
        parsed_data['analysis']['signal_evaluation'] = f"Error: Failed to parse GPT JSON response. Details: {e}"
        parsed_data['confidence_score'] = 0.0
        # Nullify trade params explicitly
        for k in ['optimal_entry', 'stop_loss', 'take_profit', 'leverage', 'position_size_usd', 'estimated_profit']: parsed_data[k] = None

    except Exception as e:
        logger.error(f"{log_prefix} Unexpected error parsing GPT response: {e}", exc_info=True)
        # Ensure defaults are set correctly on error
        parsed_data['trade_direction'] = 'hold'
        parsed_data['analysis']['signal_evaluation'] = f"Error: Unexpected error parsing GPT response. Details: {e}"
        parsed_data['confidence_score'] = 0.0
        # Nullify trade params explicitly
        for k in ['optimal_entry', 'stop_loss', 'take_profit', 'leverage', 'position_size_usd', 'estimated_profit']: parsed_data[k] = None


    # Final cleanup: If final direction is 'hold' (either from GPT or forced by validation/error),
    # ensure trade parameters are None.
    if parsed_data['trade_direction'] == 'hold':
        # Log if we are overriding parameters that might have been parsed
        # if any(parsed_data[k] is not None for k in ['optimal_entry', 'stop_loss', 'take_profit', 'leverage']):
        #     logger.info(f"{log_prefix} Final direction is 'hold', ensuring trade parameters (E, SL, TP, Lev) are nullified.")
        parsed_data['optimal_entry'] = None
        parsed_data['stop_loss'] = None
        parsed_data['take_profit'] = None
        parsed_data['leverage'] = None
        parsed_data['position_size_usd'] = None
        parsed_data['estimated_profit'] = None
        # Keep confidence score as it reflects GPT's assessment, even if we override to hold.
        # Alternatively, set confidence to None or 0 for 'hold' if desired.
        # parsed_data['confidence_score'] = 0.0 # Or None

    # Log the parsed parameters, excluding the potentially long analysis text and raw text for brevity
    log_params = {k: v for k, v in parsed_data.items() if k not in ['analysis', 'raw_text']}
    logger.debug(f"{log_prefix} Parsed GPT Params: {json.dumps(log_params, indent=2)}")

    return parsed_data