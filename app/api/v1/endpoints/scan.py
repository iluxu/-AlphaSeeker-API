# Scan endpoint logic
# app/api/v1/endpoints/scan.py

import asyncio
import time
import logging
from fastapi import APIRouter, HTTPException, Depends # Added Depends if needed later
from typing import List, Dict, Any, Optional

# Import models for request and response
# Ensure these paths are correct relative to your project root
try:
    from app.models.scan import ScanRequest, ScanResponse, ScanResultItem
    from app.models.analysis import AnalysisResponse # Needed by the service call wrapper
    from app.models.common import TickersResponse # Potentially needed if getting tickers here
except ImportError as e:
    print(f"Error importing models in scan.py: {e}")
    # Handle error or raise if models are essential

# Import the core analysis service function used by the scanner
try:
    from app.services.analysis_service import perform_single_analysis
except ImportError as e:
    print(f"Error importing analysis_service in scan.py: {e}")
    # Handle error or raise

# Import core components (clients, settings, utility functions if needed directly)
try:
    from app.core.clients import binance_futures, load_exchange_markets # Need exchange client
    from app.core.config import settings # Might use settings for defaults or config
except ImportError as e:
    print(f"Error importing core components in scan.py: {e}")
    # Handle error or raise

# Import utility functions needed directly in this endpoint (e.g., for BTC trend)
try:
    from app.utils.data_fetcher import get_real_time_data # For BTC trend
    from app.utils.indicators import apply_technical_indicators # For BTC trend
except ImportError as e:
    print(f"Error importing utils in scan.py: {e}")
    # Handle error or raise

import pandas as pd
import numpy as np

# Get a logger for this module
# Assumes logging is configured elsewhere
logger = logging.getLogger(__name__)

# --- DEFINE THE ROUTER INSTANCE --- <<< REQUIRED LINE
router = APIRouter()
# --- END ROUTER DEFINITION ---

# --- The Scan Endpoint Function ---
# Decorate the function with the router instance
@router.post("/scan", response_model=ScanResponse, tags=["Scanning"])
async def scan_market_endpoint(request: ScanRequest):
    # --- Start of the scan_market_endpoint function from original script ---
    logger.critical("--- SCAN ENDPOINT ENTERED ---")
    scan_start_time = time.time()
    # Use request.model_dump_json() for cleaner logging of input
    logger.info(f"API Scan: Starting request: {request.model_dump_json(exclude_defaults=False, exclude_none=True)}") # Exclude None for cleaner log

    # --- Lookback Warning ---
    binance_1m_limit = 1500
    effective_lookback = request.lookback
    if request.timeframe == '1m' and request.lookback > binance_1m_limit:
         logger.warning(f"API Scan: Lookback {request.lookback} for 1m timeframe likely exceeds fetch limits ({binance_1m_limit}). Analysis will use max available.")
         effective_lookback = binance_1m_limit # Use capped value for analysis call estimate
    elif request.lookback > 1500: # General warning for other timeframes too
         logger.warning(f"API Scan: Lookback {request.lookback} might exceed typical fetch limits (~1500). Analysis will use max available.")
         # effective_lookback = 1500 # Cap here too if needed

    # --- Basic Setup Checks ---
    if binance_futures is None:
        logger.error("API Scan Abort: Exchange unavailable.")
        raise HTTPException(status_code=503, detail="Exchange unavailable")
    if not binance_futures.markets:
        logger.warning("API Scan: Markets not loaded, attempting load...")
        # Use await here as it's critical for the endpoint
        if not await load_exchange_markets(binance_futures):
             logger.error("API Scan Abort: Failed load markets.")
             raise HTTPException(status_code=503, detail="Failed load markets.")

    # --- 1. Get Tickers ---
    try:
        markets = binance_futures.markets
        if not markets: raise ValueError("Markets unavailable post-load.")
        # Filter for active USDT-settled perpetual futures
        all_tickers = sorted([
            m['symbol'] for m in markets.values()
            if m.get('swap') and m.get('linear') and m.get('quote')=='USDT' and m.get('settle')=='USDT' and m.get('active')
        ])
        logger.info(f"API Scan: Found {len(all_tickers)} active USDT linear perpetuals.")
        if not all_tickers:
             # Return an empty valid response if no tickers found
             return ScanResponse(
                 scan_parameters=request, total_tickers_attempted=0, total_tickers_succeeded=0,
                 ticker_start_index=request.ticker_start_index, ticker_end_index=request.ticker_end_index,
                 total_opportunities_found=0, top_opportunities=[], errors={}
             )
    except Exception as e:
        logger.error(f"API Scan Tickers Error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Ticker retrieval error: {e}")

    # --- 2. Determine BTC Trend (Market Regime) ---
    btc_trend_state = "UNKNOWN"
    apply_btc_filter = request.filter_by_btc_trend
    if apply_btc_filter:
        logger.info(f"API Scan: BTC Trend filter enabled. Fetching BTC data for timeframe {request.timeframe}...")
        try:
            btc_symbol = "BTC/USDT:USDT" # Ensure this is the correct symbol format for your exchange
            # Ensure enough lookback for SMAs on BTC, respecting limits
            btc_lookback = max(250, effective_lookback) # Use the potentially capped lookback

            # Fetch BTC data using the utility function
            df_btc_raw = await get_real_time_data(btc_symbol, request.timeframe, btc_lookback)

            # Check if enough data was returned
            min_btc_bars = 205 # Conservative minimum for SMA200 calculation
            if df_btc_raw is None or df_btc_raw.empty or len(df_btc_raw) < min_btc_bars:
                 data_len = len(df_btc_raw) if df_btc_raw is not None else 0
                 logger.warning(f"API Scan: Insufficient BTC data ({data_len} bars < {min_btc_bars}) for trend on {request.timeframe}. Disabling BTC filter.")
                 apply_btc_filter = False
                 btc_trend_state = "UNKNOWN (Insufficient Data)"
            else:
                # Assign symbol for logging within indicator function if needed
                df_btc_raw['symbol'] = btc_symbol
                # Calculate indicators for BTC using the utility function
                df_btc_indicators = await asyncio.to_thread(apply_technical_indicators, df_btc_raw)

                if df_btc_indicators is None or df_btc_indicators.empty:
                     logger.warning(f"API Scan: BTC indicator calculation resulted in empty DataFrame. Disabling BTC filter.")
                     apply_btc_filter = False
                     btc_trend_state = "UNKNOWN (Indicator Calc Failed)"
                else:
                    btc_latest = df_btc_indicators.iloc[-1]
                    # Safely get values, checking for NaN/None
                    btc_price = btc_latest.get('close')
                    btc_sma50 = btc_latest.get('SMA_50')
                    btc_sma200 = btc_latest.get('SMA_200')

                    # Check if all needed values are valid numbers
                    if all(v is not None and np.isfinite(v) for v in [btc_price, btc_sma50, btc_sma200]):
                        if btc_price > btc_sma50 > btc_sma200:
                            btc_trend_state = "UPTREND"
                        elif btc_price < btc_sma50 < btc_sma200:
                            btc_trend_state = "DOWNTREND"
                        else:
                            btc_trend_state = "CHOPPY"
                        logger.info(f"API Scan: Determined BTC Trend ({request.timeframe}): {btc_trend_state} (P: {btc_price:.2f}, S50: {btc_sma50:.2f}, S200: {btc_sma200:.2f})")
                    else:
                        logger.warning(f"API Scan: Could not determine BTC trend (Price/SMA50/SMA200 is NaN or None). Disabling BTC filter.")
                        apply_btc_filter = False
                        btc_trend_state = "UNKNOWN (Indicator NaN)"

        except Exception as e:
             logger.error(f"API Scan: Error getting BTC trend: {e}. Disabling BTC filter.", exc_info=True)
             apply_btc_filter = False
             btc_trend_state = "ERROR"
    else:
         logger.info("API Scan: BTC Trend filter is disabled via request.")


    # --- 3. Select Tickers ---
    tickers_to_scan: List[str] = []
    total_available = len(all_tickers)
    start_index = request.ticker_start_index or 0
    end_index = request.ticker_end_index # Can be None
    actual_end_index_for_response: Optional[int] = None
    slice_desc = ""

    # Logic for slicing tickers based on start/end/max
    if start_index >= total_available > 0:
        logger.warning(f"API Scan: Start index {start_index} is out of bounds ({total_available} available). No tickers to scan.")
        # Return empty response if start index invalid
        return ScanResponse(scan_parameters=request, total_tickers_attempted=0, total_tickers_succeeded=0, ticker_start_index=start_index, ticker_end_index=end_index, total_opportunities_found=0, top_opportunities=[], errors={"config": f"Start index {start_index} out of bounds."})

    if start_index < 0: start_index = 0 # Handle negative start index

    if end_index is not None:
        # Use end_index if provided
        if end_index <= start_index:
            tickers_to_scan = []
            slice_desc = f"invalid slice requested [{start_index}:{end_index}]"
            actual_end_index_for_response = end_index # Report requested end index
        else:
            actual_end = min(end_index, total_available) # Slice up to end_index or max available
            tickers_to_scan = all_tickers[start_index:actual_end]
            slice_desc = f"requested slice [{start_index}:{end_index}], actual [{start_index}:{actual_end}]"
            actual_end_index_for_response = end_index # Report requested end index
    elif request.max_tickers is not None and request.max_tickers > 0:
        # Use max_tickers if end_index is not provided
        limit = request.max_tickers
        actual_end = min(start_index + limit, total_available) # Slice up to max_tickers limit or max available
        tickers_to_scan = all_tickers[start_index:actual_end]
        slice_desc = f"max_tickers={limit} from {start_index}, actual [{start_index}:{actual_end}]"
        actual_end_index_for_response = actual_end # Report actual end index used
    elif request.max_tickers == 0:
        # Handle max_tickers = 0 explicitly
        tickers_to_scan = []
        slice_desc = "max_tickers=0"
        actual_end_index_for_response = start_index # End index is same as start
    else:
        # Default: Scan all tickers from start_index if no end_index or max_tickers provided
        actual_end = total_available
        tickers_to_scan = all_tickers[start_index:actual_end]
        slice_desc = f"all from {start_index}, actual [{start_index}:{actual_end}]"
        actual_end_index_for_response = actual_end # Report actual end index used

    logger.info(f"API Scan: Selected {len(tickers_to_scan)} tickers to analyze ({slice_desc}).")
    total_attempted = len(tickers_to_scan)
    if total_attempted == 0:
        # Return empty if slicing resulted in no tickers
        return ScanResponse(scan_parameters=request, total_tickers_attempted=0, total_tickers_succeeded=0, ticker_start_index=start_index, ticker_end_index=actual_end_index_for_response, total_opportunities_found=0, top_opportunities=[], errors={})


    # --- 4. Run Concurrently ---
    # Use max_concurrent_tasks from request, fallback to default from settings if needed
    # Ensure settings is available before accessing
    default_concurrency = settings.DEFAULT_MAX_CONCURRENT_TASKS if settings else 10
    max_concurrency = request.max_concurrent_tasks or default_concurrency
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = []
    processed_count = 0
    progress_lock = asyncio.Lock() # Lock for updating shared counter
    log_interval = max(1, total_attempted // 20 if total_attempted > 0 else 1) # Log progress roughly every 5%

    # Wrapper function to manage semaphore and capture results/errors
    async def analyze_with_semaphore_wrapper(ticker: str) -> Any: # Return Any to handle exceptions
        nonlocal processed_count
        result = None
        task_start_time = time.time()
        async with semaphore:
            try:
                # logger.debug(f"Starting analysis for {ticker}...")
                # Call the core analysis function from the service layer
                # Ensure perform_single_analysis is imported correctly
                result = await perform_single_analysis(
                    symbol=ticker,
                    timeframe=request.timeframe,
                    lookback=request.lookback, # Pass original request, fetch func handles limits
                    account_balance=request.accountBalance,
                    max_leverage=request.maxLeverage,
                    # Pass filter values down to analysis and GPT prompt if needed
                    min_requested_rr=request.min_risk_reward_ratio,
                    min_adx_for_trigger=request.min_adx # Pass ADX threshold used for trigger
                )
                # logger.debug(f"Finished analysis for {ticker}.")
                return result
            except Exception as e:
                # Log and return the exception itself to be handled by the gather block
                logger.error(f"API Scan: Unhandled Exception in analyze_with_semaphore_wrapper for {ticker}: {e}", exc_info=True)
                return e # Return exception to be caught by gather
            finally:
                 task_duration = time.time() - task_start_time
                 # Safely update and log progress
                 async with progress_lock:
                     processed_count += 1
                     current_count = processed_count
                 # Log progress periodically
                 if current_count % log_interval == 0 or current_count == total_attempted:
                     logger.info(f"API Scan Progress: {current_count}/{total_attempted} tasks completed (Last: {ticker} took {task_duration:.2f}s).")

    logger.info(f"API Scan: Creating {total_attempted} analysis tasks...")
    # Create tasks using the wrapper
    tasks = [analyze_with_semaphore_wrapper(ticker) for ticker in tickers_to_scan]

    logger.info(f"API Scan: Gathering results for {total_attempted} tasks (Concurrency: {max_concurrency})...")
    # Use return_exceptions=True to get exceptions instead of raising them immediately
    analysis_results_raw: List[Any] = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info(f"API Scan: Finished gathering {len(analysis_results_raw)} task results.")


    # --- 5. Process Results & Apply Filters ---
    successful_analyses_count = 0 # Count analyses that ran without critical errors/exceptions
    analysis_errors: Dict[str, str] = {} # Store errors encountered per ticker
    opportunities_passing_filter: List[ScanResultItem] = [] # Store results that pass all filters
    logger.info("API Scan: Starting detailed processing and filtering of results...")

    for i, res_or_exc in enumerate(analysis_results_raw):
        # Get the symbol corresponding to this result index
        symbol = tickers_to_scan[i] if i < len(tickers_to_scan) else f"UnknownTicker_{i}"
        # logger.debug(f"--- Processing item {i+1}/{len(analysis_results_raw)} for symbol [{symbol}] ---")

        # --- Handle Results/Exceptions ---
        analysis_result: Optional[AnalysisResponse] = None
        processing_error: Optional[str] = None

        if isinstance(res_or_exc, Exception):
            # Exception occurred within the wrapper or gather itself
            logger.error(f"API Scan: Task [{symbol}] resulted in UNHANDLED Exception: {res_or_exc}", exc_info=False) # Log exception info if needed via True
            processing_error = f"Unhandled Task Exception: {str(res_or_exc)[:200]}"
            analysis_errors[symbol] = processing_error

        elif isinstance(res_or_exc, AnalysisResponse):
            analysis_result = res_or_exc
            # Check for errors reported *within* the AnalysisResponse object
            if analysis_result.error:
                 # Distinguish skips/warnings from critical errors
                 # Improved check for skip conditions
                 is_skip = any(skip_text in analysis_result.error for skip_text in ["Insufficient data", "skipped:", "Warning:", "Cannot determine technical signal"])
                 is_critical_error = not is_skip

                 if is_critical_error:
                      logger.error(f"API Scan: Critical error reported for [{analysis_result.symbol}]: {analysis_result.error}")
                      processing_error = f"Analysis Critical Error: {analysis_result.error}"
                      analysis_errors[analysis_result.symbol] = processing_error
                 else:
                      # Log skips but don't count as error, proceed to next ticker
                      logger.info(f"API Scan: Skipping filter check for [{analysis_result.symbol}] due to non-critical issue: {analysis_result.error}")
                      successful_analyses_count += 1 # Count as attempted and processed without critical failure
                      continue # Skip filtering for this ticker
            else:
                 # No error reported in the result object
                 successful_analyses_count += 1
        else:
            # Unexpected type returned from gather
            logger.error(f"API Scan: UNEXPECTED result type for [{symbol}]: {type(res_or_exc)}. Value: {str(res_or_exc)[:100]}")
            processing_error = f"Unexpected Result Type: {type(res_or_exc).__name__}"
            analysis_errors[symbol] = processing_error

        # If a critical processing error occurred above, skip filtering
        if processing_error:
            continue

        # --- Proceed with Filtering if analysis_result is valid ---
        if analysis_result is None: # Should not happen if logic above is correct, but defensive check
             logger.error(f"API Scan: Analysis result object is None for symbol {symbol} despite no error. Skipping.")
             analysis_errors[symbol] = "Internal processing error: analysis_result was None."
             continue

        # --- Extract data safely for filtering ---
        gpt_params = analysis_result.gptParams
        bt_results = analysis_result.backtest
        indicators = analysis_result.indicators
        current_price = analysis_result.currentPrice

        # Safely access nested attributes
        bt_analysis = bt_results.trade_analysis if bt_results and bt_results.trade_analysis else None
        eval_direction = gpt_params.trade_direction if gpt_params else None # Use GPT's final evaluated direction
        gpt_conf = gpt_params.confidence_score if gpt_params and gpt_params.confidence_score is not None else None
        bt_score = bt_results.strategy_score if bt_results and bt_results.strategy_score is not None else None
        adx_value = indicators.ADX if indicators and indicators.ADX is not None else None
        sma50_value = indicators.SMA_50 if indicators and indicators.SMA_50 is not None else None
        sma200_value = indicators.SMA_200 if indicators and indicators.SMA_200 is not None else None


        # Log pre-filter state (Optional: reduce verbosity)
        # log_pre_filter = (...)
        # logger.info(log_pre_filter)


        # --- Filtering Logic ---
        passes_filters = True
        filter_fail_reason = [] # Collect all reasons for failure

        # F1: Basic Direction & Confidence/Score Check
        if eval_direction not in ['long', 'short']:
            passes_filters = False; filter_fail_reason.append(f"Eval Direction not tradeable ('{eval_direction}')")
        if passes_filters and request.trade_direction and eval_direction != request.trade_direction: # Check only if direction is tradeable
            passes_filters = False; filter_fail_reason.append(f"Direction mismatch (Req: {request.trade_direction}, Eval: {eval_direction})")
        # Check confidence AFTER direction check
        if passes_filters and (gpt_conf is None or gpt_conf < request.min_gpt_confidence):
            passes_filters = False; filter_fail_reason.append(f"GPT Conf too low (Req >= {request.min_gpt_confidence:.2f}, Got: {f'{gpt_conf:.2f}' if gpt_conf is not None else 'N/A'})")
        # Check backtest score
        if passes_filters and (bt_score is None or bt_score < request.min_backtest_score):
             passes_filters = False; filter_fail_reason.append(f"BT Score too low (Req >= {request.min_backtest_score:.2f}, Got: {f'{bt_score:.2f}' if bt_score is not None else 'N/A'})")

        # F2: Backtest Stats (Only if F1 passed and required)
        if passes_filters: # Check only if previous filters passed
            if bt_analysis is None: # Check if backtest analysis exists first
                # Decide if missing backtest analysis is a failure condition
                # If filters require backtest stats, then yes.
                if request.min_backtest_trades or request.min_backtest_win_rate or request.min_backtest_profit_factor:
                    passes_filters = False; filter_fail_reason.append("Backtest analysis missing, but required by filters")
            else: # Only check stats if analysis exists
                if request.min_backtest_trades is not None and (bt_analysis.total_trades < request.min_backtest_trades):
                    passes_filters = False; filter_fail_reason.append(f"BT Trades too low (Req >= {request.min_backtest_trades}, Got: {bt_analysis.total_trades})")
                if request.min_backtest_win_rate is not None and (bt_analysis.win_rate is None or bt_analysis.win_rate < request.min_backtest_win_rate):
                    passes_filters = False; filter_fail_reason.append(f"BT Win Rate too low (Req >= {request.min_backtest_win_rate:.1%}, Got: {f'{bt_analysis.win_rate:.1%}' if bt_analysis.win_rate is not None else 'N/A'})")
                # Handle infinite profit factor correctly
                pf_val = bt_analysis.profit_factor
                if request.min_backtest_profit_factor is not None and (pf_val is None or (pf_val != float('inf') and pf_val < request.min_backtest_profit_factor)):
                     pf_str = 'inf' if pf_val == float('inf') else f'{pf_val:.2f}' if pf_val is not None else 'N/A'
                     passes_filters = False; filter_fail_reason.append(f"BT Profit Factor too low (Req >= {request.min_backtest_profit_factor:.2f}, Got: {pf_str})")

        # F3: Risk/Reward Ratio (Only if F1, F2 passed and required)
        if passes_filters and request.min_risk_reward_ratio is not None and request.min_risk_reward_ratio > 0:
            rr_ratio = None
            # Need gpt_params for SL/TP/Entry
            if gpt_params and gpt_params.optimal_entry is not None and gpt_params.stop_loss is not None and gpt_params.take_profit is not None:
                 entry = gpt_params.optimal_entry
                 sl = gpt_params.stop_loss
                 tp = gpt_params.take_profit
                 # Ensure levels are logical before calculating R/R
                 if (eval_direction == 'long' and sl < entry < tp) or \
                    (eval_direction == 'short' and tp < entry < sl):
                     risk = abs(entry - sl)
                     reward = abs(tp - entry)
                     if risk > 1e-9: rr_ratio = reward / risk
                     # else: rr_ratio remains None if risk is zero

            if rr_ratio is None or rr_ratio < request.min_risk_reward_ratio:
                 passes_filters = False; filter_fail_reason.append(f"R/R Ratio too low (Req >= {request.min_risk_reward_ratio:.2f}, Got: {f'{rr_ratio:.2f}' if rr_ratio is not None else 'N/A or Invalid Levels'})")

        # F4: Indicator Filters (ADX, SMA Alignment) (Only if F1-F3 passed)
        # Note: ADX filter might be implicitly applied by the signal generation logic already
        if passes_filters:
            # Check ADX value directly if needed (double check)
            if request.min_adx is not None and request.min_adx > 0: # Use the request min_adx value
                 if adx_value is None or adx_value < request.min_adx:
                     passes_filters = False; filter_fail_reason.append(f"ADX too low (Req >= {request.min_adx:.1f}, Got: {f'{adx_value:.1f}' if adx_value is not None else 'N/A'})")

            # Check SMA alignment if required
            if passes_filters and request.require_sma_alignment:
                sma_aligned = False
                # Check necessary values exist and are finite
                if all(v is not None and np.isfinite(v) for v in [current_price, sma50_value, sma200_value]):
                    if eval_direction == 'long' and current_price > sma50_value > sma200_value:
                        sma_aligned = True
                    elif eval_direction == 'short' and current_price < sma50_value < sma200_value:
                        sma_aligned = True

                if not sma_aligned:
                    # Format price/sma values for clearer log message
                    price_str = f"{current_price:.4f}" if current_price is not None and np.isfinite(current_price) else "N/A"
                    sma50_str = f"{sma50_value:.4f}" if sma50_value is not None and np.isfinite(sma50_value) else "N/A"
                    sma200_str = f"{sma200_value:.4f}" if sma200_value is not None and np.isfinite(sma200_value) else "N/A"
                    passes_filters = False; filter_fail_reason.append(f"SMA alignment failed (Dir: {eval_direction}, P:{price_str}, S50:{sma50_str}, S200:{sma200_str})")


        # F5: BTC Trend Alignment (Only if F1-F4 passed and filter enabled)
        # Define valid trend states that allow filtering
        valid_btc_trend_states = ["UPTREND", "DOWNTREND", "CHOPPY"]
        if passes_filters and apply_btc_filter and btc_trend_state in valid_btc_trend_states:
            # --- Choose ONE of the following filter logics ---
            # OPTION A: Strict Trend Following (Current code's default)
            if eval_direction == 'long' and btc_trend_state != 'UPTREND':
                passes_filters = False; filter_fail_reason.append(f"BTC Trend filter: Long blocked (BTC is {btc_trend_state})")
            elif eval_direction == 'short' and btc_trend_state != 'DOWNTREND':
                passes_filters = False; filter_fail_reason.append(f"BTC Trend filter: Short blocked (BTC is {btc_trend_state})")

            # OPTION B: Less Strict (Allow trades if BTC is Choppy)
            # if eval_direction == 'long' and btc_trend_state == 'DOWNTREND':
            #     passes_filters = False; filter_fail_reason.append(f"BTC Trend filter: Long blocked (BTC is DOWNTREND)")
            # elif eval_direction == 'short' and btc_trend_state == 'UPTREND':
            #     passes_filters = False; filter_fail_reason.append(f"BTC Trend filter: Short blocked (BTC is UPTREND)")
            # --- End Filter Logic Choice ---


        # --- Add Opportunity if Passes All Filters ---
        if passes_filters:
            logger.info(f"[{analysis_result.symbol}] PASSED ALL FILTERS. Adding to opportunities.")
            # Calculate combined score (handle None values)
            score_g = float(gpt_conf) if gpt_conf is not None else 0.0
            score_b = float(bt_score) if bt_score is not None else 0.0
            # Weighting: GPT Conf (prediction quality) 60%, BT Score (historical signal performance) 40%
            combined_score = round((score_g * 0.6) + (score_b * 0.4), 3)

            # Extract a concise summary from GPT analysis
            summary = "Analysis details unavailable."
            if analysis_result.gptAnalysis:
                # Prioritize signal_evaluation, then technical_analysis
                primary_summary = analysis_result.gptAnalysis.signal_evaluation or analysis_result.gptAnalysis.technical_analysis
                if primary_summary and isinstance(primary_summary, str) and "Error:" not in primary_summary:
                     # Try to get the first sentence
                     summary_parts = primary_summary.split('.')
                     summary = (summary_parts[0] + '.').strip() if summary_parts else primary_summary
                elif analysis_result.gptAnalysis.raw_text and isinstance(analysis_result.gptAnalysis.raw_text, str):
                     # Fallback to raw text's first sentence if analysis is missing
                     # Be careful with raw_text, it might contain JSON or errors
                     if "Error:" not in analysis_result.gptAnalysis.raw_text and "{" not in analysis_result.gptAnalysis.raw_text:
                        summary_parts = analysis_result.gptAnalysis.raw_text.split('.')
                        summary = (summary_parts[0] + '.').strip() if summary_parts else "Raw analysis text available."

                # Truncate summary if too long
                summary = summary[:147] + "..." if len(summary) > 150 else summary

            # Create the result item
            opportunity = ScanResultItem(
                rank=0, # Rank assigned after sorting
                symbol=analysis_result.symbol,
                timeframe=analysis_result.timeframe,
                currentPrice=current_price,
                gptConfidence=gpt_conf,
                backtestScore=bt_score,
                combinedScore=combined_score,
                tradeDirection=eval_direction, # Use the evaluated direction
                optimalEntry=gpt_params.optimal_entry if gpt_params else None,
                stopLoss=gpt_params.stop_loss if gpt_params else None,
                takeProfit=gpt_params.take_profit if gpt_params else None,
                gptAnalysisSummary=summary
            )
            opportunities_passing_filter.append(opportunity)
        else:
             # Log the failure reason(s) only if it wasn't a planned skip/error handled earlier
             # Check if analysis_result exists before accessing symbol
             log_symbol = analysis_result.symbol if analysis_result else symbol
             if not processing_error and analysis_result and not analysis_result.error: # Check if failure was due to filters
                 logger.info(f"[{log_symbol}] FAILED FILTERS. Reasons: {'; '.join(filter_fail_reason)}.")

    
    # --- End of Result Processing Loop ---
    logger.info(f"API Scan: Finished processing results. Succeeded/Filterable: {successful_analyses_count}, Errors: {len(analysis_errors)}, Passed Filters: {len(opportunities_passing_filter)}.")


    # --- 6. Rank Opportunities ---
    if opportunities_passing_filter:
        logger.info(f"API Scan: Sorting {len(opportunities_passing_filter)} opportunities by Combined Score...")
        # Sort by combined score descending, handle potential None scores safely
        opportunities_passing_filter.sort(key=lambda x: x.combinedScore if x.combinedScore is not None else 0.0, reverse=True)

        # Assign ranks and select top N
        top_opportunities = []
        for rank_idx, opp in enumerate(opportunities_passing_filter[:request.top_n]):
            opp.rank = rank_idx + 1
            top_opportunities.append(opp)
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
        ticker_end_index=actual_end_index_for_response, # Report the actual/requested end index used
        total_opportunities_found=len(opportunities_passing_filter),
        top_opportunities=top_opportunities,
        errors=analysis_errors
    )

# --- End of scan_market_endpoint function ---