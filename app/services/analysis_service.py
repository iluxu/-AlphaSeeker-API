# Core analysis service logic (perform_single_analysis)
# app/services/analysis_service.py
import time
import asyncio
import pandas as pd
import numpy as np
import logging
from typing import Optional

# Import necessary models
from app.models.analysis import AnalysisResponse, IndicatorsData, ModelOutputData, GptTradingParams, GptAnalysisText, BacktestResultsData, BacktestTradeAnalysis
from app.models.scan import ScanRequest # Might need ScanRequest defaults here

# Import utility functions
from app.utils.data_fetcher import get_real_time_data
from app.utils.indicators import apply_technical_indicators
from app.utils.stats_models import fit_garch_model, calculate_var
from app.utils.gpt_utils import gpt_generate_trading_parameters, parse_gpt_trading_parameters
from app.utils.backtesting import backtest_strategy

# Import clients if needed directly (or better: pass them if possible)
from app.core.clients import openai_client

logger = logging.getLogger(__name__)

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