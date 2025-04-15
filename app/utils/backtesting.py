# app/utils/backtesting.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

# Import models if using them for type hinting (optional but good practice)
# from app.models.analysis import BacktestTradeAnalysis # Example if needed

# Get a logger for this module
# Assumes logging is configured elsewhere (e.g., in config.py or main entry point)
logger = logging.getLogger(__name__)

# --- Backtesting (Aligned with MACD/Trend/ADX Signal) ---
def backtest_strategy(
    df_with_indicators: pd.DataFrame,
    # Pass the *initial* technical direction determined before calling GPT
    initial_signal_direction: str, # 'long' or 'short' based on MACD/Trend/ADX check
    min_adx_for_trigger: float = 20.0, # Ensure this matches the trigger logic used
    min_rr_ratio_target: float = 1.5, # Target R:R for simulated trades
    atr_sl_multiplier: float = 1.5, # How many ATRs for stop loss from entry
    max_trade_duration_bars: int = 96, # e.g., 4 days on 1h bars (adjust as needed)
    min_bars_between_trades: int = 5 # Cooldown period to avoid rapid re-entry
) -> Dict[str, Any]:
    """Backtest strategy based on *historical occurrences* of the MACD/Trend/ADX signal."""
    # Extract symbol for logging if available in DataFrame columns
    symbol_log = df_with_indicators.get('symbol', ['Unknown'])[0] if 'symbol' in df_with_indicators.columns and not df_with_indicators.empty else 'Unknown'
    log_prefix = f"[{symbol_log} Backtest]"

    # Initialize results structure
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
    trade_analysis = results['trade_analysis'] # Shortcut for easier access

    # Backtest only makes sense if the initial signal was potentially tradeable
    if initial_signal_direction not in ['long', 'short']:
        results['recommendation'] = f"Backtest skipped: Initial signal direction is '{initial_signal_direction}'."
        logger.info(f"{log_prefix} {results['recommendation']}")
        return results

    # --- Requirements for the backtest logic ---
    # Columns needed for signal generation AND trade simulation (ATR for SL)
    required_cols = ['open', 'high', 'low', 'close', 'ATR', 'MACD', 'Signal_Line', 'SMA_200', 'ADX']

    if df_with_indicators.empty or not all(col in df_with_indicators.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_with_indicators.columns]
        results['recommendation'] = "Backtest skipped: Missing required columns."
        results['warnings'].append(f"Missing required columns for backtest: {missing}")
        logger.warning(f"{log_prefix} Backtest skipped: Missing required columns: {missing}")
        return results

    # Create a clean DataFrame for signal checking, handle potential numeric errors
    df_signal_check = df_with_indicators[required_cols].copy()
    for col in required_cols:
        df_signal_check[col] = pd.to_numeric(df_signal_check[col], errors='coerce')

    # Drop rows where ANY essential column for signal or simulation is missing/invalid
    df_signal_check.dropna(subset=required_cols, inplace=True)

    # --- Add previous MACD/Signal for crossover detection ---
    # Shift requires at least 2 rows after dropping NaNs
    if len(df_signal_check) >= 2:
        df_signal_check['prev_MACD'] = df_signal_check['MACD'].shift(1)
        df_signal_check['prev_Signal_Line'] = df_signal_check['Signal_Line'].shift(1)
        # Drop the first row where prev values are NaN
        df_signal_check.dropna(subset=['prev_MACD', 'prev_Signal_Line'], inplace=True)
    else:
        # Not enough data even for shift operation
        df_signal_check = pd.DataFrame(columns=df_signal_check.columns.tolist() + ['prev_MACD', 'prev_Signal_Line']) # Empty df with cols


    if len(df_signal_check) < 50: # Need sufficient data for meaningful backtest after cleaning
        results['recommendation'] = "Backtest skipped: Insufficient clean data for simulation."
        results['warnings'].append(f"Insufficient clean data points ({len(df_signal_check)}) for backtest after cleaning.")
        logger.warning(f"{log_prefix} {results['recommendation']}")
        return results

    logger.info(f"{log_prefix} Starting backtest for initial signal '{initial_signal_direction}' over {len(df_signal_check)} potential signal bars.")

    trades: List[Dict[str, Any]] = [] # List to store trade results
    last_entry_iloc = -1 - min_bars_between_trades # Initialize to allow first trade

    # --- Simulation Loop (Iterate through cleaned signal data using iloc) ---
    entry_count = 0
    # Iterate up to second-to-last bar to allow entry on the *next* bar's open
    for i_loc in range(len(df_signal_check) - 1):
        signal_row = df_signal_check.iloc[i_loc]
        signal_timestamp = df_signal_check.index[i_loc] # Timestamp of the bar where signal conditions are met

        # --- Determine Entry Bar ---
        # Entry happens on the open of the *next* bar
        entry_bar_iloc = i_loc + 1
        entry_bar_timestamp = df_signal_check.index[entry_bar_iloc]

        # Get entry price (open) and high/low for simulation from the ORIGINAL DataFrame
        # using the timestamp to ensure we have the full data, not potentially modified data
        try:
            entry_bar_data_orig = df_with_indicators.loc[entry_bar_timestamp]
            entry_price = pd.to_numeric(entry_bar_data_orig['open'], errors='coerce')
            # Check if entry price is valid
            if pd.isna(entry_price):
                # logger.debug(f"{log_prefix} Skipping signal at {signal_timestamp}: Invalid entry price (open) at {entry_bar_timestamp}.")
                continue
        except KeyError:
            # logger.debug(f"{log_prefix} Skipping signal at {signal_timestamp}: Entry bar timestamp {entry_bar_timestamp} not found in original DataFrame.")
            continue # Skip if timestamp not found in original df (should be rare)


        # Check Cooldown based on the index location of the last *entry* bar
        if entry_bar_iloc <= last_entry_iloc + min_bars_between_trades:
             # logger.debug(f"{log_prefix} Skipping signal at {signal_timestamp}: Cooldown period active.")
             continue

        # --- Check for MACD/Trend/ADX Signal Conditions on the signal_row ---
        # Use values from the cleaned df_signal_check row
        price = signal_row['close']
        sma200 = signal_row['SMA_200']
        adx = signal_row['ADX']
        macd = signal_row['MACD']
        signal_line = signal_row['Signal_Line']
        prev_macd = signal_row['prev_MACD']
        prev_signal_line = signal_row['prev_Signal_Line']
        atr = signal_row['ATR'] # ATR from the signal bar for SL calculation

        # Check data validity again for this specific bar (already checked by dropna, but defensive)
        if pd.isna(price) or pd.isna(sma200) or pd.isna(adx) or pd.isna(macd) or \
           pd.isna(signal_line) or pd.isna(prev_macd) or pd.isna(prev_signal_line) or pd.isna(atr) or atr <= 1e-9:
            # logger.debug(f"{log_prefix} Skipping signal at {signal_timestamp}: Found NaN in critical signal check values.")
            continue # Skip bar if essential data is missing or ATR is non-positive

        # --- Apply the Signal Logic ---
        setup_found = False
        stop_loss_calc, take_profit_calc = 0.0, 0.0

        is_trending_up = price > sma200
        is_trending_down = price < sma200
        is_adx_strong = adx >= min_adx_for_trigger
        # Check for crossover: current MACD relative to signal, compared to previous state
        macd_crossed_up = (prev_macd <= prev_signal_line) and (macd > signal_line)
        macd_crossed_down = (prev_macd >= prev_signal_line) and (macd < signal_line)

        current_signal_matches_request = False
        trade_direction_for_setup = None
        if initial_signal_direction == 'long' and is_trending_up and macd_crossed_up and is_adx_strong:
            current_signal_matches_request = True
            trade_direction_for_setup = 'long'
        elif initial_signal_direction == 'short' and is_trending_down and macd_crossed_down and is_adx_strong:
            current_signal_matches_request = True
            trade_direction_for_setup = 'short'

        if current_signal_matches_request:
            # Calculate potential SL/TP based on *entry price* and signal bar's ATR
            risk_per_point = atr * atr_sl_multiplier
            if trade_direction_for_setup == 'long':
                stop_loss_calc = entry_price - risk_per_point
                take_profit_calc = entry_price + risk_per_point * min_rr_ratio_target
                # Validate levels make sense relative to entry price
                if stop_loss_calc < entry_price < take_profit_calc:
                    setup_found = True
                    # logger.debug(f"{log_prefix} Long setup found at {signal_timestamp} (MACD X Up, Trend Up, ADX Ok). Entry@{entry_bar_timestamp}: {entry_price:.4f}, SL: {stop_loss_calc:.4f}, TP: {take_profit_calc:.4f}, ATR: {atr:.4f}")
            elif trade_direction_for_setup == 'short':
                stop_loss_calc = entry_price + risk_per_point
                take_profit_calc = entry_price - risk_per_point * min_rr_ratio_target
                 # Validate levels make sense relative to entry price
                if take_profit_calc < entry_price < stop_loss_calc:
                    setup_found = True
                    # logger.debug(f"{log_prefix} Short setup found at {signal_timestamp} (MACD X Down, Trend Down, ADX Ok). Entry@{entry_bar_timestamp}: {entry_price:.4f}, SL: {stop_loss_calc:.4f}, TP: {take_profit_calc:.4f}, ATR: {atr:.4f}")
            # else: # Should not happen if direction is long/short
            #      pass

            if not setup_found and current_signal_matches_request:
                 logger.debug(f"{log_prefix} Setup found at {signal_timestamp} for {trade_direction_for_setup} but levels invalid (E:{entry_price:.4f}, SL:{stop_loss_calc:.4f}, TP:{take_profit_calc:.4f}) ATR: {atr:.4f}")


        # --- Simulate Trade if setup found ---
        if setup_found:
            entry_count += 1
            outcome = None
            exit_price = None
            exit_bar_iloc = -1 # iloc in df_signal_check where trade exits

            # Determine the maximum bar index to check for this trade's exit
            # Ensure exit simulation stays within the bounds of the df_signal_check
            max_exit_iloc = min(entry_bar_iloc + max_trade_duration_bars, len(df_signal_check) - 1)

            # Simulate bar-by-bar from entry bar onwards
            for k_loc in range(entry_bar_iloc, max_exit_iloc + 1):
                current_sim_timestamp = df_signal_check.index[k_loc]
                # Get High/Low from the original full df_with_indicators for accurate checks
                try:
                    current_bar_orig = df_with_indicators.loc[current_sim_timestamp]
                    current_low = pd.to_numeric(current_bar_orig['low'], errors='coerce')
                    current_high = pd.to_numeric(current_bar_orig['high'], errors='coerce')
                    if pd.isna(current_low) or pd.isna(current_high):
                        # logger.warning(f"{log_prefix} Trade {entry_count}: Invalid High/Low at {current_sim_timestamp}. Skipping bar for exit check.")
                        continue # Skip bar if H/L invalid
                except KeyError:
                    # logger.warning(f"{log_prefix} Trade {entry_count}: Timestamp {current_sim_timestamp} not found in original df during simulation. Skipping bar.")
                    continue # Skip if TS not in original df

                # Check for SL/TP hit
                if trade_direction_for_setup == 'long':
                    if current_low <= stop_loss_calc: # SL Hit
                        outcome, exit_price, exit_bar_iloc = 'loss', stop_loss_calc, k_loc
                        # logger.debug(f"{log_prefix} Trade {entry_count} (Long@{entry_price:.4f}) SL hit at {exit_price:.4f} on bar {current_sim_timestamp}")
                        break
                    elif current_high >= take_profit_calc: # TP Hit
                        outcome, exit_price, exit_bar_iloc = 'win', take_profit_calc, k_loc
                        # logger.debug(f"{log_prefix} Trade {entry_count} (Long@{entry_price:.4f}) TP hit at {exit_price:.4f} on bar {current_sim_timestamp}")
                        break
                elif trade_direction_for_setup == 'short':
                    if current_high >= stop_loss_calc: # SL Hit
                        outcome, exit_price, exit_bar_iloc = 'loss', stop_loss_calc, k_loc
                        # logger.debug(f"{log_prefix} Trade {entry_count} (Short@{entry_price:.4f}) SL hit at {exit_price:.4f} on bar {current_sim_timestamp}")
                        break
                    elif current_low <= take_profit_calc: # TP Hit
                        outcome, exit_price, exit_bar_iloc = 'win', take_profit_calc, k_loc
                        # logger.debug(f"{log_prefix} Trade {entry_count} (Short@{entry_price:.4f}) TP hit at {exit_price:.4f} on bar {current_sim_timestamp}")
                        break

            # If neither SL nor TP hit within duration, exit at close of the last bar checked
            if outcome is None:
                exit_bar_iloc = max_exit_iloc # Exit on the last simulated bar
                exit_bar_timestamp = df_signal_check.index[exit_bar_iloc]
                try:
                    exit_price_close = pd.to_numeric(df_with_indicators.loc[exit_bar_timestamp, 'close'], errors='coerce')
                    if pd.isna(exit_price_close): # Handle rare case exit price is NaN
                         outcome = 'error' # Mark as error if exit price is invalid
                         exit_price = entry_price # Assign entry price to avoid breaking calculations? Or None?
                         logger.warning(f"{log_prefix} Trade {entry_count} exited due to duration but close price was NaN at {exit_bar_timestamp}. Marked as error.")
                    else:
                         exit_price = exit_price_close
                         if trade_direction_for_setup == 'long':
                             outcome = 'win' if exit_price > entry_price else 'loss'
                         else: # Short
                             outcome = 'win' if exit_price < entry_price else 'loss'
                         # logger.debug(f"{log_prefix} Trade {entry_count} ({trade_direction_for_setup}@{entry_price:.4f}) exited due to duration at {exit_price:.4f} on bar {exit_bar_timestamp}. Outcome: {outcome}")

                except KeyError:
                    outcome = 'error' # Mark as error if exit bar data not found
                    exit_price = entry_price
                    logger.error(f"{log_prefix} Trade {entry_count} exit bar TS {exit_bar_timestamp} not found in original df. Marked as error.")


            # --- Record the trade details ---
            if outcome and outcome != 'error' and exit_price is not None and exit_bar_iloc != -1:
                profit_points = (exit_price - entry_price) if trade_direction_for_setup == 'long' else (entry_price - exit_price)
                # Duration in number of bars (exit bar index - entry bar index)
                trade_duration_bars = exit_bar_iloc - entry_bar_iloc + 1 # +1 because indices are inclusive start/end bar count

                trades.append({
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'stop_loss': stop_loss_calc,
                    'take_profit': take_profit_calc,
                    'direction': trade_direction_for_setup,
                    'profit': profit_points,
                    'duration': trade_duration_bars, # Duration in bars
                    'outcome': outcome,
                    'entry_timestamp': entry_bar_timestamp,
                    'exit_timestamp': df_signal_check.index[exit_bar_iloc]
                })
                last_entry_iloc = entry_bar_iloc # Update cooldown based on this entry's bar index


    # --- Analyze Backtest Results ---
    trade_analysis['total_trades'] = len(trades)
    logger.info(f"{log_prefix} Backtest simulation completed. Found {len(trades)} historical trades matching '{initial_signal_direction}' signal criteria.")

    if trades:
        # Extract results into numpy arrays for efficiency
        profits = np.array([t['profit'] for t in trades if t.get('profit') is not None])
        durations = np.array([t['duration'] for t in trades if t.get('duration') is not None])
        outcomes = [t['outcome'] for t in trades if t.get('outcome') is not None]

        # Calculate metrics only if there are valid profit numbers
        if len(profits) > 0:
            trade_analysis['winning_trades'] = sum(1 for o in outcomes if o == 'win')
            trade_analysis['losing_trades'] = len(outcomes) - trade_analysis['winning_trades'] # Assumes outcomes are only 'win' or 'loss'
            if len(outcomes) > 0:
                trade_analysis['win_rate'] = round(trade_analysis['winning_trades'] / len(outcomes), 3)

            winning_profits = profits[profits > 0]
            losing_profits = profits[profits <= 0] # Includes zero profit trades as losses for PF calc
            gross_profit = np.sum(winning_profits)
            gross_loss = abs(np.sum(losing_profits)) # Absolute value of sum of losses/zeros

            trade_analysis['avg_profit'] = round(np.mean(winning_profits), 6) if len(winning_profits) > 0 else 0.0
            trade_analysis['avg_loss'] = round(abs(np.mean(losing_profits)), 6) if len(losing_profits) > 0 else 0.0

            # Calculate Profit Factor
            if gross_loss > 1e-9: # Avoid division by zero
                trade_analysis['profit_factor'] = round(gross_profit / gross_loss, 2)
            elif gross_profit > 1e-9: # Handle case of no losses
                trade_analysis['profit_factor'] = float('inf')
            else: # No profits and no losses (or only zero-profit trades)
                trade_analysis['profit_factor'] = 0.0

            trade_analysis['total_profit'] = round(np.sum(profits), 6) # Sum of all profits/losses
            trade_analysis['largest_win'] = round(np.max(winning_profits), 6) if len(winning_profits) > 0 else 0.0
            # Largest loss is the minimum value in losing_profits (most negative)
            trade_analysis['largest_loss'] = round(np.min(losing_profits), 6) if len(losing_profits) > 0 else 0.0

            # Calculate average duration if available
            if len(durations) > 0:
                 trade_analysis['average_trade_duration'] = round(np.mean(durations), 1)

            # --- Calculate Strategy Score (Based on historical performance of the raw signal) ---
            score = 0.0
            pf = trade_analysis['profit_factor']
            wr = trade_analysis['win_rate']
            num_trades = trade_analysis['total_trades'] # Use total trades from analysis dict
            avg_w = trade_analysis['avg_profit']
            avg_l = trade_analysis['avg_loss']

            # Score Weights (Adjust as needed)
            # Profit Factor Component (Weight ~0.35)
            if pf is not None:
                if pf == float('inf') or pf >= 2.5: score += 0.35
                elif pf >= 1.7: score += 0.25
                elif pf >= 1.3: score += 0.15
                elif pf >= 1.0: score += 0.05
                else: score -= 0.25 # Penalize losing strategies

            # Win Rate Component (Weight ~0.30)
            if wr is not None:
                if wr >= 0.60: score += 0.30
                elif wr >= 0.50: score += 0.20
                elif wr >= 0.40: score += 0.10
                else: score -= 0.15 # Penalize low win rate

            # Number of Trades Component (Weight ~0.15) - Reliability indicator
            if num_trades >= 30: score += 0.15
            elif num_trades >= 15: score += 0.10
            elif num_trades < 5: score -= 0.20 # Penalize very low sample size heavily

            # Avg Win/Loss Ratio Component (Weight ~0.10) - Risk/Reward indicator
            # Ensure avg_l is not zero to avoid division error
            if avg_w is not None and avg_l is not None and avg_l > 1e-9:
                 ratio = avg_w / avg_l
                 if ratio >= 2.0: score += 0.10
                 elif ratio >= 1.2: score += 0.05

            # Overall Profitability Penalty (Weight ~0.30)
            if trade_analysis['total_profit'] is not None and trade_analysis['total_profit'] <= 0 and num_trades > 0:
                 score -= 0.30 # Strong penalty for net loss

            # Clamp score between 0.0 and 1.0
            results['strategy_score'] = max(0.0, min(1.0, round(score, 3)))
            logger.info(f"{log_prefix} Backtest Score: {results['strategy_score']:.3f} (Trades:{num_trades}, WR:{wr*100 if wr else 'N/A':.1f}%, PF:{pf if pf else 'N/A':.2f})")

            # --- Recommendation and Warnings ---
            strategy_name = "MACD/Trend/ADX Crossover" # Descriptive name
            score_val = results['strategy_score']
            if score_val >= 0.70: results['recommendation'] = f"Strong historical performance for {strategy_name} signal."
            elif score_val >= 0.55: results['recommendation'] = f"Good historical performance for {strategy_name} signal."
            elif score_val >= 0.40: results['recommendation'] = f"Moderate/Mixed historical performance for {strategy_name} signal."
            else: results['recommendation'] = f"Poor historical performance for {strategy_name} signal."

            # Add specific warnings based on thresholds
            if wr is not None and wr < 0.45: results['warnings'].append(f"Low Win Rate ({wr:.1%})")
            if pf is not None and pf < 1.25 and pf != float('inf'): results['warnings'].append(f"Low Profit Factor ({pf:.2f})")
            if num_trades < 10: results['warnings'].append(f"Low Trade Count ({num_trades}), results may not be statistically significant.")
            if trade_analysis['total_profit'] is not None and trade_analysis['total_profit'] <= 0 and num_trades > 0: results['warnings'].append("Strategy resulted in an overall loss during the backtest period.")
            # Check for large drawdowns relative to average loss
            if avg_l is not None and avg_l > 1e-9 and trade_analysis['largest_loss'] is not None:
                if abs(trade_analysis['largest_loss']) > 3 * avg_l:
                    results['warnings'].append("Risk Warning: Largest loss significantly exceeded the average loss.")

        else: # No valid profit numbers found (e.g., all trades were errors)
             results['recommendation'] = f"Backtest ran but no valid trade outcomes recorded for {strategy_name}."
             results['warnings'].append("No valid trades with profit/loss data found.")
             results['strategy_score'] = 0.0


    else: # No trades found at all
        results['recommendation'] = f"No historical {strategy_name} setups found matching criteria in the provided data."
        results['warnings'].append("No qualifying historical trade setups found.")
        results['strategy_score'] = 0.0
        logger.info(f"{log_prefix} {results['recommendation']}")

    # Log final recommendation and warnings
    if results['recommendation'] != 'N/A': logger.info(f"{log_prefix} Recommendation: {results['recommendation']}")
    if results['warnings']: logger.warning(f"{log_prefix} Warnings: {'; '.join(results['warnings'])}")

    return results