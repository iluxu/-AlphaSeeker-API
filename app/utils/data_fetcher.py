# Data fetching logic
# Example: app/utils/data_fetcher.py
import pandas as pd
import ccxt
import asyncio
import logging
# Import binance_futures from clients if needed directly,
# or pass it as an argument from the service layer
from app.core.clients import binance_futures

logger = logging.getLogger(__name__)

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