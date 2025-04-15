# Tickers endpoint logic
# app/api/v1/endpoints/tickers.py

import logging
from fastapi import APIRouter, HTTPException

# Import models for request and response
try:
    # Assuming TickersResponse is defined in common models
    from app.models.common import TickersResponse
except ImportError as e:
    print(f"Error importing TickersResponse in tickers.py: {e}")
    # Define a placeholder or raise if essential
    TickersResponse = None # Placeholder

# Import core components (clients)
try:
    from app.core.clients import binance_futures, load_exchange_markets
except ImportError as e:
    print(f"Error importing core components in tickers.py: {e}")
    # Handle error or raise
    binance_futures = None
    load_exchange_markets = None

# Get a logger for this module
logger = logging.getLogger(__name__)

# --- DEFINE THE ROUTER INSTANCE --- <<< REQUIRED LINE
router = APIRouter()
# --- END ROUTER DEFINITION ---


# --- The Tickers Endpoint Function ---
@router.get("/tickers", response_model=TickersResponse, tags=["Utility"])
async def get_crypto_tickers_endpoint():
    # --- Start of the get_crypto_tickers_endpoint function from original script ---
    logger.info("API: Request received for /tickers")

    if binance_futures is None:
        logger.error("API Tickers: Exchange client (binance_futures) is None.")
        raise HTTPException(status_code=503, detail="Exchange unavailable")
    if load_exchange_markets is None:
        logger.error("API Tickers: load_exchange_markets function is None.")
        raise HTTPException(status_code=500, detail="Internal server error: Market loading function missing.")

    # Check if markets are loaded, attempt to load if necessary
    if not binance_futures.markets:
        logger.warning("API Tickers: Markets not loaded, attempting asynchronous load...")
        # Use await here as it's critical for the endpoint function
        if not await load_exchange_markets(binance_futures):
             logger.error("API Tickers: Failed to load markets.")
             raise HTTPException(status_code=503, detail="Failed to load exchange markets.")
        # Check again after load attempt
        if not binance_futures.markets:
             logger.error("API Tickers: Markets still not loaded after attempt.")
             raise HTTPException(status_code=503, detail="Markets unavailable after load attempt.")


    try:
        # Access markets after ensuring they are loaded
        markets = binance_futures.markets
        if not markets: # Should be loaded by now, but double-check
            logger.error("API Tickers: Markets dictionary is empty after load attempt.")
            raise HTTPException(status_code=500, detail="Markets loaded empty.")

        # Filter for active USDT-settled perpetual futures (same logic as original)
        tickers = sorted([
            m['symbol'] for m in markets.values()
            if m.get('swap')           # Is it a swap/future?
            and m.get('linear')        # Is it linear (vs inverse)?
            and m.get('quote')=='USDT' # Quote currency is USDT?
            and m.get('settle')=='USDT' # Settlement currency is USDT?
            and m.get('active')        # Is the market currently active?
        ])

        logger.info(f"API Tickers: Found {len(tickers)} active USDT linear perpetuals.")
        if not tickers:
            # It's valid to find no tickers, just log it.
            logger.warning("API Tickers: No active USDT linear perpetuals found matching criteria.")

        # Ensure TickersResponse model is available before returning
        if TickersResponse is None:
             raise HTTPException(status_code=500, detail="Internal server error: Response model not loaded.")

        return TickersResponse(tickers=tickers)

    except HTTPException as h:
         # Re-raise HTTP exceptions directly
         raise h
    except Exception as e:
        # Catch any other unexpected errors during processing
        logger.error(f"API Tickers Error during processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error getting tickers: {e}")
# --- End of get_crypto_tickers_endpoint function ---