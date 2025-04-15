# External client initializations (CCXT, OpenAI)
# app/core/clients.py
import ccxt
import ccxt.async_support as ccxt_async # For async versions if needed later
import openai
import asyncio
import logging
from app.core.config import settings # Import settings

logger = logging.getLogger(__name__)

# --- CCXT Initialization ---
binance_futures = None
try:
    # Using settings from config.py
    binance_futures = ccxt.binanceusdm({
        'enableRateLimit': True,
        'options': {'adjustForTimeDifference': True},
        'timeout': settings.CCXT_TIMEOUT,
        'rateLimit': settings.CCXT_RATE_LIMIT,
        # 'apiKey': settings.BINANCE_API_KEY, # Uncomment if needed
        # 'secret': settings.BINANCE_SECRET,  # Uncomment if needed
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
if not settings.OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable not found. GPT features will be disabled.")
else:
    try:
        openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info("OpenAI client initialized.")
    except Exception as e:
        logger.error(f"Error initializing OpenAI: {e}", exc_info=True)
        openai_client = None

# --- Test OpenAI Connection ---
async def test_openai_connection(client):
     # ... (Keep the function definition from the original script) ...
     pass # Replace pass with the actual function code