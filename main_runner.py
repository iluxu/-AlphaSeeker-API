# main_runner.py
import uvicorn
import os
import logging

# Import the FastAPI app instance and settings
# Ensure the app directory is in the Python path or use appropriate relative imports if running differently
try:
    from app.main import app
    from app.core.config import settings
except ImportError as e:
     print(f"Error importing app components: {e}")
     print("Ensure you are running this script from the project root directory")
     print("or that the 'app' directory is in your PYTHONPATH.")
     exit(1)


logger = logging.getLogger(__name__) # Use logger configured in config

if __name__ == "__main__":
    # Get host/port from environment or settings
    port = int(os.getenv("PORT", 8029)) # Default port
    host = os.getenv("HOST", "127.0.0.1") # Default host
    reload_flag = os.getenv("UVICORN_RELOAD", "false").lower() in ("true", "1", "yes")
    log_level = settings.LOG_LEVEL.lower() # Use log level from settings

    print("\n" + "="*30); print(" --- Starting FastAPI Server ---")
    print(f" Host: {host}"); print(f" Port: {port}"); print(f" Auto-Reload: {reload_flag}")
    print(f" Configured Logging Level: {settings.LOG_LEVEL}")
    # You can add checks here to see if clients initialized correctly if needed
    # print(f" OpenAI Client Initialized: {openai_client is not None}") # Requires importing client
    # print(f" CCXT Client Initialized: {binance_futures is not None}") # Requires importing client
    print("="*30 + "\n")

    logger.info(f"Starting Uvicorn server on {host}:{port} with log level '{log_level}'")

    uvicorn.run(
        "app.main:app", # Point to the app instance within the app package
        host=host,
        port=port,
        reload=reload_flag,
        log_level=log_level
    )