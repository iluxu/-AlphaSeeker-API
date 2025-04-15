# Configuration and settings loading
# app/core/config.py
import os
import logging
import sys
import warnings
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Optional

# Load .env file FIRST
load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Crypto Trading Analysis & Scanning API"
    PROJECT_VERSION: str = "1.4.0_MACD_Signal" # Or load from file/env
    PROJECT_DESCRIPTION: str = "API for technical analysis, GPT evaluation, backtesting, and scanning."
    API_V1_STR: str = "/api/v1"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # OpenAI
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Binance (Add keys if needed for private endpoints)
    # BINANCE_API_KEY: Optional[str] = os.getenv("BINANCE_API_KEY")
    # BINANCE_SECRET: Optional[str] = os.getenv("BINANCE_SECRET")
    CCXT_TIMEOUT: int = 30000
    CCXT_RATE_LIMIT: int = 150 # Safer default

    # Scanner Defaults (can be overridden by request)
    DEFAULT_MAX_CONCURRENT_TASKS: int = 10
    SCANNER_DEFAULT_MIN_GPT_CONFIDENCE: float = 0.60
    SCANNER_DEFAULT_MIN_BACKTEST_SCORE: float = 0.55
    # ... other defaults if needed

    class Config:
        case_sensitive = True
        # If using .env file:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

# --- Logging Configuration ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
log_handler = logging.StreamHandler(sys.stdout) # Use stdout for container/cloud environments
log_handler.setFormatter(log_formatter)
logger = logging.getLogger() # Get root logger
logger.setLevel(settings.LOG_LEVEL)
if logger.hasHandlers(): logger.handlers.clear() # Avoid duplicate handlers on reload
logger.addHandler(log_handler)
logger.critical("--- Logging Initialized (Level: %s). App Name: %s ---", settings.LOG_LEVEL, settings.PROJECT_NAME)

# --- Warning Filters ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="ConvergenceWarning", category=UserWarning)
# Silence TensorFlow if still implicitly imported by a dependency
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)