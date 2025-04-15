# FastAPI application entry point
# app/main.py
import uvicorn
from fastapi import FastAPI
from app.api.v1.api import api_router as api_v1_router # Import the aggregated router
from app.core.config import settings # Import settings
from app.core.clients import load_exchange_markets, test_openai_connection, binance_futures, openai_client # Import startup functions/clients
import asyncio
import logging

# Logger setup might be initiated in config, but ensure app uses it
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json" # Define where OpenAPI spec lives
)

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI startup: Initializing...")
    # Initialize clients and load data asynchronously
    if binance_futures:
        asyncio.create_task(load_exchange_markets(binance_futures))
    if openai_client:
        asyncio.create_task(test_openai_connection(openai_client))
    logger.info("Startup tasks created.")

# Include the API router
app.include_router(api_v1_router, prefix=settings.API_V1_STR)

# Optional: Add a root endpoint
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to {settings.PROJECT_NAME} v{settings.PROJECT_VERSION}"}

# Note: The uvicorn.run call is now in the top-level main_runner.py