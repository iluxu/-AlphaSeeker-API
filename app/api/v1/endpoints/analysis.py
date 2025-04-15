# Analysis endpoint logic
# Example: app/api/v1/endpoints/analysis.py
from fastapi import APIRouter, HTTPException, Depends # Use Depends for dependency injection later if needed
import logging

# Import request/response models
from app.models.analysis import AnalysisRequest, AnalysisResponse

# Import the service function
from app.services.analysis_service import perform_single_analysis

# Import clients/config if needed directly for checks (e.g., binance_futures available)
from app.core.clients import binance_futures
from app.core.config import settings # Maybe for default values?

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_crypto_endpoint(request: AnalysisRequest):
    logger.info(f"API: Analyze request received for {request.symbol} ({request.timeframe})")
    if binance_futures is None: raise HTTPException(status_code=503, detail="Exchange unavailable.")

    try:
        # Pass parameters using model_dump()
        analysis_params = request.model_dump()
        # Add params not in AnalysisRequest but needed by perform_single_analysis
        analysis_params['min_requested_rr'] = 1.5 # Default R/R for single analysis? Or make it a query param?
        analysis_params['min_adx_for_trigger'] = 20.0 # Default ADX for single analysis

        result = await perform_single_analysis(**analysis_params)

        # Error Handling (same logic as before)
        if result.error:
            err = result.error; status = 500; log_func = logger.error
            if "Warning:" in err or "skipped:" in err or "Insufficient data" in err: status = 200; log_func = logger.warning # Treat warnings/skips as OK
            elif "Invalid symbol" in err: status = 400
            elif "Network error" in err or "Connection error" in err or "Failed to load markets" in err or "Data Fetch Error" in err: status = 504
            elif "Rate limit exceeded" in err or "ConnectionAbortedError" in err: status = 429
            elif "Authentication Error" in err: status = 401
            elif "OpenAI" in err or "GPT" in err: status = 502
            elif "Exchange error" in err: status = 503

            log_func(f"API Analyze: Request for {request.symbol} finished with status {status}. Detail: {err}")
            if status != 200: raise HTTPException(status_code=status, detail=err)

        logger.info(f"API Analyze: Request for {request.symbol} completed successfully.")
        return result

    except HTTPException as h: raise h
    except Exception as e:
        logger.error(f"API Analyze Endpoint Error for {request.symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected endpoint internal error: {e}")