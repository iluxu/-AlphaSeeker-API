# API v1 router aggregation
# app/api/v1/api.py
from fastapi import APIRouter

from app.api.v1.endpoints import analysis, scan, tickers # Import endpoint modules

api_router = APIRouter()

# Include routers from endpoints
api_router.include_router(analysis.router, prefix="/crypto", tags=["Analysis"])
api_router.include_router(scan.router, prefix="/crypto", tags=["Scanning"])
api_router.include_router(tickers.router, prefix="/crypto", tags=["Utility"])

# You could add more endpoints or sub-routers here