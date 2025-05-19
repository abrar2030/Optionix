"""
Main FastAPI application for Optionix platform.
Provides API endpoints for options trading, volatility prediction, and position health.
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging

# Import services
from services.blockchain_service import BlockchainService
from services.model_service import ModelService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with enhanced documentation
app = FastAPI(
    title="Optionix API",
    description="API for options trading platform with volatility prediction and blockchain integration",
    version="1.0.0",
    docs_url=None,  # Disable default docs
)

# Initialize services
blockchain_service = BlockchainService()
model_service = ModelService()

# Define request/response models
class MarketDataRequest(BaseModel):
    open: float
    high: float
    low: float
    volume: int

class VolatilityResponse(BaseModel):
    volatility: float

class PositionHealthResponse(BaseModel):
    address: str
    size: int
    is_long: bool
    entry_price: float
    liquidation_price: float

# Dependency for services
def get_blockchain_service():
    return blockchain_service

def get_model_service():
    return model_service

@app.get("/", tags=["General"])
async def root():
    """
    Root endpoint that returns API status
    
    Returns:
        dict: A message indicating the API is online
    """
    return {"message": "Welcome to Optionix API", "status": "online"}

@app.post("/predict_volatility", response_model=VolatilityResponse, tags=["Predictions"])
async def predict_volatility(
    data: MarketDataRequest,
    model_service: ModelService = Depends(get_model_service)
):
    """
    Predict volatility based on market data using ML model
    
    Parameters:
        data (MarketDataRequest): Market data with opening price, high, low, and volume
    
    Returns:
        VolatilityResponse: Predicted volatility as a float
        
    Raises:
        HTTPException: If model is unavailable or prediction fails
    
    Example:
        ```
        {
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "volume": 1000000
        }
        ```
    """
    if not model_service.is_model_available():
        raise HTTPException(status_code=503, detail="Volatility model not available")
    
    try:
        market_data = {
            'open': data.open,
            'high': data.high,
            'low': data.low,
            'volume': data.volume
        }
        volatility = model_service.predict_volatility(market_data)
        return {"volatility": volatility}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/position_health/{address}", response_model=PositionHealthResponse, tags=["Blockchain"])
async def get_position_health(
    address: str,
    blockchain_service: BlockchainService = Depends(get_blockchain_service)
):
    """
    Get health metrics for a trading position by Ethereum address
    
    Parameters:
        address (str): Ethereum address of the position owner
    
    Returns:
        PositionHealthResponse: Position health metrics
    
    Raises:
        HTTPException: If address is invalid or position cannot be fetched
    
    Example response:
        ```
        {
            "address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            "size": 100,
            "is_long": true,
            "entry_price": 1000,
            "liquidation_price": 900
        }
        ```
    """
    if not blockchain_service.is_valid_address(address):
        raise HTTPException(status_code=400, detail="Invalid Ethereum address")
    
    try:
        position = blockchain_service.get_position_health(address)
        return position
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    Custom Swagger UI with enhanced styling and configuration
    """
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=app.title + " - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """
    Return OpenAPI schema with enhanced documentation
    """
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
