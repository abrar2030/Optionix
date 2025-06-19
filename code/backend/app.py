"""
Enhanced main FastAPI application for Optionix platform.
Provides comprehensive API endpoints with security, compliance, and financial standards.
"""
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
import logging
import uvicorn
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

# Import configuration and database
from config import settings
from database import get_db, create_tables
from models import User, Account, Trade, Position

# Import schemas
from schemas import (
    UserCreate, UserLogin, UserResponse, TokenResponse,
    AccountCreate, AccountResponse, TradeRequest, TradeResponse,
    PositionResponse, PositionHealthResponse, MarketDataRequest,
    VolatilityResponse, HealthCheckResponse, ErrorResponse
)

# Import authentication and authorization
from auth import (
    authenticate_user, create_access_token, create_refresh_token,
    get_current_user, get_current_verified_user, log_auth_event
)

# Import services
from services.blockchain_service import BlockchainService
from services.model_service import ModelService
from services.financial_service import FinancialCalculationService
from services.compliance_service import compliance_service

# Import middleware
from middleware.rate_limiting import rate_limit_middleware
from middleware.audit_logging import audit_middleware, audit_logger

# Import security utilities
from security import security_service

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Optionix API...")
    try:
        create_tables()
        logger.info("Database tables created/verified")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Optionix API...")


# Create FastAPI app with enhanced configuration
app = FastAPI(
    title=settings.app_name,
    description="Comprehensive API for options trading platform with volatility prediction, blockchain integration, and financial compliance",
    version=settings.app_version,
    docs_url=None,  # Custom docs endpoint
    redoc_url=None,
    lifespan=lifespan
)

# Add security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # Configure for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(audit_middleware)

# Initialize services
blockchain_service = BlockchainService()
model_service = ModelService()
financial_service = FinancialCalculationService()

# Security scheme
security = HTTPBearer()


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Custom value error handler"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "validation_error",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An internal error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check():
    """
    System health check endpoint
    
    Returns:
        HealthCheckResponse: System health status
    """
    services_status = {
        "database": "healthy",
        "blockchain": "healthy" if blockchain_service.is_connected() else "unhealthy",
        "model": "healthy" if model_service.is_model_available() else "unhealthy",
        "redis": "healthy"  # Would check Redis connection in production
    }
    
    overall_status = "healthy" if all(
        status == "healthy" for status in services_status.values()
    ) else "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        version=settings.app_version,
        services=services_status
    )


# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """
    Root endpoint with API information
    
    Returns:
        dict: API status and information
    """
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "status": "online",
        "documentation": "/docs",
        "timestamp": datetime.utcnow().isoformat()
    }


# Authentication endpoints
@app.post("/auth/register", response_model=UserResponse, tags=["Authentication"])
async def register_user(
    user_data: UserCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Register a new user account
    
    Args:
        user_data (UserCreate): User registration data
        request (Request): FastAPI request object
        db (Session): Database session
        
    Returns:
        UserResponse: Created user information
    """
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Validate password strength
    password_validation = security_service.validate_password_strength(user_data.password)
    if not password_validation["valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password validation failed: {', '.join(password_validation['issues'])}"
        )
    
    # Create user
    from auth import get_password_hash
    hashed_password = get_password_hash(user_data.password)
    
    user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        is_active=True,
        is_verified=False,
        kyc_status="pending"
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Log registration event
    log_auth_event(
        db, user.id, "user_registration", 
        request.client.host, request.headers.get("user-agent", ""),
        "success"
    )
    
    return UserResponse.from_orm(user)


@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login_user(
    credentials: UserLogin,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return access tokens
    
    Args:
        credentials (UserLogin): Login credentials
        request (Request): FastAPI request object
        db (Session): Database session
        
    Returns:
        TokenResponse: Access and refresh tokens
    """
    # Authenticate user
    user = authenticate_user(db, credentials.email, credentials.password)
    if not user:
        log_auth_event(
            db, None, "login_failed", 
            request.client.host, request.headers.get("user-agent", ""),
            "failure", "Invalid credentials"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Create tokens
    access_token = create_access_token(data={"sub": user.user_id})
    refresh_token = create_refresh_token(data={"sub": user.user_id})
    
    # Log successful login
    log_auth_event(
        db, user.id, "login_success", 
        request.client.host, request.headers.get("user-agent", ""),
        "success"
    )
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.access_token_expire_minutes * 60
    )


# Trading endpoints
@app.post("/trades", response_model=TradeResponse, tags=["Trading"])
async def create_trade(
    trade_data: TradeRequest,
    request: Request,
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """
    Create a new trade order
    
    Args:
        trade_data (TradeRequest): Trade order data
        request (Request): FastAPI request object
        current_user (User): Current authenticated user
        db (Session): Database session
        
    Returns:
        TradeResponse: Created trade information
    """
    # Check compliance
    compliance_check = compliance_service.check_transaction_compliance(
        trade_data.dict(), current_user.id, db
    )
    
    if not compliance_check["compliant"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Trade violates compliance rules: {', '.join(compliance_check['violations'])}"
        )
    
    # Calculate trade value and fees
    total_value = trade_data.quantity * (trade_data.price or 0)
    fees = financial_service.calculate_trading_fees(total_value)
    
    # Create trade record
    trade = Trade(
        user_id=current_user.id,
        symbol=trade_data.symbol,
        trade_type=trade_data.trade_type,
        order_type=trade_data.order_type,
        quantity=trade_data.quantity,
        price=trade_data.price or 0,
        total_value=total_value,
        fees=fees,
        status="pending"
    )
    
    db.add(trade)
    db.commit()
    db.refresh(trade)
    
    # Log trade creation
    audit_logger.log_trade_event(
        "trade_created", current_user.id, trade.trade_id,
        trade_data.dict(), request.client.host, "success"
    )
    
    return TradeResponse.from_orm(trade)


@app.get("/trades", response_model=List[TradeResponse], tags=["Trading"])
async def get_user_trades(
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0
):
    """
    Get user's trading history
    
    Args:
        current_user (User): Current authenticated user
        db (Session): Database session
        limit (int): Maximum number of trades to return
        offset (int): Number of trades to skip
        
    Returns:
        List[TradeResponse]: List of user trades
    """
    trades = db.query(Trade).filter(
        Trade.user_id == current_user.id
    ).order_by(Trade.created_at.desc()).offset(offset).limit(limit).all()
    
    return [TradeResponse.from_orm(trade) for trade in trades]


# Position endpoints
@app.get("/positions", response_model=List[PositionResponse], tags=["Positions"])
async def get_user_positions(
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """
    Get user's current positions
    
    Args:
        current_user (User): Current authenticated user
        db (Session): Database session
        
    Returns:
        List[PositionResponse]: List of user positions
    """
    positions = db.query(Position).filter(
        Position.account.has(user_id=current_user.id),
        Position.status == "open"
    ).all()
    
    return [PositionResponse.from_orm(position) for position in positions]


@app.get("/position_health/{address}", response_model=PositionHealthResponse, tags=["Positions"])
async def get_position_health(
    address: str,
    current_user: User = Depends(get_current_verified_user)
):
    """
    Get comprehensive position health metrics
    
    Args:
        address (str): Ethereum address
        current_user (User): Current authenticated user
        
    Returns:
        PositionHealthResponse: Position health metrics
    """
    if not security_service.validate_ethereum_address(address):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Ethereum address"
        )
    
    try:
        health_data = blockchain_service.get_position_health(address)
        return PositionHealthResponse(**health_data)
    except Exception as e:
        logger.error(f"Error fetching position health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch position health"
        )


# Prediction endpoints
@app.post("/predict_volatility", response_model=VolatilityResponse, tags=["Predictions"])
async def predict_volatility(
    data: MarketDataRequest,
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """
    Predict volatility based on market data
    
    Args:
        data (MarketDataRequest): Market data
        current_user (User): Current authenticated user
        db (Session): Database session
        
    Returns:
        VolatilityResponse: Volatility prediction
    """
    if not model_service.is_model_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Volatility model not available"
        )
    
    try:
        prediction_result = model_service.predict_volatility(data.dict(), db)
        
        return VolatilityResponse(
            volatility=prediction_result["volatility"],
            confidence=prediction_result.get("confidence"),
            model_version=prediction_result["model_version"]
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction service error"
        )


# Account endpoints
@app.post("/accounts", response_model=AccountResponse, tags=["Accounts"])
async def create_account(
    account_data: AccountCreate,
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """
    Create a new trading account
    
    Args:
        account_data (AccountCreate): Account creation data
        current_user (User): Current authenticated user
        db (Session): Database session
        
    Returns:
        AccountResponse: Created account information
    """
    # Validate Ethereum address
    if not security_service.validate_ethereum_address(account_data.ethereum_address):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Ethereum address"
        )
    
    # Check if address already exists
    existing_account = db.query(Account).filter(
        Account.ethereum_address == account_data.ethereum_address
    ).first()
    if existing_account:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ethereum address already associated with an account"
        )
    
    # Create account
    account = Account(
        user_id=current_user.id,
        ethereum_address=account_data.ethereum_address,
        account_type=account_data.account_type,
        balance_usd=0,
        margin_available=0,
        margin_used=0
    )
    
    db.add(account)
    db.commit()
    db.refresh(account)
    
    return AccountResponse.from_orm(account)


# Documentation endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with enhanced styling"""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{settings.app_name} - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
    )


@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """Return OpenAPI schema with enhanced documentation"""
    return get_openapi(
        title=settings.app_name,
        version=settings.app_version,
        description=app.description,
        routes=app.routes,
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

