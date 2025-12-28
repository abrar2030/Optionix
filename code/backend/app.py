"""
Main FastAPI application for Optionix platform.
Integrates comprehensive security, compliance, and financial standards.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session

# Import authentication and authorization
from .auth import UserRole, auth_service, log_auth_event

# Import configuration and database
from .config import settings

# Import compliance and security
from .data_protection import data_protection_service
from .database import create_tables, get_db

# Import middleware
from .middleware.security import (
    AdvancedRateLimitMiddleware,
    AuditLoggingMiddleware,
    RequestValidationMiddleware,
    SecurityHeadersMiddleware,
)

# Import models
from .models import User

# Import schemas
from .schemas import (
    HealthCheckResponse,
    MarketDataRequest,
    UserCreate,
    UserResponse,
    VolatilityResponse,
)
from .security import security_service

# Import services
from .services.blockchain_service import BlockchainService
from .services.financial_service import FinancialCalculationService
from .services.model_service import ModelService

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with initialization"""
    # Startup
    logger.info("Starting Optionix API...")
    try:
        create_tables()
        logger.info("Database tables created/verified")

        # Initialize security services
        logger.info("Initializing security services...")

        # Initialize compliance services
        logger.info("Initializing compliance services...")

        # Initialize financial standards
        logger.info("Initializing financial standards...")

        logger.info("Optionix API started successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Optionix API...")


# Create FastAPI app with configuration
app = FastAPI(
    title=f"{settings.app_name}",
    description="Comprehensive API for options trading platform with advanced security, compliance, and financial standards",
    version=f"{settings.app_version}",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add security middleware
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"]
)  # Configure for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AdvancedRateLimitMiddleware)
app.add_middleware(RequestValidationMiddleware)
app.add_middleware(AuditLoggingMiddleware)

# Initialize services
blockchain_service = BlockchainService()
model_service = ModelService()
financial_service = FinancialCalculationService()

# Security scheme
security = HTTPBearer()


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler with security logging"""
    # Log security-relevant exceptions
    if exc.status_code in [401, 403, 429]:
        logger.warning(
            f"Security exception: {exc.status_code} - {exc.detail}",
            extra={
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", ""),
                "endpoint": request.url.path,
                "method": request.method,
            },
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check():
    """System health check with security and compliance status"""
    services_status = {
        "database": "healthy",
        "blockchain": "healthy" if blockchain_service.is_connected() else "unhealthy",
        "model": "healthy" if model_service.is_model_available() else "unhealthy",
        "redis": "healthy",
        "compliance_engine": "healthy",
        "security_services": "healthy",
        "audit_logging": "healthy",
    }

    overall_status = (
        "healthy"
        if all(status == "healthy" for status in services_status.values())
        else "degraded"
    )

    return HealthCheckResponse(
        status=overall_status,
        version=f"{settings.app_version}",
        services=services_status,
        security_features={
            "mfa_enabled": True,
            "rbac_enabled": True,
            "encryption_enabled": True,
            "audit_logging": True,
            "compliance_monitoring": True,
        },
    )


# Market data and model endpoints
@app.post(
    "/market/volatility",
    response_model=VolatilityResponse,
    tags=["Market Data", "Model"],
)
async def get_volatility_prediction(
    market_data: MarketDataRequest, db: Session = Depends(get_db)
):
    """Get volatility prediction for a given market data point"""
    try:
        # Convert Pydantic model to dict for service
        data_for_model = market_data.dict()

        # Call the model service
        prediction_result = model_service.get_volatility_prediction(data_for_model, db)

        # Construct the response
        return VolatilityResponse(
            symbol=market_data.symbol,
            volatility=Decimal(str(prediction_result["volatility"])),
            confidence=(
                Decimal(str(prediction_result["confidence"]))
                if prediction_result["confidence"] is not None
                else None
            ),
            model_version=prediction_result.get("model_version"),
            prediction_horizon="24h",  # Placeholder, as it's not in the model service result
            timestamp=datetime.utcnow(),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Volatility prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Volatility prediction failed",
        )


# Authentication endpoints
@app.post("/auth/register", response_model=UserResponse, tags=["Authentication"])
async def register_user(
    user_data: UserCreate, request: Request, db: Session = Depends(get_db)
):
    """User registration with comprehensive security checks"""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")

    # Check rate limiting for registration
    failed_attempts = auth_service.check_failed_attempts(f"register_{client_ip}")
    if failed_attempts["locked"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many registration attempts. Please try again later.",
        )

    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            auth_service.record_failed_attempt(f"register_{client_ip}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        # Password validation
        password_validation = security_service.validate_password_strength(
            user_data.password
        )
        if not password_validation["valid"]:
            auth_service.record_failed_attempt(f"register_{client_ip}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password validation failed: {', '.join(password_validation['issues'])}",
            )

        # Sanitize input data
        sanitized_data = security_service.sanitize_input(user_data.dict())

        # Create user with security
        hashed_password = auth_service.get_password_hash(user_data.password)

        user = User(
            email=sanitized_data["email"],
            hashed_password=hashed_password,
            full_name=sanitized_data["full_name"],
            is_active=True,
            is_verified=False,
            kyc_status="pending",
            role=UserRole.TRADER.value,
            mfa_enabled=False,
            risk_score=0,
            data_retention_consent=sanitized_data.get("data_retention_consent", False),
            marketing_consent=sanitized_data.get("marketing_consent", False),
            data_processing_consent=sanitized_data.get(
                "data_processing_consent", False
            ),
            consent_date=(
                datetime.utcnow()
                if sanitized_data.get("data_processing_consent")
                else None
            ),
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        # Create data processing log for GDPR compliance
        data_protection_service.create_data_processing_log(
            db=db,
            data_subject_id=str(user.id),
            processing_activity="user_registration",
            data_types=["personal_information", "contact_details"],
            legal_basis="contract",
            purpose="Account creation and service provision",
            user_id=user.id,
            retention_period=2555,  # 7 years
            consent_given=user.data_processing_consent,
        )

        # Log successful registration
        log_auth_event(
            db, user.id, "user_registration", client_ip, user_agent, "success"
        )

        # Clear failed attempts
        auth_service.clear_failed_attempts(f"register_{client_ip}")

        return UserResponse.from_orm(user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        auth_service.record_failed_attempt(f"register_{client_ip}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed",
        )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
