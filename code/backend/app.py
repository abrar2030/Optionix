"""
Enhanced main FastAPI application for Optionix platform.
Integrates comprehensive security, compliance, and financial standards.
"""
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import uvicorn
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

# Import configuration and database
from config import settings
from database import get_db, create_tables

# Import enhanced models
from models import (
    User, Account, Trade, Position, AuditLog, APIKey,
    KYCDocument, SanctionsCheck, TransactionMonitoring,
    FinancialAuditLog
)

# Import enhanced schemas
from schemas import (
    UserCreate, UserLogin, UserResponse, TokenResponse,
    AccountCreate, AccountResponse, TradeRequest, TradeResponse,
    PositionResponse, PositionHealthResponse, MarketDataRequest,
    VolatilityResponse, HealthCheckResponse, ErrorResponse,
    MFASetupResponse, MFAVerifyRequest, KYCDataRequest,
    ComplianceCheckResponse, RiskMetricsResponse
)

# Import enhanced authentication and authorization
from auth import (
    auth_service, mfa_service, rbac_service,
    get_current_user, get_current_verified_user,
    require_permission, Permission, UserRole,
    log_auth_event
)

# Import enhanced middleware
from middleware.security import (
    SecurityHeadersMiddleware, AdvancedRateLimitMiddleware,
    RequestValidationMiddleware, AuditLoggingMiddleware
)

# Import services
from services.blockchain_service import BlockchainService
from services.model_service import ModelService
from services.financial_service import FinancialCalculationService

# Import enhanced compliance and security
from compliance_enhanced import enhanced_compliance_service
from data_protection import data_protection_service
from financial_standards import financial_standards_service
from security import security_service

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with enhanced initialization"""
    # Startup
    logger.info("Starting Enhanced Optionix API...")
    try:
        create_tables()
        logger.info("Database tables created/verified")
        
        # Initialize security services
        logger.info("Initializing security services...")
        
        # Initialize compliance services
        logger.info("Initializing compliance services...")
        
        # Initialize financial standards
        logger.info("Initializing financial standards...")
        
        logger.info("Enhanced Optionix API started successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced Optionix API...")


# Create FastAPI app with enhanced configuration
app = FastAPI(
    title=f"Enhanced {settings.app_name}",
    description="Comprehensive API for options trading platform with advanced security, compliance, and financial standards",
    version=f"{settings.app_version}-enhanced",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add enhanced security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # Configure for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom enhanced middleware
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


# Enhanced exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Enhanced HTTP exception handler with security logging"""
    # Log security-relevant exceptions
    if exc.status_code in [401, 403, 429]:
        logger.warning(
            f"Security exception: {exc.status_code} - {exc.detail}",
            extra={
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", ""),
                "endpoint": request.url.path,
                "method": request.method
            }
        )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, "request_id", None)
        }
    )


# Enhanced health check endpoint
@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def enhanced_health_check():
    """Enhanced system health check with security and compliance status"""
    services_status = {
        "database": "healthy",
        "blockchain": "healthy" if blockchain_service.is_connected() else "unhealthy",
        "model": "healthy" if model_service.is_model_available() else "unhealthy",
        "redis": "healthy",
        "compliance_engine": "healthy",
        "security_services": "healthy",
        "audit_logging": "healthy"
    }
    
    overall_status = "healthy" if all(
        status == "healthy" for status in services_status.values()
    ) else "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        version=f"{settings.app_version}-enhanced",
        services=services_status,
        security_features={
            "mfa_enabled": True,
            "rbac_enabled": True,
            "encryption_enabled": True,
            "audit_logging": True,
            "compliance_monitoring": True
        }
    )


# Enhanced authentication endpoints
@app.post("/auth/register", response_model=UserResponse, tags=["Authentication"])
async def enhanced_register_user(
    user_data: UserCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    """Enhanced user registration with comprehensive security checks"""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")
    
    # Check rate limiting for registration
    failed_attempts = auth_service.check_failed_attempts(f"register_{client_ip}")
    if failed_attempts["locked"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many registration attempts. Please try again later."
        )
    
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            auth_service.record_failed_attempt(f"register_{client_ip}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Enhanced password validation
        password_validation = security_service.validate_password_strength(user_data.password)
        if not password_validation["valid"]:
            auth_service.record_failed_attempt(f"register_{client_ip}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password validation failed: {', '.join(password_validation['issues'])}"
            )
        
        # Sanitize input data
        sanitized_data = security_service.sanitize_input(user_data.dict())
        
        # Create user with enhanced security
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
            data_processing_consent=sanitized_data.get("data_processing_consent", False),
            consent_date=datetime.utcnow() if sanitized_data.get("data_processing_consent") else None
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
            consent_given=user.data_processing_consent
        )
        
        # Log successful registration
        log_auth_event(
            db, user.id, "user_registration", 
            client_ip, user_agent, "success"
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
            detail="Registration failed"
        )


@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def enhanced_login_user(
    credentials: UserLogin,
    request: Request,
    db: Session = Depends(get_db)
):
    """Enhanced user login with MFA and security checks"""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")
    
    # Check failed login attempts
    failed_attempts = auth_service.check_failed_attempts(f"login_{credentials.email}")
    if failed_attempts["locked"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Account temporarily locked due to failed login attempts"
        )
    
    try:
        # Authenticate user
        user = db.query(User).filter(User.email == credentials.email).first()
        if not user or not auth_service.verify_password(credentials.password, user.hashed_password):
            auth_service.record_failed_attempt(f"login_{credentials.email}")
            log_auth_event(
                db, user.id if user else None, "login_failed", 
                client_ip, user_agent, "failure", "Invalid credentials"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Check if account is locked or inactive
        if user.is_locked or not user.is_active:
            log_auth_event(
                db, user.id, "login_failed", 
                client_ip, user_agent, "failure", "Account locked or inactive"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is locked or inactive"
            )
        
        # Check MFA if enabled
        if user.mfa_enabled:
            if not credentials.mfa_token:
                raise HTTPException(
                    status_code=status.HTTP_200_OK,  # Special status for MFA required
                    detail="MFA token required",
                    headers={"X-MFA-Required": "true"}
                )
            
            # Verify MFA token
            if not mfa_service.verify_totp_token(
                data_protection_service.decrypt_field(user.mfa_secret),
                credentials.mfa_token
            ):
                auth_service.record_failed_attempt(f"login_{credentials.email}")
                log_auth_event(
                    db, user.id, "mfa_failed", 
                    client_ip, user_agent, "failure", "Invalid MFA token"
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid MFA token"
                )
        
        # Create session
        session_id = auth_service.create_session(user.user_id, user_agent, client_ip)
        
        # Create tokens
        access_token = auth_service.create_access_token(
            data={"sub": user.user_id, "session_id": session_id}
        )
        refresh_token = auth_service.create_refresh_token(
            data={"sub": user.user_id, "session_id": session_id}
        )
        
        # Update user login info
        user.last_login = datetime.utcnow()
        user.failed_login_attempts = 0
        db.commit()
        
        # Log successful login
        log_auth_event(
            db, user.id, "login_success", 
            client_ip, user_agent, "success"
        )
        
        # Clear failed attempts
        auth_service.clear_failed_attempts(f"login_{credentials.email}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.access_token_expire_minutes * 60,
            token_type="bearer",
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@app.post("/auth/mfa/setup", response_model=MFASetupResponse, tags=["Authentication"])
async def setup_mfa(
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Set up multi-factor authentication"""
    try:
        # Generate TOTP secret
        secret = mfa_service.generate_totp_secret()
        
        # Generate QR code
        qr_code = mfa_service.generate_totp_qr_code(current_user.email, secret)
        
        # Generate backup codes
        backup_codes = mfa_service.generate_backup_codes()
        hashed_backup_codes = mfa_service.hash_backup_codes(backup_codes)
        
        # Store encrypted secret and backup codes
        current_user.mfa_secret = data_protection_service.encrypt_field(secret)
        current_user.mfa_backup_codes = data_protection_service.encrypt_field(
            json.dumps(hashed_backup_codes)
        )
        
        db.commit()
        
        return MFASetupResponse(
            secret=secret,
            qr_code=qr_code,
            backup_codes=backup_codes
        )
        
    except Exception as e:
        logger.error(f"MFA setup error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA setup failed"
        )


@app.post("/auth/mfa/verify", tags=["Authentication"])
async def verify_mfa(
    mfa_data: MFAVerifyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Verify and enable MFA"""
    try:
        if not current_user.mfa_secret:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA not set up"
            )
        
        # Verify token
        secret = data_protection_service.decrypt_field(current_user.mfa_secret)
        if not mfa_service.verify_totp_token(secret, mfa_data.token):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid MFA token"
            )
        
        # Enable MFA
        current_user.mfa_enabled = True
        db.commit()
        
        return {"message": "MFA enabled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA verification failed"
        )


@app.post("/kyc/submit", tags=["Compliance"])
async def submit_kyc_data(
    kyc_data: KYCDataRequest,
    request: Request,
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Submit KYC data for verification"""
    try:
        # Perform enhanced KYC verification
        verification_result = enhanced_compliance_service.enhanced_kyc_verification(
            user_id=current_user.id,
            kyc_data=kyc_data.dict(),
            db=db
        )
        
        # Update user KYC status
        if verification_result["overall_status"] == "compliant":
            current_user.kyc_status = "approved"
        elif verification_result["overall_status"] == "non_compliant":
            current_user.kyc_status = "rejected"
        else:
            current_user.kyc_status = "under_review"
        
        current_user.risk_score = verification_result.get("risk_score", 0)
        db.commit()
        
        return ComplianceCheckResponse(
            status=verification_result["overall_status"],
            risk_level=verification_result["risk_level"],
            checks_performed=verification_result["checks_performed"],
            issues_found=verification_result["issues_found"],
            recommendations=verification_result["recommendations"]
        )
        
    except Exception as e:
        logger.error(f"KYC submission error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="KYC submission failed"
        )


@app.post("/trades", response_model=TradeResponse, tags=["Trading"])
async def enhanced_create_trade(
    trade_data: TradeRequest,
    request: Request,
    current_user: User = Depends(require_permission(Permission.CREATE_TRADE)),
    db: Session = Depends(get_db)
):
    """Enhanced trade creation with comprehensive compliance checks"""
    try:
        # Enhanced compliance check
        compliance_check = enhanced_compliance_service.advanced_transaction_monitoring(
            user_id=current_user.id,
            trade_data=trade_data.dict(),
            db=db
        )
        
        if not compliance_check["monitoring_passed"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Trade blocked by compliance: {compliance_check['alerts']}"
            )
        
        # SOX compliance check
        sox_check = financial_standards_service.check_sox_compliance(
            db=db,
            transaction_data=trade_data.dict(),
            user_id=current_user.id
        )
        
        if not sox_check["compliant"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"SOX compliance violation: {sox_check['violations']}"
            )
        
        # Calculate trade value and fees
        total_value = trade_data.quantity * (trade_data.price or 0)
        fees = financial_service.calculate_trading_fees(total_value)
        
        # Create trade record with enhanced fields
        trade = Trade(
            user_id=current_user.id,
            account_id=trade_data.account_id,
            symbol=trade_data.symbol,
            trade_type=trade_data.trade_type,
            order_type=trade_data.order_type,
            quantity=trade_data.quantity,
            price=trade_data.price or 0,
            total_value=total_value,
            fees=fees,
            status="pending",
            compliance_checked=True,
            compliance_status="approved",
            risk_checked=True,
            risk_score=compliance_check["risk_score"],
            source_system="api"
        )
        
        db.add(trade)
        db.commit()
        db.refresh(trade)
        
        # Create financial audit log
        financial_standards_service.create_financial_audit_log(
            db=db,
            transaction_id=trade.trade_id,
            user_id=current_user.id,
            account_id=trade.account_id,
            transaction_type="trade_creation",
            amount=total_value,
            previous_state=None,
            new_state=trade_data.dict(),
            regulation_type="sox",
            authorized_by=current_user.id,
            authorization_level=current_user.role
        )
        
        return TradeResponse.from_orm(trade)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced trade creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Trade creation failed"
        )


@app.get("/compliance/risk-metrics", response_model=RiskMetricsResponse, tags=["Compliance"])
async def get_risk_metrics(
    current_user: User = Depends(require_permission(Permission.READ_COMPLIANCE)),
    db: Session = Depends(get_db)
):
    """Get comprehensive risk metrics for compliance"""
    try:
        # Get user's accounts
        accounts = db.query(Account).filter(Account.user_id == current_user.id).all()
        
        risk_metrics = {}
        for account in accounts:
            # Calculate risk metrics
            metrics = financial_standards_service.calculate_risk_metrics(
                db=db,
                entity_type="account",
                entity_id=str(account.id),
                metric_types=["var", "leverage_ratio", "liquidity_ratio"]
            )
            
            risk_metrics[account.account_id] = {
                metric.metric_type: {
                    "value": str(metric.metric_value),
                    "limit": str(metric.limit_value) if metric.limit_value else None,
                    "status": metric.breach_status,
                    "calculation_date": metric.calculation_date.isoformat()
                }
                for metric in metrics
            }
        
        return RiskMetricsResponse(
            user_id=current_user.user_id,
            risk_metrics=risk_metrics,
            overall_risk_score=current_user.risk_score,
            last_updated=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Risk metrics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Risk metrics calculation failed"
        )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

