"""
Enhanced Pydantic schemas for Optionix platform.
Includes schemas for security, compliance, and financial standards features.
"""
from pydantic import BaseModel, EmailStr, validator, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from decimal import Decimal
from enum import Enum


# Enums
class UserRole(str, Enum):
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"
    COMPLIANCE_OFFICER = "compliance_officer"
    RISK_MANAGER = "risk_manager"
    API_USER = "api_user"


class KYCStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    UNDER_REVIEW = "under_review"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    PENDING = "pending"


# Base schemas
class BaseResponse(BaseModel):
    """Base response schema with common fields"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class ErrorResponse(BaseResponse):
    """Error response schema"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseResponse):
    """Health check response schema"""
    status: str
    version: str
    services: Dict[str, str]
    security_features: Optional[Dict[str, bool]] = None


# User schemas
class UserCreate(BaseModel):
    """User creation schema with enhanced validation"""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: str = Field(..., min_length=2, max_length=255)
    data_retention_consent: bool = False
    marketing_consent: bool = False
    data_processing_consent: bool = True
    
    @validator('full_name')
    def validate_full_name(cls, v):
        if not v.replace(' ', '').replace('-', '').replace("'", '').replace('.', '').isalpha():
            raise ValueError('Full name must contain only letters, spaces, hyphens, apostrophes, and periods')
        return v.strip()
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserLogin(BaseModel):
    """User login schema with MFA support"""
    email: EmailStr
    password: str
    mfa_token: Optional[str] = Field(None, min_length=6, max_length=6)
    remember_me: bool = False


class UserResponse(BaseModel):
    """User response schema"""
    user_id: str
    email: str
    full_name: str
    role: UserRole
    is_active: bool
    is_verified: bool
    kyc_status: KYCStatus
    mfa_enabled: bool
    risk_score: int
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class TokenResponse(BaseResponse):
    """Token response schema"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    session_id: Optional[str] = None


# MFA schemas
class MFASetupResponse(BaseResponse):
    """MFA setup response schema"""
    secret: str
    qr_code: str
    backup_codes: List[str]


class MFAVerifyRequest(BaseModel):
    """MFA verification request schema"""
    token: str = Field(..., min_length=6, max_length=6)


# KYC schemas
class AddressData(BaseModel):
    """Address data schema"""
    street: str = Field(..., min_length=5, max_length=255)
    city: str = Field(..., min_length=2, max_length=100)
    state: Optional[str] = Field(None, max_length=100)
    postal_code: str = Field(..., min_length=3, max_length=20)
    country: str = Field(..., min_length=2, max_length=3)


class KYCDataRequest(BaseModel):
    """KYC data submission schema"""
    full_name: str = Field(..., min_length=2, max_length=255)
    date_of_birth: str = Field(..., regex=r'^\d{4}-\d{2}-\d{2}$')
    nationality: str = Field(..., min_length=2, max_length=3)
    address: AddressData
    document_type: str = Field(..., regex=r'^(passport|national_id|drivers_license)$')
    document_number: str = Field(..., min_length=5, max_length=50)
    document_country: str = Field(..., min_length=2, max_length=3)
    document_expiry: str = Field(..., regex=r'^\d{4}-\d{2}-\d{2}$')
    
    @validator('date_of_birth')
    def validate_date_of_birth(cls, v):
        try:
            birth_date = datetime.strptime(v, '%Y-%m-%d')
            age = (datetime.now() - birth_date).days / 365.25
            if age < 18:
                raise ValueError('Must be at least 18 years old')
            if age > 120:
                raise ValueError('Invalid date of birth')
        except ValueError as e:
            if 'Invalid date of birth' in str(e) or 'Must be at least' in str(e):
                raise e
            raise ValueError('Invalid date format. Use YYYY-MM-DD')
        return v


# Account schemas
class AccountCreate(BaseModel):
    """Account creation schema"""
    ethereum_address: str = Field(..., regex=r'^0x[a-fA-F0-9]{40}$')
    account_type: str = Field(default="standard", regex=r'^(standard|premium|institutional|demo)$')
    initial_deposit: Optional[Decimal] = Field(None, ge=0, le=1000000)


class AccountResponse(BaseModel):
    """Account response schema"""
    account_id: str
    ethereum_address: str
    account_type: str
    account_status: str
    balance_usd: Decimal
    margin_available: Decimal
    margin_used: Decimal
    risk_limit: Decimal
    created_at: datetime
    
    class Config:
        from_attributes = True


# Trading schemas
class TradeRequest(BaseModel):
    """Trade request schema with enhanced validation"""
    account_id: int
    symbol: str = Field(..., min_length=3, max_length=20)
    trade_type: str = Field(..., regex=r'^(buy|sell)$')
    order_type: str = Field(..., regex=r'^(market|limit|stop|stop_limit)$')
    quantity: Decimal = Field(..., gt=0, le=1000000)
    price: Optional[Decimal] = Field(None, gt=0, le=1000000)
    stop_loss: Optional[Decimal] = Field(None, gt=0)
    take_profit: Optional[Decimal] = Field(None, gt=0)
    client_order_id: Optional[str] = Field(None, max_length=100)
    
    @validator('price')
    def validate_price_for_limit_orders(cls, v, values):
        order_type = values.get('order_type')
        if order_type in ['limit', 'stop_limit'] and v is None:
            raise ValueError('Price is required for limit and stop-limit orders')
        return v


class TradeResponse(BaseModel):
    """Trade response schema"""
    trade_id: str
    symbol: str
    trade_type: str
    order_type: str
    quantity: Decimal
    price: Decimal
    executed_price: Optional[Decimal]
    total_value: Decimal
    fees: Decimal
    status: str
    compliance_status: str
    risk_score: int
    created_at: datetime
    executed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


# Position schemas
class PositionResponse(BaseModel):
    """Position response schema"""
    position_id: str
    symbol: str
    position_type: str
    size: Decimal
    entry_price: Decimal
    current_price: Optional[Decimal]
    liquidation_price: Optional[Decimal]
    unrealized_pnl: Decimal
    margin_requirement: Decimal
    status: str
    created_at: datetime
    
    # Greeks for options
    delta: Optional[Decimal] = None
    gamma: Optional[Decimal] = None
    theta: Optional[Decimal] = None
    vega: Optional[Decimal] = None
    rho: Optional[Decimal] = None
    
    class Config:
        from_attributes = True


class PositionHealthResponse(BaseModel):
    """Position health response schema"""
    position_id: str
    health_score: Decimal
    margin_ratio: Decimal
    liquidation_risk: RiskLevel
    time_to_liquidation: Optional[int]  # minutes
    recommended_actions: List[str]
    risk_metrics: Dict[str, Decimal]


# Market data schemas
class MarketDataRequest(BaseModel):
    """Market data request schema"""
    symbol: str = Field(..., min_length=3, max_length=20)
    timeframe: str = Field(default="1h", regex=r'^(1m|5m|15m|1h|4h|1d)$')
    limit: int = Field(default=100, ge=1, le=1000)


class VolatilityResponse(BaseModel):
    """Volatility prediction response schema"""
    symbol: str
    volatility: Decimal
    confidence: Optional[Decimal]
    model_version: str
    prediction_horizon: str
    timestamp: datetime


# Compliance schemas
class ComplianceCheckResponse(BaseResponse):
    """Compliance check response schema"""
    status: ComplianceStatus
    risk_level: RiskLevel
    checks_performed: List[str]
    issues_found: List[str]
    recommendations: List[str]
    next_review_date: Optional[datetime] = None


class RiskMetricsResponse(BaseResponse):
    """Risk metrics response schema"""
    user_id: str
    risk_metrics: Dict[str, Dict[str, Any]]
    overall_risk_score: int
    last_updated: str


class TransactionMonitoringAlert(BaseModel):
    """Transaction monitoring alert schema"""
    alert_id: str
    alert_type: str
    alert_category: str
    description: str
    risk_score: int
    risk_level: RiskLevel
    status: str
    created_at: datetime


class SanctionsCheckResponse(BaseModel):
    """Sanctions check response schema"""
    check_id: str
    matches_found: bool
    match_details: List[Dict[str, Any]]
    lists_checked: List[str]
    risk_score: int
    checked_at: datetime


# API Key schemas
class APIKeyCreate(BaseModel):
    """API key creation schema"""
    key_name: str = Field(..., min_length=3, max_length=100)
    permissions: List[str]
    ip_whitelist: Optional[List[str]] = None
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """API key response schema"""
    key_id: str
    key_name: str
    key_prefix: str
    permissions: List[str]
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class APIKeyCreateResponse(APIKeyResponse):
    """API key creation response with secret"""
    api_key: str  # Only returned once during creation


# Audit schemas
class AuditLogResponse(BaseModel):
    """Audit log response schema"""
    log_id: str
    action: str
    action_category: str
    resource_type: Optional[str]
    resource_id: Optional[str]
    status: str
    ip_address: Optional[str]
    timestamp: datetime
    compliance_impact: str
    
    class Config:
        from_attributes = True


class AuditLogQuery(BaseModel):
    """Audit log query schema"""
    user_id: Optional[int] = None
    action_category: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


# Financial reporting schemas
class FinancialReportRequest(BaseModel):
    """Financial report request schema"""
    report_type: str = Field(..., regex=r'^(daily|weekly|monthly|quarterly|annual)$')
    regulation_type: str = Field(..., regex=r'^(sox|mifid_ii|dodd_frank|basel_iii|cftc)$')
    period_start: datetime
    period_end: datetime
    include_details: bool = False


class FinancialReportResponse(BaseResponse):
    """Financial report response schema"""
    report_id: str
    report_type: str
    regulation_type: str
    period_start: datetime
    period_end: datetime
    status: str
    generated_at: datetime
    data_summary: Dict[str, Any]


# Data protection schemas
class DataSubjectRequest(BaseModel):
    """GDPR data subject request schema"""
    request_type: str = Field(..., regex=r'^(access|rectification|erasure|portability|restriction)$')
    description: Optional[str] = Field(None, max_length=1000)
    verification_method: Optional[str] = Field(None, regex=r'^(email|phone|document)$')


class DataSubjectRequestResponse(BaseResponse):
    """Data subject request response schema"""
    request_id: str
    request_type: str
    status: str
    estimated_completion: Optional[datetime]
    verification_required: bool


class DataExportResponse(BaseResponse):
    """Data export response schema"""
    export_id: str
    file_path: str
    file_size: int
    expires_at: datetime


# Reconciliation schemas
class ReconciliationRequest(BaseModel):
    """Reconciliation request schema"""
    reconciliation_type: str = Field(..., regex=r'^(daily|monthly|trade|position)$')
    business_date: datetime
    tolerance_threshold: Optional[Decimal] = Field(Decimal('0.01'), ge=0)


class ReconciliationResponse(BaseResponse):
    """Reconciliation response schema"""
    reconciliation_id: str
    reconciliation_type: str
    status: str
    internal_balance: Decimal
    external_balance: Decimal
    difference: Decimal
    within_tolerance: bool
    business_date: datetime


# Pagination schemas
class PaginatedResponse(BaseModel):
    """Paginated response schema"""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
    has_next: bool
    has_prev: bool


# Bulk operation schemas
class BulkOperationRequest(BaseModel):
    """Bulk operation request schema"""
    operation_type: str
    items: List[Dict[str, Any]]
    batch_size: int = Field(default=100, ge=1, le=1000)


class BulkOperationResponse(BaseResponse):
    """Bulk operation response schema"""
    operation_id: str
    total_items: int
    processed_items: int
    failed_items: int
    status: str
    errors: List[Dict[str, Any]]


# Notification schemas
class NotificationPreferences(BaseModel):
    """Notification preferences schema"""
    email_enabled: bool = True
    sms_enabled: bool = False
    push_enabled: bool = True
    trade_notifications: bool = True
    compliance_notifications: bool = True
    security_notifications: bool = True
    marketing_notifications: bool = False


class NotificationResponse(BaseModel):
    """Notification response schema"""
    notification_id: str
    type: str
    title: str
    message: str
    priority: str
    read: bool
    created_at: datetime


# System configuration schemas
class SystemConfigResponse(BaseModel):
    """System configuration response schema"""
    trading_enabled: bool
    maintenance_mode: bool
    api_version: str
    supported_symbols: List[str]
    trading_hours: Dict[str, str]
    fee_structure: Dict[str, Decimal]
    risk_limits: Dict[str, Decimal]

