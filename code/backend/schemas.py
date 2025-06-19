"""
Pydantic schemas for request/response validation in Optionix API.
"""
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal
import re


# User schemas
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")
    full_name: str = Field(..., min_length=2, max_length=255)
    
    @validator('password')
    def validate_password(cls, v):
        if not re.search(r"[A-Z]", v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r"[a-z]", v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r"\d", v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError('Password must contain at least one special character')
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    user_id: str
    email: str
    full_name: str
    is_active: bool
    is_verified: bool
    kyc_status: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


# Account schemas
class AccountCreate(BaseModel):
    ethereum_address: str = Field(..., regex=r"^0x[a-fA-F0-9]{40}$")
    account_type: str = Field(default="standard", regex=r"^(standard|premium|institutional)$")


class AccountResponse(BaseModel):
    account_id: str
    ethereum_address: str
    account_type: str
    balance_usd: Decimal
    margin_available: Decimal
    margin_used: Decimal
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


# Trading schemas
class MarketDataRequest(BaseModel):
    open: float = Field(..., gt=0, description="Opening price must be positive")
    high: float = Field(..., gt=0, description="High price must be positive")
    low: float = Field(..., gt=0, description="Low price must be positive")
    volume: int = Field(..., gt=0, description="Volume must be positive")
    
    @validator('high')
    def validate_high_price(cls, v, values):
        if 'low' in values and v < values['low']:
            raise ValueError('High price must be greater than or equal to low price')
        return v
    
    @validator('low')
    def validate_low_price(cls, v, values):
        if 'open' in values and v > values['open']:
            # Allow but warn - low can be higher than open in some cases
            pass
        return v


class VolatilityResponse(BaseModel):
    volatility: float = Field(..., ge=0, description="Volatility must be non-negative")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Prediction confidence")
    model_version: str


class TradeRequest(BaseModel):
    symbol: str = Field(..., regex=r"^[A-Z]{3,6}-[A-Z]{3,6}$", description="Symbol format: BTC-USD")
    trade_type: str = Field(..., regex=r"^(buy|sell)$")
    order_type: str = Field(..., regex=r"^(market|limit|stop)$")
    quantity: Decimal = Field(..., gt=0, description="Quantity must be positive")
    price: Optional[Decimal] = Field(None, gt=0, description="Price for limit/stop orders")
    
    @validator('price')
    def validate_price_for_order_type(cls, v, values):
        if values.get('order_type') in ['limit', 'stop'] and v is None:
            raise ValueError('Price is required for limit and stop orders')
        return v


class TradeResponse(BaseModel):
    trade_id: str
    symbol: str
    trade_type: str
    order_type: str
    quantity: Decimal
    price: Decimal
    total_value: Decimal
    fees: Decimal
    status: str
    blockchain_tx_hash: Optional[str]
    created_at: datetime
    executed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class PositionResponse(BaseModel):
    position_id: str
    symbol: str
    position_type: str
    size: Decimal
    entry_price: Decimal
    current_price: Optional[Decimal]
    liquidation_price: Optional[Decimal]
    margin_requirement: Decimal
    unrealized_pnl: Decimal
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class PositionHealthResponse(BaseModel):
    address: str
    positions: List[PositionResponse]
    total_margin_used: Decimal
    total_margin_available: Decimal
    health_ratio: float = Field(..., description="Margin health ratio (available/used)")
    liquidation_risk: str = Field(..., description="Risk level: low, medium, high, critical")


# Market data schemas
class MarketDataResponse(BaseModel):
    symbol: str
    timestamp: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    volatility: Optional[Decimal]
    
    class Config:
        from_attributes = True


# Error schemas
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationErrorResponse(BaseModel):
    error: str = "validation_error"
    message: str = "Request validation failed"
    details: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# API Key schemas
class APIKeyCreate(BaseModel):
    key_name: str = Field(..., min_length=1, max_length=100)
    permissions: List[str] = Field(default=["read"])
    expires_in_days: Optional[int] = Field(None, gt=0, le=365)


class APIKeyResponse(BaseModel):
    key_id: str
    key_name: str
    key_value: str  # Only returned on creation
    permissions: List[str]
    is_active: bool
    expires_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True


# Audit log schemas
class AuditLogResponse(BaseModel):
    log_id: str
    action: str
    resource_type: Optional[str]
    resource_id: Optional[str]
    ip_address: Optional[str]
    status: str
    timestamp: datetime
    
    class Config:
        from_attributes = True


# Health check schemas
class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    services: Dict[str, str]  # service_name -> status

