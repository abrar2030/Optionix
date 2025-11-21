"""
Enhanced database models for Optionix platform.
Includes all models for security, compliance, and financial standards.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class User(Base):
    """Enhanced user account information with security features"""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        String(36), unique=True, index=True, default=lambda: str(uuid.uuid4())
    )
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)

    # Security fields
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_locked = Column(Boolean, default=False)
    failed_login_attempts = Column(Integer, default=0)
    last_login = Column(DateTime, nullable=True)
    password_changed_at = Column(DateTime, default=datetime.utcnow)

    # Role-based access control
    role = Column(
        String(50), default="trader"
    )  # admin, trader, viewer, compliance_officer, risk_manager, api_user
    permissions = Column(Text, nullable=True)  # JSON array of additional permissions

    # Multi-factor authentication
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String(255), nullable=True)  # Encrypted TOTP secret
    mfa_backup_codes = Column(Text, nullable=True)  # Encrypted JSON array

    # KYC and compliance
    kyc_status = Column(
        String(50), default="pending"
    )  # pending, approved, rejected, under_review
    kyc_level = Column(String(20), default="basic")  # basic, enhanced, premium
    risk_score = Column(Integer, default=0)
    compliance_status = Column(String(50), default="pending")

    # Data protection
    data_retention_consent = Column(Boolean, default=False)
    marketing_consent = Column(Boolean, default=False)
    data_processing_consent = Column(Boolean, default=False)
    consent_date = Column(DateTime, nullable=True)

    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    updated_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Relationships
    accounts = relationship("Account", back_populates="user")
    trades = relationship("Trade", back_populates="user")
    audit_logs = relationship(
        "AuditLog", back_populates="user", foreign_keys="AuditLog.user_id"
    )
    api_keys = relationship("APIKey", back_populates="user")
    kyc_documents = relationship("KYCDocument", back_populates="user")
    sanctions_checks = relationship("SanctionsCheck", back_populates="user")

    # Indexes
    __table_args__ = (
        Index("idx_user_email_active", "email", "is_active"),
        Index("idx_user_kyc_status", "kyc_status"),
        Index("idx_user_role", "role"),
    )


class Account(Base):
    """Enhanced user trading accounts with compliance features"""

    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(
        String(36), unique=True, index=True, default=lambda: str(uuid.uuid4())
    )
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Account details
    ethereum_address = Column(String(42), unique=True, index=True)
    account_type = Column(
        String(50), default="standard"
    )  # standard, premium, institutional, demo
    account_status = Column(
        String(20), default="active"
    )  # active, suspended, closed, frozen

    # Financial data (encrypted)
    balance_usd = Column(Numeric(precision=18, scale=8), default=0.0)
    margin_available = Column(Numeric(precision=18, scale=8), default=0.0)
    margin_used = Column(Numeric(precision=18, scale=8), default=0.0)
    margin_requirement = Column(Numeric(precision=18, scale=8), default=0.0)

    # Risk management
    max_leverage = Column(Numeric(precision=5, scale=2), default=10.0)
    risk_limit = Column(Numeric(precision=18, scale=8), default=100000.0)
    daily_loss_limit = Column(Numeric(precision=18, scale=8), default=10000.0)

    # Compliance
    aml_status = Column(String(20), default="pending")
    sanctions_checked = Column(Boolean, default=False)
    last_sanctions_check = Column(DateTime, nullable=True)

    # Activity tracking
    is_active = Column(Boolean, default=True)
    last_activity = Column(DateTime, nullable=True)

    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="accounts")
    positions = relationship("Position", back_populates="account")
    trades = relationship("Trade", back_populates="account")
    financial_audit_logs = relationship("FinancialAuditLog", back_populates="account")

    # Indexes
    __table_args__ = (
        Index("idx_account_user_status", "user_id", "account_status"),
        Index("idx_account_ethereum", "ethereum_address"),
    )


class Position(Base):
    """Enhanced trading positions with risk metrics"""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    position_id = Column(
        String(36), unique=True, index=True, default=lambda: str(uuid.uuid4())
    )
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)

    # Position details
    symbol = Column(String(20), nullable=False)
    position_type = Column(String(10), nullable=False)  # "long" or "short"
    size = Column(Numeric(precision=18, scale=8), nullable=False)

    # Pricing
    entry_price = Column(Numeric(precision=18, scale=8), nullable=False)
    current_price = Column(Numeric(precision=18, scale=8))
    mark_price = Column(Numeric(precision=18, scale=8))
    liquidation_price = Column(Numeric(precision=18, scale=8))

    # Risk metrics
    margin_requirement = Column(Numeric(precision=18, scale=8), nullable=False)
    maintenance_margin = Column(Numeric(precision=18, scale=8), nullable=False)
    unrealized_pnl = Column(Numeric(precision=18, scale=8), default=0.0)
    realized_pnl = Column(Numeric(precision=18, scale=8), default=0.0)

    # Greeks (for options)
    delta = Column(Numeric(precision=10, scale=6), nullable=True)
    gamma = Column(Numeric(precision=10, scale=6), nullable=True)
    theta = Column(Numeric(precision=10, scale=6), nullable=True)
    vega = Column(Numeric(precision=10, scale=6), nullable=True)
    rho = Column(Numeric(precision=10, scale=6), nullable=True)

    # Status and lifecycle
    status = Column(String(20), default="open")  # open, closed, liquidated, expired
    auto_close_enabled = Column(Boolean, default=True)
    stop_loss = Column(Numeric(precision=18, scale=8), nullable=True)
    take_profit = Column(Numeric(precision=18, scale=8), nullable=True)

    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    closed_at = Column(DateTime, nullable=True)

    # Relationships
    account = relationship("Account", back_populates="positions")
    trades = relationship("Trade", back_populates="position")

    # Indexes
    __table_args__ = (
        Index("idx_position_account_symbol", "account_id", "symbol"),
        Index("idx_position_status", "status"),
        Index("idx_position_symbol_status", "symbol", "status"),
    )


class Trade(Base):
    """Enhanced trade execution records with compliance tracking"""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    trade_id = Column(
        String(36), unique=True, index=True, default=lambda: str(uuid.uuid4())
    )
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)
    position_id = Column(Integer, ForeignKey("positions.id"), nullable=True)

    # Trade details
    symbol = Column(String(20), nullable=False)
    trade_type = Column(String(10), nullable=False)  # "buy", "sell"
    order_type = Column(
        String(20), nullable=False
    )  # "market", "limit", "stop", "stop_limit"

    # Quantities and pricing
    quantity = Column(Numeric(precision=18, scale=8), nullable=False)
    price = Column(Numeric(precision=18, scale=8), nullable=False)
    executed_price = Column(Numeric(precision=18, scale=8), nullable=True)
    total_value = Column(Numeric(precision=18, scale=8), nullable=False)

    # Fees and costs
    fees = Column(Numeric(precision=18, scale=8), default=0.0)
    commission = Column(Numeric(precision=18, scale=8), default=0.0)
    spread = Column(Numeric(precision=18, scale=8), default=0.0)
    slippage = Column(Numeric(precision=18, scale=8), default=0.0)

    # Execution details
    status = Column(
        String(20), default="pending"
    )  # pending, executed, cancelled, failed, rejected
    execution_venue = Column(String(100), default="internal")
    execution_algorithm = Column(String(50), nullable=True)

    # Blockchain integration
    blockchain_tx_hash = Column(String(66))  # Ethereum transaction hash
    blockchain_status = Column(String(20), default="pending")
    gas_used = Column(Numeric(precision=18, scale=0), nullable=True)
    gas_price = Column(Numeric(precision=18, scale=0), nullable=True)

    # Compliance and risk
    compliance_checked = Column(Boolean, default=False)
    compliance_status = Column(String(20), default="pending")
    risk_checked = Column(Boolean, default=False)
    risk_score = Column(Integer, default=0)

    # MiFID II fields
    mifid_reportable = Column(Boolean, default=False)
    mifid_reported = Column(Boolean, default=False)
    mifid_report_id = Column(String(100), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    executed_at = Column(DateTime(timezone=True), nullable=True)
    settled_at = Column(DateTime(timezone=True), nullable=True)

    # Client information
    client_order_id = Column(String(100), nullable=True)
    source_system = Column(String(50), default="web")  # web, mobile, api

    # Relationships
    user = relationship("User", back_populates="trades")
    account = relationship("Account", back_populates="trades")
    position = relationship("Position", back_populates="trades")
    transaction_monitoring = relationship(
        "TransactionMonitoring", back_populates="trade"
    )

    # Indexes
    __table_args__ = (
        Index("idx_trade_user_symbol", "user_id", "symbol"),
        Index("idx_trade_status", "status"),
        Index("idx_trade_created_at", "created_at"),
        Index("idx_trade_account_status", "account_id", "status"),
        Index("idx_trade_compliance", "compliance_status"),
    )


class AuditLog(Base):
    """Enhanced audit trail for all critical operations"""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    log_id = Column(
        String(36), unique=True, index=True, default=lambda: str(uuid.uuid4())
    )
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Action details
    action = Column(String(100), nullable=False)
    action_category = Column(
        String(50), nullable=False
    )  # authentication, trading, compliance, admin
    resource_type = Column(String(50))
    resource_id = Column(String(36))

    # Request/Response data
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    session_id = Column(String(100), nullable=True)
    request_data = Column(Text)  # JSON string of request data
    response_data = Column(Text)  # JSON string of response data

    # Result and status
    status = Column(String(20))  # "success", "failure", "error"
    status_code = Column(Integer, nullable=True)
    error_message = Column(Text)

    # Security context
    authentication_method = Column(String(50), nullable=True)
    authorization_level = Column(String(50), nullable=True)
    security_context = Column(Text, nullable=True)  # JSON

    # Compliance
    regulation_type = Column(String(50), nullable=True)
    compliance_impact = Column(String(20), default="low")  # low, medium, high, critical

    # Timing
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    processing_time_ms = Column(Integer, nullable=True)

    # Data integrity
    data_hash = Column(String(64), nullable=True)  # SHA-256 hash

    # Relationships
    user = relationship("User", back_populates="audit_logs", foreign_keys=[user_id])

    # Indexes
    __table_args__ = (
        Index("idx_audit_user_action", "user_id", "action"),
        Index("idx_audit_timestamp", "timestamp"),
        Index("idx_audit_resource", "resource_type", "resource_id"),
        Index("idx_audit_category", "action_category"),
        Index("idx_audit_compliance", "regulation_type", "compliance_impact"),
    )


class APIKey(Base):
    """Enhanced API keys for external integrations"""

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key_id = Column(
        String(36), unique=True, index=True, default=lambda: str(uuid.uuid4())
    )
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Key details
    key_name = Column(String(100), nullable=False)
    key_hash = Column(String(255), nullable=False)  # Hashed API key
    key_prefix = Column(
        String(10), nullable=False
    )  # First few characters for identification

    # Permissions and restrictions
    permissions = Column(Text)  # JSON string of permissions
    ip_whitelist = Column(Text, nullable=True)  # JSON array of allowed IPs
    rate_limit = Column(Integer, default=1000)  # Requests per hour

    # Status and lifecycle
    is_active = Column(Boolean, default=True)
    is_revoked = Column(Boolean, default=False)
    revocation_reason = Column(String(255), nullable=True)

    # Usage tracking
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    last_used_ip = Column(String(45), nullable=True)
    usage_count = Column(Integer, default=0)

    # Expiration
    expires_at = Column(DateTime(timezone=True), nullable=True)
    auto_rotate = Column(Boolean, default=False)
    rotation_interval_days = Column(Integer, default=90)

    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    revoked_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Relationships
    user = relationship("User", back_populates="api_keys", foreign_keys=[user_id])

    # Indexes
    __table_args__ = (
        Index("idx_api_key_user", "user_id"),
        Index("idx_api_key_active", "is_active"),
        Index("idx_api_key_hash", "key_hash"),
    )


class KYCDocument(Base):
    """KYC document storage and verification"""

    __tablename__ = "kyc_documents"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(
        String(36), unique=True, index=True, default=lambda: str(uuid.uuid4())
    )
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Document details
    document_type = Column(
        String(50), nullable=False
    )  # passport, national_id, drivers_license
    document_number = Column(String(255), nullable=False)  # Encrypted
    document_country = Column(String(3), nullable=False)
    document_expiry = Column(DateTime, nullable=True)

    # Verification
    verification_status = Column(
        String(20), default="pending"
    )  # pending, verified, rejected, expired
    verification_date = Column(DateTime, nullable=True)
    verification_method = Column(
        String(50), nullable=True
    )  # manual, automated, third_party
    verification_score = Column(Numeric(precision=5, scale=2), nullable=True)

    # Risk assessment
    risk_score = Column(Integer, default=0)
    risk_factors = Column(Text, nullable=True)  # JSON array

    # Document integrity
    document_hash = Column(String(64), nullable=False)  # SHA-256 hash
    file_path = Column(String(500), nullable=True)  # Encrypted file path
    file_size = Column(Integer, nullable=True)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    verified_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Relationships
    user = relationship("User", back_populates="kyc_documents", foreign_keys=[user_id])

    # Indexes
    __table_args__ = (
        Index("idx_kyc_user_status", "user_id", "verification_status"),
        Index("idx_kyc_document_type", "document_type"),
    )


class SanctionsCheck(Base):
    """Sanctions screening results"""

    __tablename__ = "sanctions_checks"

    id = Column(Integer, primary_key=True, index=True)
    check_id = Column(
        String(36), unique=True, index=True, default=lambda: str(uuid.uuid4())
    )
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Check details
    check_type = Column(String(50), nullable=False)  # name, address, entity, ongoing
    search_terms = Column(Text, nullable=False)  # JSON
    lists_checked = Column(Text, nullable=False)  # JSON array

    # Results
    matches_found = Column(Boolean, default=False)
    match_details = Column(Text, nullable=True)  # JSON
    false_positive = Column(Boolean, default=False)

    # Risk assessment
    risk_score = Column(Integer, default=0)
    confidence_score = Column(Numeric(precision=5, scale=2), nullable=True)

    # Scheduling
    checked_at = Column(DateTime, default=datetime.utcnow)
    next_check_due = Column(DateTime, nullable=False)
    check_frequency_days = Column(Integer, default=30)

    # Resolution
    resolution_status = Column(
        String(20), default="pending"
    )  # pending, cleared, escalated
    resolution_notes = Column(Text, nullable=True)
    resolved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship(
        "User", back_populates="sanctions_checks", foreign_keys=[user_id]
    )

    # Indexes
    __table_args__ = (
        Index("idx_sanctions_user_status", "user_id", "resolution_status"),
        Index("idx_sanctions_next_check", "next_check_due"),
        Index("idx_sanctions_matches", "matches_found"),
    )


class TransactionMonitoring(Base):
    """Transaction monitoring alerts and investigations"""

    __tablename__ = "transaction_monitoring"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(
        String(36), unique=True, index=True, default=lambda: str(uuid.uuid4())
    )
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)

    # Alert details
    alert_type = Column(String(100), nullable=False)
    alert_category = Column(
        String(50), nullable=False
    )  # threshold, pattern, velocity, behavioral
    alert_description = Column(Text, nullable=False)

    # Risk assessment
    risk_score = Column(Integer, nullable=False)
    risk_level = Column(String(20), nullable=False)  # low, medium, high, critical
    threshold_breached = Column(String(100), nullable=False)

    # Investigation
    alert_status = Column(
        String(20), default="open"
    )  # open, investigating, closed, false_positive
    assigned_to = Column(Integer, ForeignKey("users.id"), nullable=True)
    investigation_priority = Column(
        String(20), default="medium"
    )  # low, medium, high, urgent

    # Resolution
    resolution_notes = Column(Text, nullable=True)
    resolution_action = Column(String(100), nullable=True)
    escalated = Column(Boolean, default=False)
    escalation_reason = Column(Text, nullable=True)

    # Regulatory reporting
    sar_filed = Column(Boolean, default=False)
    sar_reference = Column(String(100), nullable=True)
    regulatory_reported = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    assigned_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    trade = relationship("Trade", back_populates="transaction_monitoring")
    assigned_user = relationship("User", foreign_keys=[assigned_to])

    # Indexes
    __table_args__ = (
        Index("idx_tm_user_status", "user_id", "alert_status"),
        Index("idx_tm_risk_level", "risk_level"),
        Index("idx_tm_assigned", "assigned_to", "alert_status"),
        Index("idx_tm_created", "created_at"),
    )


class FinancialAuditLog(Base):
    """Financial audit log for SOX compliance"""

    __tablename__ = "financial_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    audit_id = Column(String(100), unique=True, nullable=False)
    transaction_id = Column(String(100), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=True)

    # Financial data
    transaction_type = Column(String(50), nullable=False)
    amount = Column(Numeric(precision=18, scale=8), nullable=False)
    currency = Column(String(10), default="USD")

    # Audit trail
    previous_state = Column(Text, nullable=True)  # JSON
    new_state = Column(Text, nullable=False)  # JSON
    state_hash = Column(String(64), nullable=False)  # SHA-256

    # Compliance
    regulation_type = Column(String(50), nullable=False)
    compliance_status = Column(String(20), default="compliant")
    control_reference = Column(String(100), nullable=True)

    # Authorization
    authorized_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    authorization_level = Column(String(50), nullable=True)

    # Timestamps
    business_date = Column(DateTime, nullable=False)
    system_timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    account = relationship("Account", back_populates="financial_audit_logs")

    # Indexes
    __table_args__ = (
        Index("idx_financial_audit_transaction", "transaction_id"),
        Index("idx_financial_audit_user", "user_id"),
        Index("idx_financial_audit_date", "business_date"),
        Index("idx_financial_audit_regulation", "regulation_type"),
    )


class MarketData(Base):
    """Market data for volatility prediction and historical analysis"""

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)

    # OHLCV data
    open_price = Column(Numeric(precision=18, scale=8), nullable=False)
    high_price = Column(Numeric(precision=18, scale=8), nullable=False)
    low_price = Column(Numeric(precision=18, scale=8), nullable=False)
    close_price = Column(Numeric(precision=18, scale=8), nullable=False)
    volume = Column(Numeric(precision=18, scale=8), nullable=False)

    # Model prediction
    volatility = Column(Numeric(precision=10, scale=6), nullable=True)

    # Indexes
    __table_args__ = (
        Index("idx_market_data_symbol_time", "symbol", "timestamp", unique=True),
        Index("idx_market_data_time", "timestamp"),
    )
