"""
Database models for Optionix platform.
Defines all database tables and relationships using SQLAlchemy.
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, Numeric, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()


class User(Base):
    """User account information"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    kyc_status = Column(String(50), default="pending")  # pending, approved, rejected
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    accounts = relationship("Account", back_populates="user")
    trades = relationship("Trade", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")


class Account(Base):
    """User trading accounts"""
    __tablename__ = "accounts"
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    ethereum_address = Column(String(42), unique=True, index=True)
    account_type = Column(String(50), default="standard")  # standard, premium, institutional
    balance_usd = Column(Numeric(precision=18, scale=8), default=0.0)
    margin_available = Column(Numeric(precision=18, scale=8), default=0.0)
    margin_used = Column(Numeric(precision=18, scale=8), default=0.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="accounts")
    positions = relationship("Position", back_populates="account")
    trades = relationship("Trade", back_populates="account")


class Position(Base):
    """Trading positions"""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    position_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)
    symbol = Column(String(20), nullable=False)  # e.g., "BTC-USD", "ETH-USD"
    position_type = Column(String(10), nullable=False)  # "long" or "short"
    size = Column(Numeric(precision=18, scale=8), nullable=False)
    entry_price = Column(Numeric(precision=18, scale=8), nullable=False)
    current_price = Column(Numeric(precision=18, scale=8))
    liquidation_price = Column(Numeric(precision=18, scale=8))
    margin_requirement = Column(Numeric(precision=18, scale=8), nullable=False)
    unrealized_pnl = Column(Numeric(precision=18, scale=8), default=0.0)
    status = Column(String(20), default="open")  # open, closed, liquidated
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    account = relationship("Account", back_populates="positions")
    trades = relationship("Trade", back_populates="position")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_position_account_symbol', 'account_id', 'symbol'),
        Index('idx_position_status', 'status'),
    )


class Trade(Base):
    """Trade execution records"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    trade_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)
    position_id = Column(Integer, ForeignKey("positions.id"), nullable=True)
    symbol = Column(String(20), nullable=False)
    trade_type = Column(String(10), nullable=False)  # "buy", "sell"
    order_type = Column(String(20), nullable=False)  # "market", "limit", "stop"
    quantity = Column(Numeric(precision=18, scale=8), nullable=False)
    price = Column(Numeric(precision=18, scale=8), nullable=False)
    total_value = Column(Numeric(precision=18, scale=8), nullable=False)
    fees = Column(Numeric(precision=18, scale=8), default=0.0)
    status = Column(String(20), default="pending")  # pending, executed, cancelled, failed
    blockchain_tx_hash = Column(String(66))  # Ethereum transaction hash
    executed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="trades")
    account = relationship("Account", back_populates="trades")
    position = relationship("Position", back_populates="trades")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_trade_user_symbol', 'user_id', 'symbol'),
        Index('idx_trade_status', 'status'),
        Index('idx_trade_created_at', 'created_at'),
    )


class AuditLog(Base):
    """Audit trail for all critical operations"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    log_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)  # e.g., "login", "trade_executed", "position_liquidated"
    resource_type = Column(String(50))  # e.g., "user", "trade", "position"
    resource_id = Column(String(36))
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    request_data = Column(Text)  # JSON string of request data
    response_data = Column(Text)  # JSON string of response data
    status = Column(String(20))  # "success", "failure", "error"
    error_message = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_audit_user_action', 'user_id', 'action'),
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
    )


class MarketData(Base):
    """Market data for volatility prediction and analysis"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    open_price = Column(Numeric(precision=18, scale=8), nullable=False)
    high_price = Column(Numeric(precision=18, scale=8), nullable=False)
    low_price = Column(Numeric(precision=18, scale=8), nullable=False)
    close_price = Column(Numeric(precision=18, scale=8), nullable=False)
    volume = Column(Numeric(precision=18, scale=8), nullable=False)
    volatility = Column(Numeric(precision=10, scale=6))  # Predicted or calculated volatility
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_market_data_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_market_data_timestamp', 'timestamp'),
    )


class APIKey(Base):
    """API keys for external integrations"""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key_name = Column(String(100), nullable=False)
    key_hash = Column(String(255), nullable=False)  # Hashed API key
    permissions = Column(Text)  # JSON string of permissions
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_api_key_user', 'user_id'),
        Index('idx_api_key_active', 'is_active'),
    )

