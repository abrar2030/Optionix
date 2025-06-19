"""
Enhanced configuration management for Optionix platform.
Includes security, compliance, and financial standards settings.
"""
import os
from typing import List, Optional
from pydantic import BaseSettings, validator
from decimal import Decimal


class Settings(BaseSettings):
    """Enhanced application settings with comprehensive configuration"""
    
    # Application settings
    app_name: str = "Optionix Enhanced Trading Platform"
    app_version: str = "2.0.0-enhanced"
    debug: bool = False
    testing: bool = False
    environment: str = "production"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Database settings
    database_url: str = "postgresql://user:password@localhost/optionix_enhanced"
    database_pool_size: int = 20
    database_max_overflow: int = 30
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600
    
    # Redis settings
    redis_url: str = "redis://localhost:6379/0"
    redis_session_db: int = 1
    redis_cache_db: int = 2
    redis_rate_limit_db: int = 3
    
    # Security settings
    secret_key: str = "your-super-secret-key-change-in-production"
    encryption_key: str = "your-encryption-key-32-bytes-long"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    session_expire_hours: int = 24
    
    # Password policy
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special: bool = True
    password_max_age_days: int = 90
    password_history_count: int = 5
    
    # Account lockout policy
    max_failed_login_attempts: int = 5
    account_lockout_duration_minutes: int = 30
    failed_attempt_window_minutes: int = 15
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst_size: int = 200
    rate_limit_window_minutes: int = 1
    
    # API settings
    api_key_length: int = 43
    api_key_prefix: str = "ok_"
    api_rate_limit_per_hour: int = 10000
    
    # CORS settings
    cors_origins: List[str] = ["http://localhost:3000", "https://optionix.com"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_allow_headers: List[str] = ["*"]
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    log_max_size: int = 10485760  # 10MB
    log_backup_count: int = 5
    
    # Audit settings
    audit_log_retention_days: int = 2555  # 7 years
    audit_log_encryption: bool = True
    audit_log_integrity_check: bool = True
    
    # Compliance settings
    kyc_required: bool = True
    kyc_document_retention_years: int = 7
    aml_monitoring_enabled: bool = True
    sanctions_screening_enabled: bool = True
    sanctions_check_frequency_days: int = 30
    
    # Financial standards
    sox_compliance_enabled: bool = True
    mifid_ii_reporting_enabled: bool = True
    basel_iii_monitoring_enabled: bool = True
    dodd_frank_compliance_enabled: bool = True
    
    # Risk management
    default_leverage_limit: Decimal = Decimal("10.0")
    max_leverage_limit: Decimal = Decimal("100.0")
    default_risk_limit: Decimal = Decimal("100000.0")
    var_confidence_level: Decimal = Decimal("0.95")
    var_time_horizon_days: int = 1
    
    # Trading settings
    trading_enabled: bool = True
    trading_hours_start: str = "00:00"
    trading_hours_end: str = "23:59"
    trading_timezone: str = "UTC"
    max_order_size: Decimal = Decimal("1000000.0")
    min_order_size: Decimal = Decimal("0.001")
    
    # Fee structure
    trading_fee_percentage: Decimal = Decimal("0.001")  # 0.1%
    withdrawal_fee_flat: Decimal = Decimal("5.0")
    deposit_fee_percentage: Decimal = Decimal("0.0")
    
    # Blockchain settings
    ethereum_rpc_url: str = "https://mainnet.infura.io/v3/your-project-id"
    ethereum_chain_id: int = 1
    gas_price_gwei: int = 20
    gas_limit: int = 21000
    
    # Data protection (GDPR)
    data_retention_default_years: int = 7
    data_anonymization_enabled: bool = True
    data_export_enabled: bool = True
    data_deletion_enabled: bool = True
    consent_management_enabled: bool = True
    
    # Monitoring and alerting
    monitoring_enabled: bool = True
    alerting_enabled: bool = True
    health_check_interval_seconds: int = 30
    performance_monitoring_enabled: bool = True
    
    # External services
    email_service_enabled: bool = True
    sms_service_enabled: bool = True
    notification_service_enabled: bool = True
    
    # Backup and disaster recovery
    backup_enabled: bool = True
    backup_frequency_hours: int = 6
    backup_retention_days: int = 30
    disaster_recovery_enabled: bool = True
    
    # Performance settings
    database_connection_pool_size: int = 20
    cache_ttl_seconds: int = 300
    session_cleanup_interval_minutes: int = 60
    
    # Security headers
    security_headers_enabled: bool = True
    hsts_max_age_seconds: int = 31536000  # 1 year
    content_security_policy: str = "default-src 'self'"
    
    # File upload settings
    max_file_size_mb: int = 10
    allowed_file_types: List[str] = ["pdf", "jpg", "jpeg", "png", "doc", "docx"]
    file_scan_enabled: bool = True
    
    # Internationalization
    default_language: str = "en"
    supported_languages: List[str] = ["en", "es", "fr", "de", "zh"]
    default_timezone: str = "UTC"
    
    # Feature flags
    mfa_required_for_trading: bool = False
    enhanced_kyc_required: bool = True
    real_time_monitoring: bool = True
    advanced_analytics: bool = True
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError('Secret key must be at least 32 characters long')
        return v
    
    @validator('encryption_key')
    def validate_encryption_key(cls, v):
        if len(v) != 32:
            raise ValueError('Encryption key must be exactly 32 characters long')
        return v
    
    @validator('environment')
    def validate_environment(cls, v):
        if v not in ['development', 'staging', 'production', 'testing']:
            raise ValueError('Environment must be one of: development, staging, production, testing')
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class SecuritySettings:
    """Security-specific settings and constants"""
    
    # Encryption algorithms
    SYMMETRIC_ALGORITHM = "AES-256-GCM"
    ASYMMETRIC_ALGORITHM = "RSA-4096"
    HASH_ALGORITHM = "SHA-256"
    
    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
    }
    
    # Content Security Policy
    CSP_DIRECTIVES = {
        "default-src": "'self'",
        "script-src": "'self' 'unsafe-inline'",
        "style-src": "'self' 'unsafe-inline'",
        "img-src": "'self' data: https:",
        "font-src": "'self'",
        "connect-src": "'self'",
        "frame-ancestors": "'none'",
        "base-uri": "'self'",
        "form-action": "'self'"
    }
    
    # Password validation patterns
    PASSWORD_PATTERNS = {
        "uppercase": r"[A-Z]",
        "lowercase": r"[a-z]",
        "numbers": r"\d",
        "special": r"[!@#$%^&*(),.?\":{}|<>]"
    }
    
    # Sensitive data patterns for detection
    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
    }


class ComplianceSettings:
    """Compliance-specific settings and constants"""
    
    # KYC risk levels
    KYC_RISK_LEVELS = {
        "low": {"score_range": (0, 30), "review_required": False},
        "medium": {"score_range": (31, 60), "review_required": True},
        "high": {"score_range": (61, 80), "review_required": True},
        "critical": {"score_range": (81, 100), "review_required": True}
    }
    
    # High-risk countries (simplified list)
    HIGH_RISK_COUNTRIES = [
        "AF", "BY", "CF", "CD", "CU", "ER", "GN", "GW", "HT", "IR",
        "IQ", "LB", "LY", "ML", "MM", "NI", "KP", "SO", "SS", "SD",
        "SY", "VE", "YE", "ZW"
    ]
    
    # Sanctions lists to check
    SANCTIONS_LISTS = [
        "OFAC_SDN",  # Office of Foreign Assets Control Specially Designated Nationals
        "UN_SANCTIONS",  # United Nations Sanctions List
        "EU_SANCTIONS",  # European Union Sanctions List
        "UK_SANCTIONS",  # UK HM Treasury Sanctions List
        "FATF_GREYLIST"  # Financial Action Task Force Grey List
    ]
    
    # Transaction monitoring thresholds
    TRANSACTION_THRESHOLDS = {
        "large_transaction": Decimal("10000.0"),
        "velocity_daily": Decimal("50000.0"),
        "velocity_weekly": Decimal("200000.0"),
        "velocity_monthly": Decimal("500000.0"),
        "unusual_pattern_score": 75
    }
    
    # Regulatory reporting requirements
    REGULATORY_REPORTS = {
        "mifid_ii": {
            "frequency": "daily",
            "threshold": Decimal("15000.0"),  # EUR
            "deadline_hours": 24
        },
        "dodd_frank": {
            "frequency": "daily",
            "threshold": Decimal("25000.0"),  # USD
            "deadline_hours": 24
        },
        "cftc": {
            "frequency": "daily",
            "threshold": Decimal("50000.0"),  # USD
            "deadline_hours": 24
        }
    }


class FinancialSettings:
    """Financial standards and risk management settings"""
    
    # Basel III requirements
    BASEL_III_REQUIREMENTS = {
        "leverage_ratio_minimum": Decimal("0.03"),  # 3%
        "liquidity_coverage_ratio": Decimal("1.0"),  # 100%
        "net_stable_funding_ratio": Decimal("1.0"),  # 100%
        "capital_adequacy_ratio": Decimal("0.08"),  # 8%
        "tier1_capital_ratio": Decimal("0.06"),  # 6%
        "common_equity_tier1_ratio": Decimal("0.045")  # 4.5%
    }
    
    # SOX controls
    SOX_CONTROLS = {
        "segregation_of_duties": True,
        "authorization_matrix": {
            "trade_execution": ["trader", "senior_trader"],
            "position_modification": ["risk_manager", "senior_trader"],
            "account_creation": ["admin", "compliance_officer"],
            "large_transactions": ["senior_trader", "risk_manager"],
            "system_configuration": ["admin", "system_admin"]
        },
        "audit_retention_years": 7,
        "control_testing_frequency_days": 90,
        "financial_close_controls": True,
        "it_general_controls": True
    }
    
    # Risk limits by account type
    RISK_LIMITS = {
        "standard": {
            "max_leverage": Decimal("10.0"),
            "max_position_size": Decimal("100000.0"),
            "daily_loss_limit": Decimal("10000.0"),
            "var_limit": Decimal("5000.0")
        },
        "premium": {
            "max_leverage": Decimal("20.0"),
            "max_position_size": Decimal("500000.0"),
            "daily_loss_limit": Decimal("50000.0"),
            "var_limit": Decimal("25000.0")
        },
        "institutional": {
            "max_leverage": Decimal("50.0"),
            "max_position_size": Decimal("10000000.0"),
            "daily_loss_limit": Decimal("500000.0"),
            "var_limit": Decimal("250000.0")
        }
    }
    
    # Market data requirements
    MARKET_DATA_REQUIREMENTS = {
        "price_update_frequency_ms": 100,
        "volatility_calculation_window_days": 30,
        "correlation_calculation_window_days": 90,
        "var_calculation_confidence": Decimal("0.95"),
        "stress_test_scenarios": 10
    }


# Global settings instance
settings = Settings()

# Environment-specific overrides
if settings.environment == "development":
    settings.debug = True
    settings.log_level = "DEBUG"
    settings.cors_origins = ["http://localhost:3000", "http://localhost:3001"]

elif settings.environment == "testing":
    settings.testing = True
    settings.database_url = "sqlite:///./test.db"
    settings.redis_url = "redis://localhost:6379/15"  # Use different DB for tests

elif settings.environment == "production":
    settings.debug = False
    settings.log_level = "WARNING"
    # Production settings should be set via environment variables

