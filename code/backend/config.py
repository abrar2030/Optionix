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

