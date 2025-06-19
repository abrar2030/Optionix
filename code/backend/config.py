"""
Configuration management for Optionix backend.
Handles all environment variables and application settings.
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "Optionix API"
    app_version: str = "2.0.0"
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # Database
    database_url: str
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    
    # Blockchain
    ethereum_provider_url: str
    ethereum_network_id: int = 1  # 1 for mainnet, 3 for ropsten, etc.
    futures_contract_address: str
    
    # ML Model
    model_path: str = "/app/models/volatility_model.h5"
    model_version: str = "1.0.0"
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # CORS
    cors_origins: list = ["*"]  # In production, specify exact origins
    
    # Financial
    max_position_size: float = 1000000.0  # Maximum position size in USD
    min_position_size: float = 100.0      # Minimum position size in USD
    liquidation_threshold: float = 0.8    # 80% of initial margin
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

