"""
Configuration settings for the AI Music Generation Platform
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""
    
    # App Info
    app_name: str = "AI Music Generation Platform"
    app_version: str = "2.0.0"
    debug: bool = False
    environment: str = "development"  # development | staging | production
    log_level: str = "INFO"
    fast_dev: bool = False  # enables dev-only conveniences (dev token endpoint, relaxed CORS)
    environment: str = "development"  # development | staging | production
    log_level: str = "INFO"
    fast_dev: bool = False  # enables dev-only conveniences (dev token endpoint, relaxed CORS)
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Database
    database_url: str = "postgresql+asyncpg://user:password@localhost/aimusicgen"
    database_echo: bool = False
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    
    # Security
    secret_key: str = "your-super-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174", 
        "http://localhost:5175",
        "http://localhost:5176",
        "https://aimusicgen.com",
        "https://app.aimusicgen.com"
    ]
    
    # File Storage
    upload_directory: str = "uploads"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_audio_formats: List[str] = ["mp3", "wav", "flac", "m4a"]
    
    # AI Models
    model_cache_dir: str = "models"
    huggingface_token: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # Audio Processing
    sample_rate: int = 44100
    audio_output_dir: str = "generated_audio"
    max_audio_duration: int = 600  # 10 minutes
    
    # Payment Processing
    stripe_secret_key: Optional[str] = None
    stripe_publishable_key: Optional[str] = None
    paypal_client_id: Optional[str] = None
    paypal_client_secret: Optional[str] = None
    
    # Blockchain
    ethereum_rpc_url: Optional[str] = None
    web3_provider_key: Optional[str] = None
    nft_contract_address: Optional[str] = None
    
    # Email Service
    sendgrid_api_key: Optional[str] = None
    from_email: str = "noreply@aimusicgen.com"
    
    # Analytics
    mixpanel_token: Optional[str] = None
    amplitude_api_key: Optional[str] = None
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Subscription Plans
    free_tier_limit: int = 3  # songs per month
    pro_tier_limit: int = 100
    enterprise_tier_limit: int = -1  # unlimited
    
    # Feature Flags
    enable_blockchain: bool = True
    enable_collaboration: bool = True
    enable_marketplace: bool = True
    enable_enterprise: bool = True
    
    class Config:
        env_file = ".env"
        env_prefix = "AIMUSIC_"

# Global settings instance
_settings = None

def get_settings() -> Settings:
    """Get application settings"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
