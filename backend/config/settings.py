"""Application configuration.

Adds backward-compatible support for both prefixed (AIMUSIC_*) and legacy
unprefixed env vars (FAST_DEV, PERSIST_ENABLED / PERSIST, SECRET_KEY, etc.).
Prefixed names win; legacy names are only used if the prefixed variant is
absent. This helps local developers who set FAST_DEV=1 out of habit.
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


# --- Environment normalization (run before Settings instantiation) ------------
_FALLBACK_ENV_MAP = {
    # legacy_name: canonical_prefixed_name
    "FAST_DEV": "AIMUSIC_FAST_DEV",
    "PERSIST_ENABLED": "AIMUSIC_PERSIST_ENABLED",
    "PERSIST": "AIMUSIC_PERSIST_ENABLED",
    "SECRET_KEY": "AIMUSIC_SECRET_KEY",
    "DATABASE_URL": "AIMUSIC_DATABASE_URL",
    "REDIS_URL": "AIMUSIC_REDIS_URL",
}

for legacy, canonical in _FALLBACK_ENV_MAP.items():
    if legacy in os.environ and canonical not in os.environ:
        os.environ[canonical] = os.environ[legacy]


def _coerce_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on", "y"}

class Settings(BaseSettings):
    """Application settings (values loaded from environment)."""

    # App Info
    app_name: str = "AI Music Generation Platform"
    app_version: str = "2.0.0"
    debug: bool = False
    environment: str = "development"  # development | staging | production
    log_level: str = "INFO"
    fast_dev: bool = False  # dev conveniences
    persist_enabled: bool = False  # DB persistence toggle
    
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

    # Post-init normalization for booleans that may come in as strings
    def model_post_init(self, __context):  # type: ignore[override]
        # Preserve original explicit values but allow legacy raw env overrides
        self.fast_dev = bool(self.fast_dev) or _coerce_bool(os.getenv("AIMUSIC_FAST_DEV"))
        self.persist_enabled = bool(self.persist_enabled) or _coerce_bool(os.getenv("AIMUSIC_PERSIST_ENABLED"))

# Global settings instance
_settings = None

def get_settings() -> Settings:
    """Singleton accessor for settings (instantiated once)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
