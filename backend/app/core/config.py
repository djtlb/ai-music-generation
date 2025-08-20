from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    """Application settings with Supabase integration"""
    
    # App info
    PROJECT_NAME: str = "AI Music Generation API"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = os.getenv("ENV", "production")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Supabase Configuration
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
    
    # Database (Supabase PostgreSQL)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./music_gen.db")
    
    # Security
    API_KEY: str = os.getenv("API_KEY", "ai-music-gen-2024-secure-key")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super-secure-secret-key")
    
    # CORS for web deployment
    CORS_ORIGINS: List[str] = [
        "https://*.hostinger.com",
        "https://*.hostingerapp.com", 
        "https://*.netlify.app",
        "https://*.vercel.app",
        "https://your-domain.com",
        "http://localhost:3000",
        "*"  # Allow all for development
    ]
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "memory://")
    
    # File limits
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", 10485760))  # 10MB
    MAX_GENERATION_TIME: int = int(os.getenv("MAX_GENERATION_TIME", 60))  # 1 minute
    
    # Server config
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    WORKERS: int = int(os.getenv("WORKERS", 1))
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
