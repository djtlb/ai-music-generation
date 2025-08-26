#!/usr/bin/env python3
"""
Setup script for AI Music Generation Backend
Creates all necessary directories and files to get the server running.
"""

import os
from pathlib import Path

def create_directory_structure():
    """Create the required directory structure."""
    directories = [
        "app",
        "app/core",
        "app/api",
        "app/api/endpoints", 
        "app/services",
        "app/models",
        "static",
        "generated_audio",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_init_files():
    """Create __init__.py files for Python packages."""
    init_files = [
        "app/__init__.py",
        "app/core/__init__.py",
        "app/api/__init__.py",
        "app/api/endpoints/__init__.py",
        "app/services/__init__.py",
        "app/models/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✅ Created: {init_file}")

def create_config_file():
    """Create the configuration file."""
    config_content = '''import os
from typing import List, Optional, Union
from pydantic import validator, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment-based configuration."""
    
    # Basic app info
    PROJECT_NAME: str = "AI Music Generation"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENV")
    DEBUG: bool = Field(default=True, env="DEBUG")
    API_V1_STR: str = "/api/v1"
    
    # Server settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=1, env="WORKERS")
    
    # Security
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    ALLOWED_HOSTS: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "https://localhost:3000", "http://127.0.0.1:3000"],
        env="CORS_ORIGINS"
    )
    
    # Database
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./test.db", env="DATABASE_URL")
    DB_POOL_SIZE: int = Field(default=10, env="DB_POOL_SIZE")
    DB_MAX_OVERFLOW: int = Field(default=20, env="DB_MAX_OVERFLOW")
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    REDIS_MAX_CONNECTIONS: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")
    
    # Monitoring
    SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")
    SENTRY_TRACES_SAMPLE_RATE: float = Field(default=0.1, env="SENTRY_TRACES_SAMPLE_RATE")
    
    # API Keys
    API_KEY: str = Field(default="test-api-key-123", env="API_KEY")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
'''
    
    with open("app/core/config.py", "w") as f:
        f.write(config_content)
    print("✅ Created: app/core/config.py")

def create_core_files():
    """Create core module files."""
    
    # Database module
    database_content = '''import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
import structlog

from app.core.config import settings

logger = structlog.get_logger()

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()


async def init_db():
    """Initialize database connection and create tables."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")


async def close_db():
    """Close database connections."""
    await engine.dispose()
    logger.info("Database connections closed")


async def get_db_session():
    """Get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
'''
    
    with open("app/core/database.py", "w") as f:
        f.write(database_content)
    print("✅ Created: app/core/database.py")
    
    # Redis module
    redis_content = '''import structlog
from app.core.config import settings

logger = structlog.get_logger()

redis_client = None


async def init_redis():
    """Initialize Redis connection."""
    global redis_client
    try:
        import redis.asyncio as redis
        redis_client = redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
        )
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}")
        # Create mock Redis client for development
        redis_client = MockRedisClient()


async def close_redis():
    """Close Redis connection."""
    global redis_client
    if redis_client and hasattr(redis_client, 'close'):
        await redis_client.close()
        logger.info("Redis connection closed")


async def get_redis_client():
    """Get Redis client instance."""
    return redis_client


class MockRedisClient:
    """Mock Redis client for development when Redis is unavailable."""
    
    def __init__(self):
        self.data = {}
    
    async def ping(self):
        return True
    
    async def set(self, key, value, ex=None):
        self.data[key] = value
        return True
    
    async def get(self, key):
        return self.data.get(key)
    
    async def delete(self, key):
        self.data.pop(key, None)
        return True
    
    async def close(self):
        pass
'''
    
    with open("app/core/redis.py", "w") as f:
        f.write(redis_content)
    print("✅ Created: app/core/redis.py")

def create_service_files():
    """Create service modules."""
    
    services = {
        "ai_orchestrator": '''import asyncio
import structlog
from typing import Dict, Any

logger = structlog.get_logger()


class AIOrchestrator:
    """Main AI orchestrator for music generation."""
    
    def __init__(self):
        self.initialized = False
        self.models = {}
    
    async def initialize(self):
        """Initialize AI models and services."""
        logger.info("Initializing AI Orchestrator...")
        await asyncio.sleep(0.1)
        self.initialized = True
        logger.info("AI Orchestrator initialized")
    
    async def cleanup(self):
        """Clean up AI resources."""
        logger.info("Cleaning up AI Orchestrator...")
        self.initialized = False
    
    async def health_check(self):
        """Check AI service health."""
        return "healthy" if self.initialized else "unhealthy"
    
    async def generate_full_song(self, project_id: str, style_config: Dict, 
                                advanced_options: Dict, websocket_manager=None, user_id=None):
        """Generate a complete song."""
        logger.info(f"Starting full song generation for project {project_id}")
        
        stages = ["Analyzing prompt", "Generating melody", "Adding harmony", "Mixing", "Finalizing"]
        
        for i, stage in enumerate(stages):
            logger.info(f"Stage {i+1}/{len(stages)}: {stage}")
            await asyncio.sleep(1)
            
            if websocket_manager and user_id:
                await websocket_manager.send_personal_message({
                    "type": "generation_progress",
                    "project_id": project_id,
                    "stage": stage,
                    "progress": (i + 1) / len(stages) * 100
                }, user_id)
        
        logger.info(f"Song generation completed for project {project_id}")
        return {"status": "completed", "audio_url": f"/audio/{project_id}.mp3"}
    
    async def create_project(self, name: str, user_id: str, style_config: Dict):
        """Create a new music project."""
        import uuid
        project_id = str(uuid.uuid4())
        logger.info(f"Created project {project_id} for user {user_id}")
        return project_id
    
    async def get_total_songs(self):
        """Get total number of songs generated."""
        return 10000
    
    async def get_performance_metrics(self):
        """Get AI performance metrics."""
        return {
            "avg_generation_time": "2.5s",
            "success_rate": "99.2%",
            "active_models": 5,
            "queue_size": 3
        }


# Global instance
ai_orchestrator = AIOrchestrator()
''',
        
        "audio_engine": '''import asyncio
import structlog

logger = structlog.get_logger()


class AudioEngine:
    """Audio processing and rendering engine."""
    
    def __init__(self):
        self.initialized = False
        self.processing_queue = []
    
    async def initialize(self):
        """Initialize audio engine."""
        logger.info("Initializing Audio Engine...")
        await asyncio.sleep(0.1)
        self.initialized = True
        logger.info("Audio Engine initialized")
    
    async def cleanup(self):
        """Clean up audio resources."""
        logger.info("Cleaning up Audio Engine...")
        self.initialized = False
    
    async def health_check(self):
        """Check audio engine health."""
        return "healthy" if self.initialized else "unhealthy"
    
    async def get_queue_size(self):
        """Get processing queue size."""
        return len(self.processing_queue)


# Global instance
audio_engine = AudioEngine()
''',
        
        "blockchain_service": '''import asyncio
import structlog

logger = structlog.get_logger()


class BlockchainService:
    """Blockchain integration for NFT minting and transactions."""
    
    def __init__(self):
        self.initialized = False
        self.transaction_count = 0
    
    async def initialize(self):
        """Initialize blockchain connections."""
        logger.info("Initializing Blockchain Service...")
        await asyncio.sleep(0.1)
        self.initialized = True
        logger.info("Blockchain Service initialized")
    
    async def cleanup(self):
        """Clean up blockchain connections."""
        logger.info("Cleaning up Blockchain Service...")
        self.initialized = False
    
    async def health_check(self):
        """Check blockchain service health."""
        return "healthy" if self.initialized else "unhealthy"
    
    async def get_transaction_count(self):
        """Get total transaction count."""
        return self.transaction_count


# Global instance
blockchain_service = BlockchainService()
''',
        
        "websocket_manager": '''from fastapi import WebSocket
from typing import Dict, List
import json
import structlog

logger = structlog.get_logger()


class WebSocketManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, List[str]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept and store new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = []
        logger.info(f"WebSocket client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        logger.info(f"WebSocket client {client_id} disconnected")
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def subscribe(self, client_id: str, events: List[str]):
        """Subscribe client to specific events."""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].extend(events)
            logger.info(f"Client {client_id} subscribed to {events}")
    
    async def unsubscribe(self, client_id: str, events: List[str]):
        """Unsubscribe client from specific events."""
        if client_id in self.subscriptions:
            for event in events:
                if event in self.subscriptions[client_id]:
                    self.subscriptions[client_id].remove(event)
            logger.info(f"Client {client_id} unsubscribed from {events}")


# Global instance
websocket_manager = WebSocketManager()
'''
    }
    
    for service_name, content in services.items():
        with open(f"app/services/{service_name}.py", "w") as f:
            f.write(content)
        print(f"✅ Created: app/services/{service_name}.py")

def create_api_files():
    """Create API route files."""
    
    # Main routes file
    routes_content = '''from fastapi import APIRouter
from app.api.endpoints import (
    auth, music, projects, collaboration, 
    marketplace, analytics, enterprise, 
    nft, payments, collab_lab, dev
)

api_router = APIRouter()

# Include all route modules
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(music.router, prefix="/music", tags=["Music Generation"])
api_router.include_router(projects.router, prefix="/projects", tags=["Projects"])
# api_router.include_router(collaboration.router, prefix="/collaboration", tags=["Collaboration"])
# api_router.include_router(marketplace.router, prefix="/marketplace", tags=["Marketplace"])
# api_router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
# api_router.include_router(enterprise.router, prefix="/enterprise", tags=["Enterprise"])
# api_router.include_router(nft.router, prefix="/nft", tags=["NFT & Blockchain"])
# api_router.include_router(payments.router, prefix="/payments", tags=["Payments"])
# api_router.include_router(collab_lab.router, prefix="/collab-lab", tags=["Collab Lab"])
# api_router.include_router(dev.router, prefix="/dev", tags=["Development"])
'''
    
    with open("app/api/routes.py", "w") as f:
        f.write(routes_content)
    print("✅ Created: app/api/routes.py")
    
    # Create endpoint files
    endpoints = {
        "auth": '''from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid

router = APIRouter()

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str

@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """User login endpoint."""
    if request.email and request.password:
        return TokenResponse(
            access_token="mock-jwt-token",
            token_type="bearer",
            user_id=str(uuid.uuid4())
        )
    raise HTTPException(status_code=401, detail="Invalid credentials")

@router.get("/me")
async def get_current_user():
    """Get current user info."""
    return {"user_id": "mock-user-id", "email": "user@example.com", "premium": True}
''',
        
        "music": '''from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import uuid

router = APIRouter()

class MusicGenerationRequest(BaseModel):
    prompt: str
    style: Optional[str] = "pop"
    duration: Optional[int] = 30

@router.post("/generate")
async def generate_music(request: MusicGenerationRequest, background_tasks: BackgroundTasks):
    """Generate music from text prompt."""
    generation_id = str(uuid.uuid4())
    return {
        "generation_id": generation_id,
        "status": "processing",
        "message": "Music generation started"
    }

@router.get("/styles")
async def get_music_styles():
    """Get available music styles."""
    return {"styles": ["pop", "rock", "jazz", "classical", "electronic"]}
''',
        
        "projects": '''from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import uuid

router = APIRouter()

class Project(BaseModel):
    id: str
    name: str
    status: str
    created_at: str

@router.get("/", response_model=List[Project])
async def get_projects():
    """Get all projects for user."""
    return [
        Project(
            id=str(uuid.uuid4()),
            name="My First Song",
            status="completed",
            created_at="2024-01-01T00:00:00Z"
        )
    ]

@router.get("/{project_id}/status")
async def get_project_status(project_id: str):
    """Get project status."""
    return {"project_id": project_id, "status": "completed", "progress": 100}
''',
        
        "collaboration": '''from fastapi import APIRouter
router = APIRouter()

@router.get("/rooms")
async def get_collaboration_rooms():
    return {"rooms": [{"id": "room1", "name": "Beat Lab", "participants": 3}]}
''',
        
        "marketplace": '''from fastapi import APIRouter
router = APIRouter()

@router.get("/tracks")
async def get_marketplace_tracks():
    return {"tracks": [{"id": "track1", "title": "Summer Vibes", "price": 9.99}]}
''',
        
        "analytics": '''from fastapi import APIRouter
router = APIRouter()

@router.get("/dashboard")
async def get_analytics_dashboard():
    return {"plays": 10000, "revenue": 5000, "top_tracks": ["Track A"]}
''',
        
        "enterprise": '''from fastapi import APIRouter
router = APIRouter()

@router.get("/teams")
async def get_enterprise_teams():
    return {"teams": [{"id": "team1", "name": "Production Team"}]}
''',
        
        "nft": '''from fastapi import APIRouter
router = APIRouter()

@router.post("/mint")
async def mint_music_nft():
    return {"nft_id": "nft_123", "status": "minted"}
''',
        
        "payments": '''from fastapi import APIRouter
router = APIRouter()

@router.get("/billing")
async def get_billing_info():
    return {"plan": "premium", "amount": 29.99}
''',
        
        "collab_lab": '''from fastapi import APIRouter
router = APIRouter()

@router.get("/sessions")
async def get_collab_sessions():
    return {"sessions": [{"id": "session1", "title": "Beat Making Session"}]}
''',
        
        "dev": '''from fastapi import APIRouter
router = APIRouter()

@router.get("/info")
async def dev_info():
    return {"environment": "development", "debug": True, "version": "2.0.0"}
'''
    }
    
    for endpoint_name, content in endpoints.items():
        with open(f"app/api/endpoints/{endpoint_name}.py", "w") as f:
            f.write(content)
        print(f"✅ Created: app/api/endpoints/{endpoint_name}.py")

def create_env_file():
    """Create a .env file with default settings."""
    env_content = '''# Development Environment Configuration
ENV=development
DEBUG=true
PORT=8000
LOG_LEVEL=INFO

# API Configuration
API_KEY=test-api-key-123
SECRET_KEY=dev-secret-key-change-in-production

# Database
DATABASE_URL=sqlite+aiosqlite:///./test.db

# Redis (optional)
REDIS_URL=redis://localhost:6379

# CORS Origins (comma-separated)
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,https://localhost:3000
'''
    
    with open(".env", "w") as f:
        f.write(env_content)
    print("✅ Created: .env")

def main():
    """Run the setup process."""
    print("🎵 Setting up AI Music Generation Backend...")
    print("=" * 50)
    
    create_directory_structure()
    create_init_files()
    create_config_file()
    create_core_files()
    create_service_files()
    create_api_files()
    create_env_file()
    
    print("\n" + "=" * 50)
    print("✅ Setup complete! 🎉")
    print("\nNext steps:")
    print("1. Run: python main.py")
    print("2. Open: http://localhost:8000/docs")
    print("3. Test: curl http://localhost:8000/health")

if __name__ == "__main__":
    main()
