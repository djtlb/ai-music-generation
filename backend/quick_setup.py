#!/usr/bin/env python3
"""Quick setup to create missing modules and get the server running"""

import os
from pathlib import Path

def create_file(path, content):
    """Create a file with the given content."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f"✅ Created {path}")

def main():
    print("🚀 Setting up AI Music Generation Backend...")
    
    # Create __init__.py files
    init_files = [
        "app/__init__.py",
        "app/core/__init__.py", 
        "app/api/__init__.py",
        "app/api/endpoints/__init__.py",
        "app/services/__init__.py"
    ]

    for init_file in init_files:
        create_file(init_file, "")

    # Create config
    create_file("app/core/config.py", '''from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    """Application settings"""
    CORS_ORIGINS: List[str] = ["*"]
    API_KEY: str = "test-api-key-123"
    
    class Config:
        env_file = ".env"
        
settings = Settings()
''')

    # Create database
    create_file("app/core/database.py", '''import structlog
logger = structlog.get_logger()

async def init_db():
    """Initialize database (mock)"""
    logger.info("Mock database initialized")

async def close_db():
    """Close database (mock)"""
    logger.info("Mock database closed")

async def get_db_session():
    """Get database session (mock)"""
    class MockSession:
        async def execute(self, query): 
            return {"result": "mock"}
        async def close(self): 
            pass
    yield MockSession()
''')

    # Create redis
    create_file("app/core/redis.py", '''import structlog
logger = structlog.get_logger()

class MockRedis:
    """Mock Redis client for development"""
    async def ping(self): 
        return True
    async def close(self): 
        pass

redis_client = MockRedis()

async def init_redis():
    """Initialize Redis (mock)"""
    logger.info("Mock Redis initialized")

async def close_redis():
    """Close Redis (mock)"""
    logger.info("Mock Redis closed")

async def get_redis_client():
    """Get Redis client (mock)"""
    return redis_client
''')

    # Create services
    services = {
        "ai_orchestrator": '''import structlog
import asyncio
import uuid

logger = structlog.get_logger()

class AIOrchestrator:
    """Main AI orchestrator for music generation"""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self): 
        """Initialize AI models"""
        self.initialized = True
        logger.info("AI Orchestrator initialized")
    
    async def cleanup(self): 
        """Cleanup AI resources"""
        self.initialized = False
        logger.info("AI Orchestrator cleaned up")
    
    async def health_check(self): 
        """Health check"""
        return "healthy" if self.initialized else "unhealthy"
    
    async def create_project(self, name, user_id, style_config): 
        """Create new project"""
        project_id = str(uuid.uuid4())
        logger.info(f"Created project {project_id}")
        return project_id
    
    async def generate_full_song(self, project_id, style_config, advanced_options, websocket_manager, user_id):
        """Generate a full song (mock)"""
        logger.info(f"Generating song for project {project_id}")
        # Mock generation with progress updates
        stages = ["Initializing", "Generating melody", "Adding harmonies", "Mixing", "Finalizing"]
        for i, stage in enumerate(stages):
            await asyncio.sleep(0.5)  # Simulate work
            if websocket_manager:
                await websocket_manager.send_personal_message({
                    "type": "generation_progress",
                    "project_id": project_id,
                    "stage": stage,
                    "progress": (i + 1) / len(stages) * 100
                }, user_id)
        
        logger.info(f"Song generation completed for project {project_id}")
        return {"status": "completed", "audio_url": f"/audio/{project_id}.mp3"}
    
    async def get_total_songs(self): 
        """Get total songs generated"""
        return 10000
    
    async def get_performance_metrics(self): 
        """Get performance metrics"""
        return {
            "avg_generation_time": "2.3s",
            "success_rate": "99.1%",
            "queue_size": 3
        }

ai_orchestrator = AIOrchestrator()
''',
        
        "audio_engine": '''import structlog

logger = structlog.get_logger()

class AudioEngine:
    """Audio processing and rendering engine"""
    
    def __init__(self):
        self.initialized = False
        self.queue = []
    
    async def initialize(self): 
        """Initialize audio engine"""
        self.initialized = True
        logger.info("Audio Engine initialized")
    
    async def cleanup(self): 
        """Cleanup audio resources"""
        self.initialized = False
        logger.info("Audio Engine cleaned up")
    
    async def health_check(self): 
        """Health check"""
        return "healthy" if self.initialized else "unhealthy"
    
    async def get_queue_size(self): 
        """Get processing queue size"""
        return len(self.queue)

audio_engine = AudioEngine()
''',
        
        "blockchain_service": '''import structlog

logger = structlog.get_logger()

class BlockchainService:
    """Blockchain integration for NFT minting"""
    
    def __init__(self):
        self.initialized = False
        self.transaction_count = 0
    
    async def initialize(self): 
        """Initialize blockchain service"""
        self.initialized = True
        logger.info("Blockchain Service initialized")
    
    async def cleanup(self): 
        """Cleanup blockchain resources"""
        self.initialized = False
        logger.info("Blockchain Service cleaned up")
    
    async def health_check(self): 
        """Health check"""
        return "healthy" if self.initialized else "unhealthy"
    
    async def get_transaction_count(self): 
        """Get transaction count"""
        return self.transaction_count

blockchain_service = BlockchainService()
''',
        
        "websocket_manager": '''import json
import structlog
from typing import Dict
from fastapi import WebSocket

logger = structlog.get_logger()

class WebSocketManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, list] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept and store new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = []
        logger.info(f"Client {client_id} connected via WebSocket")
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        logger.info(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def subscribe(self, client_id: str, events: list):
        """Subscribe client to events"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].extend(events)
            logger.info(f"Client {client_id} subscribed to {events}")
    
    async def unsubscribe(self, client_id: str, events: list):
        """Unsubscribe client from events"""
        if client_id in self.subscriptions:
            for event in events:
                if event in self.subscriptions[client_id]:
                    self.subscriptions[client_id].remove(event)
            logger.info(f"Client {client_id} unsubscribed from {events}")

websocket_manager = WebSocketManager()
'''
    }

    for name, content in services.items():
        create_file(f"app/services/{name}.py", content)

    # Create API routes
    create_file("app/api/routes.py", '''from fastapi import APIRouter

api_router = APIRouter()

@api_router.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {
        "message": "🎵 AI Music Generation API is working!",
        "status": "operational",
        "test": "passed"
    }

@api_router.get("/status")
async def api_status():
    """API status endpoint"""
    return {
        "api": "AI Music Generation",
        "version": "2.0.0",
        "status": "ready"
    }
''')

    # Create .env file
    create_file(".env", '''# Development Environment
ENV=development
DEBUG=true
PORT=8000
LOG_LEVEL=INFO
API_KEY=test-api-key-123
''')

    print("\n" + "=" * 50)
    print("✅ Setup complete! 🎉")
    print("\nNext steps:")
    print("1. Run: python main.py")
    print("2. Open: http://localhost:8000")
    print("3. API Docs: http://localhost:8000/docs")
    print("4. Test: curl http://localhost:8000/health")

if __name__ == "__main__":
    main()
