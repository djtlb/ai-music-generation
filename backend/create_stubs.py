#!/usr/bin/env python3
"""Create minimal stub files to get your REAL code running"""

import os
from pathlib import Path

def create_file(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f"✅ Created stub: {path}")

# Create __init__.py files
init_files = [
    "app/__init__.py",
    "app/core/__init__.py",
    "app/api/__init__.py", 
    "app/services/__init__.py"
]

for init_file in init_files:
    create_file(init_file, "")

# Create config stub
create_file("app/core/config.py", '''
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    CORS_ORIGINS: List[str] = ["*"]
    API_KEY: str = "test-api-key-123"

settings = Settings()
''')

# Create database stub
create_file("app/core/database.py", '''
async def init_db(): pass
async def close_db(): pass
async def get_db_session():
    class MockSession:
        async def execute(self, query): pass
    yield MockSession()
''')

# Create redis stub
create_file("app/core/redis.py", '''
class MockRedis:
    async def ping(self): return True
    async def close(self): pass

redis_client = MockRedis()

async def init_redis(): pass
async def close_redis(): pass
async def get_redis_client(): return redis_client
''')

# Create service stubs
create_file("app/services/ai_orchestrator.py", '''
class AIOrchestrator:
    async def initialize(self): pass
    async def cleanup(self): pass
    async def health_check(self): return "healthy"
    async def create_project(self, name, user_id, style_config): return "proj_123"
    async def generate_full_song(self, *args, **kwargs): pass
    async def get_total_songs(self): return 10000
    async def get_performance_metrics(self): return {}

ai_orchestrator = AIOrchestrator()
''')

create_file("app/services/audio_engine.py", '''
class AudioEngine:
    async def initialize(self): pass
    async def cleanup(self): pass
    async def health_check(self): return "healthy"
    async def get_queue_size(self): return 0

audio_engine = AudioEngine()
''')

create_file("app/services/blockchain_service.py", '''
class BlockchainService:
    async def initialize(self): pass
    async def cleanup(self): pass
    async def health_check(self): return "healthy"
    async def get_transaction_count(self): return 0

blockchain_service = BlockchainService()
''')

create_file("app/services/websocket_manager.py", '''
class WebSocketManager:
    async def connect(self, websocket, client_id):
        await websocket.accept()
    def disconnect(self, client_id): pass
    async def send_personal_message(self, message, client_id): pass
    async def subscribe(self, client_id, events): pass
    async def unsubscribe(self, client_id, events): pass

websocket_manager = WebSocketManager()
''')

# Create routes stub
create_file("app/api/routes.py", '''
from fastapi import APIRouter

api_router = APIRouter()

@api_router.get("/test")
async def test():
    return {"message": "API working"}
''')

print("🚀 Stubs created! Now replace these with your REAL implementations and run: python main.py")
