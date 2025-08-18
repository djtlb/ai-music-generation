"""
Million-Dollar AI Music Generation System
FastAPI Backend Server

This is the main FastAPI application that powers a comprehensive AI music generation platform.
Features include:
- Advanced AI music composition
- Real-time collaboration
- Professional audio rendering
- NFT minting and blockchain integration
- Subscription management
- Enterprise analytics
"""

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
import asyncio
import io
from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timedelta
import os
from pathlib import Path

# Import custom modules
from config.settings import get_settings
from api.routes import (
    auth_router,
    music_generation_router,
    projects_router,
    collaboration_router,
    marketplace_router,
    analytics_router,
    enterprise_router,
    nft_router,
    payments_router
)
from api.routes.collab_router import router as collab_lab_router
from core.database import init_db, get_db_session
from core.redis_client import get_redis_client
from core.websocket_manager import WebSocketManager
from core.security import verify_api_key
from middleware.rate_limiting import RateLimitingMiddleware
from middleware.monitoring import PrometheusMiddleware
from services.ai_orchestrator import AIOrchestrator
from services.audio_engine import AudioEngine
from services.blockchain_service import BlockchainService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
websocket_manager = WebSocketManager()
ai_orchestrator = AIOrchestrator()
audio_engine = AudioEngine()
blockchain_service = BlockchainService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Million-Dollar AI Music Generation System...")
    
    # Initialize database
    await init_db()
    
    # Initialize Redis connection
    redis_client = await get_redis_client()
    await redis_client.ping()
    logger.info("✅ Redis connection established")
    
    # Initialize AI models
    await ai_orchestrator.initialize()
    logger.info("✅ AI models loaded")
    
    # Initialize audio engine
    await audio_engine.initialize()
    logger.info("✅ Audio engine ready")
    
    # Initialize blockchain service
    await blockchain_service.initialize()
    logger.info("✅ Blockchain service connected")
    
    logger.info("🎵 System ready for million-dollar music generation!")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down system...")
    await ai_orchestrator.cleanup()
    await audio_engine.cleanup()
    await blockchain_service.cleanup()
    logger.info("✅ Shutdown complete")

# Create FastAPI app with lifespan
app = FastAPI(
    title="AI Music Generation API",
    description="Million-Dollar AI Music Generation Platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RateLimitingMiddleware)
app.add_middleware(PrometheusMiddleware)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/audio", StaticFiles(directory="generated_audio"), name="audio")

# Include routers
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(music_generation_router, prefix="/api/v1/music", tags=["Music Generation"])
app.include_router(projects_router, prefix="/api/v1/projects", tags=["Projects"])
app.include_router(collaboration_router, prefix="/api/v1/collaboration", tags=["Collaboration"])
app.include_router(marketplace_router, prefix="/api/v1/marketplace", tags=["Marketplace"])
app.include_router(analytics_router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(enterprise_router, prefix="/api/v1/enterprise", tags=["Enterprise"])
app.include_router(nft_router, prefix="/api/v1/nft", tags=["NFT & Blockchain"])
app.include_router(payments_router, prefix="/api/v1/payments", tags=["Payments"])
app.include_router(collab_lab_router, prefix="/api/v1/collab-lab", tags=["Collab Lab"])

@app.get("/")
async def root():
    """Root endpoint with system status"""
    return {
        "message": "🎵 Million-Dollar AI Music Generation API",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "AI Music Composition",
            "Real-time Collaboration", 
            "Professional Audio Rendering",
            "NFT Minting",
            "Blockchain Integration",
            "Enterprise Analytics",
            "Payment Processing",
            "Global Distribution"
        ],
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
            "websocket": "/ws"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        # Check database
        db_status = "healthy"
        try:
            db_session = await get_db_session()
            await db_session.execute("SELECT 1")
            await db_session.close()
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"
        
        # Check Redis
        redis_status = "healthy"
        try:
            redis_client = await get_redis_client()
            await redis_client.ping()
        except Exception as e:
            redis_status = f"unhealthy: {str(e)}"
        
        # Check AI services
        ai_status = await ai_orchestrator.health_check()
        
        # Check audio engine
        audio_status = await audio_engine.health_check()
        
        # Check blockchain
        blockchain_status = await blockchain_service.health_check()
        
        overall_status = "healthy" if all([
            "healthy" in db_status,
            "healthy" in redis_status,
            ai_status == "healthy",
            audio_status == "healthy",
            blockchain_status == "healthy"
        ]) else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": db_status,
                "redis": redis_status,
                "ai_orchestrator": ai_status,
                "audio_engine": audio_status,
                "blockchain": blockchain_status
            },
            "version": "2.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return StreamingResponse(
        io.BytesIO(generate_latest()),
        media_type=CONTENT_TYPE_LATEST
    )

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket, client_id)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await websocket_manager.send_personal_message(
                    {"type": "pong", "timestamp": datetime.utcnow().isoformat()},
                    client_id
                )
            elif message.get("type") == "subscribe":
                # Subscribe to specific events
                await websocket_manager.subscribe(client_id, message.get("events", []))
            elif message.get("type") == "unsubscribe":
                # Unsubscribe from events
                await websocket_manager.unsubscribe(client_id, message.get("events", []))
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        websocket_manager.disconnect(client_id)

@app.post("/api/v1/generate/full-song")
async def generate_full_song(
    request: dict,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Generate a complete song using the full AI pipeline
    This is the main endpoint that ties everything together
    """
    try:
        # Extract parameters
        project_name = request.get("project_name", "Untitled Song")
        style_config = request.get("style_config", {})
        user_id = request.get("user_id")
        advanced_options = request.get("advanced_options", {})
        
        # Validate required parameters
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Create project
        project_id = await ai_orchestrator.create_project(
            name=project_name,
            user_id=user_id,
            style_config=style_config
        )
        
        # Start generation in background
        background_tasks.add_task(
            ai_orchestrator.generate_full_song,
            project_id=project_id,
            style_config=style_config,
            advanced_options=advanced_options,
            websocket_manager=websocket_manager,
            user_id=user_id
        )
        
        return {
            "message": "Song generation started",
            "project_id": project_id,
            "estimated_completion": datetime.utcnow() + timedelta(minutes=5),
            "websocket_url": f"/ws/{user_id}",
            "status_endpoint": f"/api/v1/projects/{project_id}/status"
        }
        
    except Exception as e:
        logger.error(f"Full song generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/system/stats")
async def system_stats(api_key: str = Depends(verify_api_key)):
    """Get comprehensive system statistics"""
    try:
        stats = {
            "total_songs_generated": await ai_orchestrator.get_total_songs(),
            "active_users": await get_active_users_count(),
            "revenue_today": await get_daily_revenue(),
            "server_uptime": get_server_uptime(),
            "ai_model_performance": await ai_orchestrator.get_performance_metrics(),
            "audio_processing_queue": await audio_engine.get_queue_size(),
            "blockchain_transactions": await blockchain_service.get_transaction_count(),
            "storage_usage": await get_storage_usage(),
            "api_calls_today": await get_api_calls_count(),
            "premium_subscriptions": await get_premium_count()
        }
        
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist",
        "docs": "/docs"
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "support": "contact@aimusicgen.com"
    }

# Helper functions (to be implemented)
async def get_active_users_count():
    """Get count of active users in the last 24 hours"""
    # Implementation would query database
    return 1500

async def get_daily_revenue():
    """Get today's revenue"""
    # Implementation would query payment records
    return 25000.00

def get_server_uptime():
    """Get server uptime"""
    # Implementation would track startup time
    return "2 days, 5 hours, 23 minutes"

async def get_storage_usage():
    """Get storage usage statistics"""
    # Implementation would check file system usage
    return {
        "audio_files": "2.5 TB",
        "project_data": "500 GB",
        "user_uploads": "1.2 TB",
        "total_used": "4.2 TB",
        "total_capacity": "10 TB"
    }

async def get_api_calls_count():
    """Get API calls count for today"""
    # Implementation would query metrics
    return 45000

async def get_premium_count():
    """Get premium subscription count"""
    # Implementation would query subscription database
    return 3500

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
        log_level="info"
    )
