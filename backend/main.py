"""
Million-Dollar AI Music Generation System
FastAPI Backend Server
"""

import asyncio
import json
import io
import os
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

# Import our modules
from app.core.config import settings
from app.core.database import init_db, close_db, get_db_session
from app.core.redis import init_redis, close_redis, get_redis_client
from app.api.routes import api_router
from app.services.ai_orchestrator import ai_orchestrator
from app.services.audio_engine import audio_engine
from app.services.blockchain_service import blockchain_service
from app.services.websocket_manager import websocket_manager

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Environment variables
ENV = os.getenv("ENV", "production")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
fast_dev = ENV == "development"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    logger.info("🎵 Starting AI Music Generation System", version="2.0.0", env=ENV)
    
    async def safe(label: str, coro):
        try:
            await coro
            logger.info(f"✅ {label}")
        except Exception as e:
            if fast_dev:
                logger.warning(f"⚠️ FAST_DEV skip {label}: {e}")
            else:
                logger.warning(f"⚠️ {label} failed: {e}")

    # Initialize components
    await safe("Database initialized", init_db())
    
    # Initialize Redis
    try:
        await init_redis()
        redis_client = await get_redis_client()
        await redis_client.ping()
        logger.info("✅ Redis connection established")
    except Exception as e:
        logger.warning(f"⚠️ Redis unavailable: {e}")
    
    await safe("AI models loaded", ai_orchestrator.initialize())
    await safe("Audio engine ready", audio_engine.initialize())
    await safe("Blockchain service connected", blockchain_service.initialize())
    
    logger.info(f"🎵 System ready (ENV={ENV})")

    yield

    logger.info("🔄 Shutting down system...")
    await ai_orchestrator.cleanup()
    await audio_engine.cleanup()
    await blockchain_service.cleanup()
    await close_redis()
    await close_db()
    logger.info("✅ Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="AI Music Generation API",
    description="Million-Dollar AI Music Generation Platform",
    version="2.0.0",
    docs_url="/docs" if ENV == "development" else None,
    redoc_url="/redoc" if ENV == "development" else None,
    openapi_url="/openapi.json" if ENV == "development" else None,
    lifespan=lifespan
)

# Add middleware - optimized for web hosting
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Health endpoints
@app.get("/healthz")
async def liveness_probe():
    """Simple liveness probe for container orchestrators."""
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness_probe():
    """Readiness probe for orchestrators / load balancers."""
    return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}

@app.get("/readyz")
async def readiness_probe_alt():
    """Alternative path for readiness probe."""
    return await readiness_probe()

# Create directories for static files
os.makedirs("static", exist_ok=True)
os.makedirs("generated_audio", exist_ok=True)

# Mount static files
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
if Path("generated_audio").exists():
    app.mount("/audio", StaticFiles(directory="generated_audio"), name="audio")

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint with API information."""
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
            async for session in get_db_session():
                await session.execute("SELECT 1")
                break
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"
        
        # Check Redis
        redis_status = "healthy"
        try:
            redis_client = await get_redis_client()
            await redis_client.ping()
        except Exception as e:
            redis_status = f"unhealthy: {str(e)}"
        
        # Check services
        ai_status = await ai_orchestrator.health_check()
        audio_status = await audio_engine.health_check()
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
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return {"error": "prometheus_client not installed", "metrics": "unavailable"}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket_manager.send_personal_message(
                    {"type": "pong", "timestamp": datetime.utcnow().isoformat()},
                    client_id
                )
            elif message.get("type") == "subscribe":
                await websocket_manager.subscribe(client_id, message.get("events", []))
            elif message.get("type") == "unsubscribe":
                await websocket_manager.unsubscribe(client_id, message.get("events", []))
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        websocket_manager.disconnect(client_id)

# API key verification (simple mock)
async def verify_api_key(api_key: str = "test-api-key-123"):
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

@app.post("/api/v1/generate/full-song")
async def generate_full_song(
    request: dict,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Generate a complete song using the full AI pipeline"""
    try:
        project_name = request.get("project_name", "Untitled Song")
        style_config = request.get("style_config", {})
        user_id = request.get("user_id")
        advanced_options = request.get("advanced_options", {})
        
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
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist",
            "docs": "/docs" if ENV == "development" else "API Documentation not available in production"
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "support": "contact@aimusicgen.com"
        }
    )

# IMPORTANT: Mount the static frontend last, as it's a catch-all for any path not defined above.
if Path("dist").exists():  # built frontend assets
    app.mount("/", StaticFiles(directory="dist", html=True), name="frontend")

# Web hosting optimized startup
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,  # Always False for production
        workers=1,  # Single worker for most web hosts
        log_level=LOG_LEVEL.lower(),
        access_log=False,  # Disable access logs for performance
    )

# WSGI/ASGI app for web hosting platforms
application = app  # For gunicorn/uvicorn deployment
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "support": "contact@aimusicgen.com"
        }
    )

# IMPORTANT: Mount the static frontend last, as it's a catch-all for any path not defined above.
if Path("dist").exists():  # built frontend assets
    app.mount("/", StaticFiles(directory="dist", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=ENV != "production",
        workers=int(os.getenv("WORKERS", "1")),
        log_level=LOG_LEVEL.lower(),
    )
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
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist",
            "docs": "/docs"
        }
    )

# IMPORTANT: Mount the static frontend last, as it's a catch-all for any path not defined above.
if Path("dist").exists():  # built frontend assets
    app.mount("/", StaticFiles(directory="dist", html=True), name="frontend")

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "support": "contact@aimusicgen.com"
        }
    )

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
        port=int(os.getenv("PORT", 8000)),
        reload=ENV != "production",
        workers=int(os.getenv("WORKERS", "1")),
        log_level=LOG_LEVEL.lower(),
    )
