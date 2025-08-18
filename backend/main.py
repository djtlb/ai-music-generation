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
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
import io
from typing import Sequence
import json
from datetime import datetime, timedelta
import os
from pathlib import Path

# ----------------------------------------------------------------------------------
# Environment & Logging Setup
# ----------------------------------------------------------------------------------
ENV = os.getenv("ENV", "development").lower()
FAST_DEV = os.getenv("FAST_DEV") == "1"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def _configure_logging():
    """Configure structured logging in production, plain logging in development."""
    # Avoid reconfiguration if already set
    if logging.getLogger().handlers:
        return
    if ENV == "production":
        try:
            import structlog  # type: ignore

            logging.basicConfig(
                level=LOG_LEVEL,
                format="%(message)s",
                handlers=[logging.StreamHandler()]
            )
            structlog.configure(
                processors=[
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.add_log_level,
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer()
                ],
                wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, LOG_LEVEL, logging.INFO)),
                cache_logger_on_first_use=True,
            )
        except Exception:  # pragma: no cover - best effort
            logging.basicConfig(level=LOG_LEVEL)
    else:
        logging.basicConfig(
            level=LOG_LEVEL,
            format="[%(levelname)s] %(asctime)s %(name)s: %(message)s"
        )

_configure_logging()

logger = logging.getLogger(__name__)

# Import custom modules (development tolerant, production fail-fast)
missing_optional: list[str] = []
def _opt_import(path, name=None):
    try:
        mod = __import__(path, fromlist=['*'])
        return mod if name is None else getattr(mod, name)
    except Exception as e:  # pragma: no cover - dev convenience
        missing_optional.append(f"{path}{'.'+name if name else ''}: {e}")
        return None

get_settings = _opt_import('config.settings', 'get_settings') or (lambda: type('S', (), {'allowed_origins':'*'})())

auth_router = _opt_import('api.routes', 'auth_router')
music_generation_router = _opt_import('api.routes', 'music_generation_router')
projects_router = _opt_import('api.routes', 'projects_router')
collaboration_router = _opt_import('api.routes', 'collaboration_router')
marketplace_router = _opt_import('api.routes', 'marketplace_router')
analytics_router = _opt_import('api.routes', 'analytics_router')
enterprise_router = _opt_import('api.routes', 'enterprise_router')
nft_router = _opt_import('api.routes', 'nft_router')
payments_router = _opt_import('api.routes', 'payments_router')
collab_lab_router = _opt_import('api.routes.collab_router', 'router')
dev_router = _opt_import('api.routes.dev', 'router')

init_db = _opt_import('core.database', 'init_db')
get_db_session = _opt_import('core.database', 'get_db_session')
get_redis_client = _opt_import('core.redis_client', 'get_redis_client')
WebSocketManager = _opt_import('core.websocket_manager', 'WebSocketManager')
verify_api_key = _opt_import('core.security', 'verify_api_key') or (lambda : lambda : True)
RateLimitingMiddleware = _opt_import('middleware.rate_limiting', 'RateLimitingMiddleware')
PrometheusMiddleware = _opt_import('middleware.monitoring', 'PrometheusMiddleware')
AIOrchestrator = _opt_import('services.ai_orchestrator', 'AIOrchestrator')
AudioEngine = _opt_import('services.audio_engine', 'AudioEngine')
BlockchainService = _opt_import('services.blockchain_service', 'BlockchainService')

class _AsyncStub:
    async def initialize(self): pass
    async def cleanup(self): pass
    async def health_check(self): return 'healthy'

class _AIStub(_AsyncStub):
    async def create_project(self, **k): return 'proj'
    async def generate_full_song(self, **k): return None
    async def get_total_songs(self): return 0
    async def get_performance_metrics(self): return {}

class _AudioStub(_AsyncStub):
    async def get_queue_size(self): return 0

class _BCStub(_AsyncStub):
    async def get_transaction_count(self): return 0

if AIOrchestrator is None:
    AIOrchestrator = _AIStub  # type: ignore
if AudioEngine is None:
    AudioEngine = _AudioStub  # type: ignore
if BlockchainService is None:
    BlockchainService = _BCStub  # type: ignore

if ENV == "production" and missing_optional:
    # Fail fast with explicit module list
    raise RuntimeError(
        "Missing required modules in production mode: " + ", ".join(missing_optional)
    )

if FAST_DEV and ENV == "production":  # Guard against misconfiguration
    logger.warning("FAST_DEV=1 ignored in production environment")

# Global instances
if WebSocketManager is None:
    class _WSStub:
        async def connect(self, *a, **k): pass
        async def send_personal_message(self, *a, **k): pass
        async def subscribe(self, *a, **k): pass
        async def unsubscribe(self, *a, **k): pass
        def disconnect(self, *a, **k): pass
    websocket_manager = _WSStub()
else:
    websocket_manager = WebSocketManager()
ai_orchestrator = AIOrchestrator()
audio_engine = AudioEngine()
blockchain_service = BlockchainService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager (FAST_DEV tolerant)."""
    logger.info("🚀 Starting Million-Dollar AI Music Generation System...", extra={"env": ENV, "fast_dev": FAST_DEV})
    fast_dev = FAST_DEV

    async def safe(label: str, coro):
        try:
            await coro
            logger.info(f"✅ {label}")
        except Exception as e:  # pragma: no cover (dev convenience)
            if fast_dev:
                logger.warning(f"⚠️ FAST_DEV skip {label}: {e}")
            else:
                raise

    # Init components
    if init_db is not None:
        await safe("Database initialized", init_db())
    if get_redis_client is not None:
        try:
            redis_client = await get_redis_client()
            await redis_client.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            if fast_dev:
                logger.warning(f"⚠️ FAST_DEV Redis unavailable: {e}")
            else:
                raise
    await safe("AI models loaded", ai_orchestrator.initialize())
    await safe("Audio engine ready", audio_engine.initialize())
    await safe("Blockchain service connected", blockchain_service.initialize())
    logger.info(f"🎵 System ready (FAST_DEV={'ON' if fast_dev else 'OFF'})")

    yield

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
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=os.environ.get("CORS_ALLOW_CREDENTIALS", "true").lower() == "true",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

class _NoopMiddleware:
    def __init__(self, app):
        self.app = app
    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)

if not isinstance(RateLimitingMiddleware, type):
    RateLimitingMiddleware = _NoopMiddleware  # type: ignore
if not isinstance(PrometheusMiddleware, type):
    PrometheusMiddleware = _NoopMiddleware  # type: ignore
app.add_middleware(RateLimitingMiddleware)
app.add_middleware(PrometheusMiddleware)

# Mount static & frontend build (if present)
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
if Path("generated_audio").exists():
    app.mount("/audio", StaticFiles(directory="generated_audio"), name="audio")
if Path("dist").exists():  # built frontend assets
    app.mount("/", StaticFiles(directory="dist", html=True), name="frontend")

def _add_router(router, prefix: str, tags: Sequence[str]):  # Helper to avoid None errors in FAST_DEV
    if router is None:
        logger.warning(f"Skipping router {prefix} (module missing)")
        return
    app.include_router(router, prefix=prefix, tags=list(tags))

_add_router(auth_router, "/api/v1/auth", ["Authentication"])
_add_router(music_generation_router, "/api/v1/music", ["Music Generation"])
_add_router(projects_router, "/api/v1/projects", ["Projects"])
_add_router(collaboration_router, "/api/v1/collaboration", ["Collaboration"])
_add_router(marketplace_router, "/api/v1/marketplace", ["Marketplace"])
_add_router(analytics_router, "/api/v1/analytics", ["Analytics"])
_add_router(enterprise_router, "/api/v1/enterprise", ["Enterprise"])
_add_router(nft_router, "/api/v1/nft", ["NFT & Blockchain"])
_add_router(payments_router, "/api/v1/payments", ["Payments"])
_add_router(collab_lab_router, "/api/v1/collab-lab", ["Collab Lab"])
if ENV != "production":
    _add_router(dev_router, "/api/v1/dev", ["Development"])

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
        if get_db_session is None:
            db_status = "unknown"
        else:
            db_status = "healthy"
            try:
                gen = get_db_session()
                session = None
                if hasattr(gen, "__anext__"):
                    session = await gen.__anext__()
                else:
                    session = await gen  # type: ignore
                if hasattr(session, "execute"):
                    await session.execute("SELECT 1")
                if hasattr(session, "close"):
                    await session.close()
            except Exception as e:
                db_status = f"unhealthy: {str(e)}"
        
        # Check Redis
        if get_redis_client is None:
            redis_status = "unknown"
        else:
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

@app.get("/health/ready")
async def readiness_probe():
    """Readiness probe (strict) for orchestrators / load balancers.
    Returns 200 only if all core services are healthy.
    """
    result = await health_check()  # reuse logic; may return dict
    if isinstance(result, dict) and result.get("status") == "healthy":
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
    raise HTTPException(status_code=503, detail="not ready")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST  # type: ignore
    except Exception:
        raise HTTPException(status_code=503, detail="prometheus_client not installed")
    return StreamingResponse(io.BytesIO(generate_latest()), media_type=CONTENT_TYPE_LATEST)

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
        port=int(os.getenv("PORT", 8000)),
        reload=ENV != "production",
        workers=int(os.getenv("WORKERS", "1")),
        log_level=LOG_LEVEL.lower(),
    )
