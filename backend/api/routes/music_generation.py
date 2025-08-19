"""
Music Generation API Endpoints
The core endpoints for AI-powered music creation
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from core.security import get_current_user, check_subscription_tier
from services.ai_orchestrator import get_ai_orchestrator
from services.audio_engine import get_audio_engine  # type: ignore
from core.redis_client import cache  # runtime-provided client (may be stubbed)
try:
    from config.settings import get_settings  # type: ignore
except ModuleNotFoundError:
    from backend.config.settings import get_settings  # type: ignore
from sqlalchemy import select
try:
    from core.database import init_db  # type: ignore
except ModuleNotFoundError:
    from backend.core.database import init_db  # type: ignore
try:
    from models.project_models import Project  # type: ignore
except ModuleNotFoundError:
    from backend.models.project_models import Project  # type: ignore

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models
class StyleConfig(BaseModel):
    genre: str = Field(..., description="Music genre (pop, rock, jazz, etc.)")
    subgenre: Optional[str] = Field(None, description="Specific subgenre")
    energy: float = Field(0.7, ge=0.0, le=1.0, description="Energy level (0-1)")
    mood: str = Field("uplifting", description="Overall mood")
    tempo: int = Field(120, ge=60, le=200, description="BPM")
    key: str = Field("C", description="Musical key")
    time_signature: str = Field("4/4", description="Time signature")

class LyricsRequest(BaseModel):
    theme: str = Field(..., description="Lyrical theme or topic")
    style: str = Field("pop", description="Lyrical style")
    mood: str = Field("uplifting", description="Emotional mood")
    language: str = Field("english", description="Language")
    explicit: bool = Field(False, description="Allow explicit content")

class ArrangementRequest(BaseModel):
    style_config: StyleConfig
    duration: int = Field(180, ge=30, le=600, description="Duration in seconds")
    complexity: str = Field("medium", description="Arrangement complexity")
    instruments: List[str] = Field(default_factory=list, description="Preferred instruments")

class CompositionRequest(BaseModel):
    arrangement_id: str
    style_config: StyleConfig
    lyrics_id: Optional[str] = None
    advanced_options: Dict[str, Any] = Field(default_factory=dict)

class SoundDesignRequest(BaseModel):
    composition_id: str
    style_config: StyleConfig
    audio_quality: str = Field("high", description="Audio quality preset")
    effects_preset: str = Field("modern", description="Effects preset")

class MixMasterRequest(BaseModel):
    sound_design_id: str
    mastering_style: str = Field("modern", description="Mastering style")
    loudness_target: float = Field(-14.0, description="LUFS target")
    dynamic_range: str = Field("medium", description="Dynamic range preference")

class FullSongRequest(BaseModel):
    project_name: str = Field(..., description="Project name")
    style_config: StyleConfig
    lyrics_request: Optional[LyricsRequest] = None
    advanced_options: Dict[str, Any] = Field(default_factory=dict)
    collaboration_enabled: bool = Field(False, description="Enable real-time collaboration")

@router.post("/generate/lyrics")
async def generate_lyrics(
    request: LyricsRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    ai_orchestrator = Depends(get_ai_orchestrator)
):
    """Generate AI-powered lyrics"""
    try:
        user_id = current_user["user_id"]
        
        # Check rate limits
        cache_key = f"lyrics_generation:{user_id}"
        recent_generations = await cache.get(cache_key) or 0
        
        if recent_generations >= 10:  # 10 per hour limit
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Start generation
        task_id = await ai_orchestrator.generate_lyrics(
            theme=request.theme,
            style=request.style,
            mood=request.mood,
            language=request.language,
            explicit=request.explicit,
            user_id=user_id
        )
        
        # Update rate limit
        await cache.increment(cache_key)
        # cache.client may be a redis instance; ignore type checker if abstraction differs
        if getattr(cache, 'client', None):  # lightweight safety
            try:
                await cache.client.expire(cache_key, 3600)  # type: ignore[attr-defined]
            except Exception:
                pass  # non-fatal
        
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Lyrics generation started",
            "estimated_completion": "2-3 minutes"
        }
        
    except Exception as e:
        logger.error(f"Lyrics generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate/arrangement")
async def generate_arrangement(
    request: ArrangementRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    ai_orchestrator = Depends(get_ai_orchestrator)
):
    """Generate song arrangement and structure"""
    try:
        user_id = current_user["user_id"]
        
        task_id = await ai_orchestrator.generate_arrangement(
            style_config=request.style_config.dict(),
            duration=request.duration,
            complexity=request.complexity,
            instruments=request.instruments,
            user_id=user_id
        )
        
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Arrangement generation started",
            "estimated_completion": "1-2 minutes"
        }
        
    except Exception as e:
        logger.error(f"Arrangement generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate/composition")
async def generate_composition(
    request: CompositionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(check_subscription_tier("pro")),
    ai_orchestrator = Depends(get_ai_orchestrator)
):
    """Generate full musical composition with melody and harmony"""
    try:
        user_id = current_user["user_id"]
        
        task_id = await ai_orchestrator.generate_composition(
            arrangement_id=request.arrangement_id,
            style_config=request.style_config.dict(),
            lyrics_id=request.lyrics_id,
            advanced_options=request.advanced_options,
            user_id=user_id
        )
        
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Composition generation started",
            "estimated_completion": "3-5 minutes"
        }
        
    except Exception as e:
        logger.error(f"Composition generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate/sound-design")
async def generate_sound_design(
    request: SoundDesignRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(check_subscription_tier("pro")),
    ai_orchestrator = Depends(get_ai_orchestrator)
):
    """Generate sound design and audio textures"""
    try:
        user_id = current_user["user_id"]
        
        task_id = await ai_orchestrator.generate_sound_design(
            composition_id=request.composition_id,
            style_config=request.style_config.dict(),
            audio_quality=request.audio_quality,
            effects_preset=request.effects_preset,
            user_id=user_id
        )
        
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Sound design generation started",
            "estimated_completion": "2-4 minutes"
        }
        
    except Exception as e:
        logger.error(f"Sound design generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate/mix-master")
async def generate_mix_master(
    request: MixMasterRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(check_subscription_tier("pro")),
    ai_orchestrator = Depends(get_ai_orchestrator),
    audio_engine = Depends(get_audio_engine)
):
    """Generate final mix and master"""
    try:
        user_id = current_user["user_id"]
        
        task_id = await ai_orchestrator.generate_mix_master(
            sound_design_id=request.sound_design_id,
            mastering_style=request.mastering_style,
            loudness_target=request.loudness_target,
            dynamic_range=request.dynamic_range,
            user_id=user_id
        )
        
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Mix and master started",
            "estimated_completion": "3-6 minutes"
        }
        
    except Exception as e:
        logger.error(f"Mix/master generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate/full-song")
async def generate_full_song(
    request: FullSongRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    ai_orchestrator = Depends(get_ai_orchestrator)
):
    """
    🎵 MILLION-DOLLAR ENDPOINT 🎵
    Generate a complete song from concept to final master
    This is the flagship feature that ties everything together
    """
    try:
        user_id = current_user["user_id"]
        
        # Check subscription limits
        user_tier = current_user.get("payload", {}).get("subscription_tier", "free")
        monthly_usage = await cache.get(f"monthly_songs:{user_id}") or 0
        
        tier_limits = {"free": 3, "pro": 100, "enterprise": -1}
        if tier_limits[user_tier] != -1 and monthly_usage >= tier_limits[user_tier]:
            raise HTTPException(
                status_code=402, 
                detail=f"Monthly limit reached for {user_tier} tier. Upgrade for more songs."
            )
        
        # Create project and start full pipeline
        project_id = await ai_orchestrator.create_project(
            name=request.project_name,
            user_id=user_id,
            style_config=request.style_config.dict()
        )
        
        # Start the full generation pipeline in background
        background_tasks.add_task(
            ai_orchestrator.generate_full_song_pipeline,
            project_id=project_id,
            style_config=request.style_config.dict(),
            lyrics_request=request.lyrics_request.dict() if request.lyrics_request else None,
            advanced_options=request.advanced_options,
            user_id=user_id,
            collaboration_enabled=request.collaboration_enabled
        )
        
        # Update usage counter
        await cache.increment(f"monthly_songs:{user_id}")
        
        return {
            "project_id": project_id,
            "status": "processing",
            "message": "🎵 Full song generation started! This is where the magic happens.",
            "estimated_completion": "8-12 minutes",
            "pipeline_stages": [
                "Lyrics Generation",
                "Song Arrangement", 
                "Melody & Harmony Composition",
                "Sound Design & Synthesis",
                "Professional Mix & Master"
            ],
            "websocket_url": f"/ws/{user_id}",
            "collaboration_enabled": request.collaboration_enabled,
            "premium_features": {
                "ai_vocals": user_tier in ["pro", "enterprise"],
                "stem_separation": user_tier in ["pro", "enterprise"], 
                "commercial_license": user_tier == "enterprise",
                "priority_processing": user_tier in ["pro", "enterprise"]
            }
        }
        
    except Exception as e:
        logger.error(f"Full song generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/task/{task_id}/status")
async def get_task_status(
    task_id: str,
    current_user: dict = Depends(get_current_user),
    ai_orchestrator = Depends(get_ai_orchestrator)
):
    """Get status of a generation task"""
    try:
        status = await ai_orchestrator.get_task_status(task_id)
        return status
        
    except Exception as e:
        logger.error(f"Task status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/project/{project_id}/status")
async def get_project_status(
    project_id: str,
    current_user: dict = Depends(get_current_user),
    ai_orchestrator = Depends(get_ai_orchestrator)
):
    """Get aggregated status/progress of a project"""
    try:
        project = await ai_orchestrator.get_project_status(project_id)
        if project.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Project not found")
        # Derive progress if not already set
        stages = project.get("stages", {})
        total = len(stages) if stages else 0
        completed = sum(1 for v in stages.values() if v)
        progress = project.get("progress")
        if progress is None and total:
            progress = int((completed / total) * 100)
        return {
            "project_id": project_id,
            "name": project.get("name"),
            "status": project.get("status"),
            "created_at": project.get("created_at"),
            "completed_at": project.get("completed_at"),
            "stages": {k: (v is not None) for k, v in stages.items()},
            "progress": progress or 0
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Project status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/project/{project_id}/aggregate")
async def get_project_aggregate(
    project_id: str,
    current_user: dict = Depends(get_current_user),
    ai_orchestrator = Depends(get_ai_orchestrator)
):
    """Get detailed aggregate project info including stage metadata and progress"""
    try:
        agg = await ai_orchestrator.get_project_aggregate(project_id)
        if agg.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Project not found")
        return agg
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Project aggregate error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{file_id}")
async def download_file(
    file_id: str,
    current_user: dict = Depends(get_current_user),
    audio_engine = Depends(get_audio_engine)
):
    """Download generated audio file"""
    try:
        file_path = await audio_engine.get_file_path(file_id, current_user["user_id"])
        
        if not file_path:
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            media_type="audio/wav",
            filename=f"generated_song_{file_id}.wav"
        )
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/reference")
async def upload_reference_audio(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    audio_engine = Depends(get_audio_engine)
):
    """Upload reference audio for style transfer"""
    try:
        if file.content_type not in ["audio/wav", "audio/mp3", "audio/flac"]:
            raise HTTPException(status_code=400, detail="Invalid audio format")
        size_val = getattr(file, 'size', None)
        if isinstance(size_val, int) and size_val > 50 * 1024 * 1024:  # 50MB limit (size may be None in some frameworks)
            raise HTTPException(status_code=400, detail="File too large")

        file_id = await audio_engine.save_reference_audio(
            file=file,
            user_id=current_user["user_id"]
        )

        return {
            "file_id": file_id,
            "message": "Reference audio uploaded successfully",
            "processing_status": "analyzing"
        }
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/styles/presets")
async def get_style_presets():
    """Get available style presets"""
    return {
        "genres": [
            {"id": "pop", "name": "Pop", "description": "Modern pop music"},
            {"id": "rock", "name": "Rock", "description": "Rock and alternative"},
            {"id": "jazz", "name": "Jazz", "description": "Jazz and fusion"},
            {"id": "electronic", "name": "Electronic", "description": "EDM and electronic"},
            {"id": "classical", "name": "Classical", "description": "Classical and orchestral"},
            {"id": "hip-hop", "name": "Hip-Hop", "description": "Hip-hop and rap"},
            {"id": "country", "name": "Country", "description": "Country and folk"},
            {"id": "blues", "name": "Blues", "description": "Blues and soul"}
        ],
        "moods": [
            "uplifting", "melancholic", "energetic", "calm", 
            "mysterious", "romantic", "aggressive", "peaceful"
        ],
        "keys": [
            "C", "C#", "D", "D#", "E", "F", 
            "F#", "G", "G#", "A", "A#", "B"
        ],
        "time_signatures": ["4/4", "3/4", "6/8", "2/4", "5/4"]
    }

@router.get("/analytics/generation-stats")
async def get_generation_stats(
    current_user: dict = Depends(get_current_user)
):
    """Get user's generation statistics"""
    try:
        user_id = current_user["user_id"]
        
        stats = {
            "total_songs": await cache.get(f"user_songs:{user_id}") or 0,
            "monthly_usage": await cache.get(f"monthly_songs:{user_id}") or 0,
            "favorite_genre": await cache.get(f"user_genre:{user_id}") or "pop",
            "total_playtime": await cache.get(f"user_playtime:{user_id}") or 0,
            "subscription_tier": current_user.get("payload", {}).get("subscription_tier", "free")
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects")
async def list_projects(
    limit: int = 20,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    ai_orchestrator = Depends(get_ai_orchestrator)
):
    """List recent projects for the current user (DB-backed if persistence enabled)."""
    try:
        user_id = current_user["user_id"]
        settings = get_settings()
        projects: List[Dict[str, Any]] = []
        if settings.persist_enabled:
            from core.database import async_session_maker as maker
            if maker is None:
                await init_db()
            if maker is None:
                raise HTTPException(status_code=500, detail="Database not initialized")
            async with maker() as session:  # type: ignore
                stmt = (
                    select(Project)
                    .where(Project.user_id == user_id)
                    .order_by(Project.created_at.desc())
                    .offset(offset)
                    .limit(limit)
                )
                result = await session.execute(stmt)
                rows = result.scalars().all()
                for p in rows:
                    projects.append({
                        "id": p.id,
                        "name": p.name,
                        "status": p.status,
                        "progress": p.progress,
                        "created_at": p.created_at.isoformat() if p.created_at else None,
                        "completed_at": p.completed_at.isoformat() if p.completed_at else None,
                        "error": p.error,
                    })
        else:
            for proj in ai_orchestrator.projects.values():
                if proj.get("user_id") == user_id:
                    projects.append({
                        "id": proj.get("id"),
                        "name": proj.get("name"),
                        "status": proj.get("status"),
                        "progress": proj.get("progress"),
                        "created_at": proj.get("created_at"),
                        "completed_at": proj.get("completed_at"),
                        "error": proj.get("error"),
                    })
            projects.sort(key=lambda x: x.get("created_at") or "", reverse=True)
            projects = projects[offset: offset + limit]
        return {"items": projects, "count": len(projects), "limit": limit, "offset": offset}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List projects error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
