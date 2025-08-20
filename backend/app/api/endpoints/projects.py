from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import uuid

router = APIRouter()


class Project(BaseModel):
    id: str
    name: str
    status: str
    created_at: str
    audio_url: Optional[str] = None


@router.get("/", response_model=List[Project])
async def get_projects():
    """Get all projects for user."""
    return [
        Project(
            id=str(uuid.uuid4()),
            name="My First Song",
            status="completed",
            created_at="2024-01-01T00:00:00Z",
            audio_url="/audio/song1.mp3"
        ),
        Project(
            id=str(uuid.uuid4()),
            name="Electronic Vibes",
            status="processing",
            created_at="2024-01-02T00:00:00Z"
        )
    ]


@router.get("/{project_id}/status")
async def get_project_status(project_id: str):
    """Get project status."""
    return {
        "project_id": project_id,
        "status": "completed",
        "progress": 100,
        "stages_completed": 5,
        "total_stages": 5
    }
