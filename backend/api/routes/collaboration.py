from fastapi import APIRouter

router = APIRouter()

@router.get("/sessions")
async def list_sessions():
    return {"sessions": []}
