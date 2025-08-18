from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/status")
async def auth_status():
    return {"service": "auth", "status": "ok"}
