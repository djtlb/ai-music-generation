from fastapi import APIRouter

router = APIRouter()

@router.get("/summary")
async def summary():
    return {"events": 0, "users": 0}
