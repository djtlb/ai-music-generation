from fastapi import APIRouter

router = APIRouter()

@router.get("/plans")
async def plans():
    return {"plans": ["free", "pro", "enterprise"]}
