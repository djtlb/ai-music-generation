from fastapi import APIRouter

router = APIRouter()

@router.get("/items")
async def list_items():
    return {"items": []}
