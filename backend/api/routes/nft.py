from fastapi import APIRouter

router = APIRouter()

@router.get("/contracts")
async def contracts():
    return {"contracts": []}
