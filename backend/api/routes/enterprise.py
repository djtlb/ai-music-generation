from fastapi import APIRouter

router = APIRouter()

@router.get("/features")
async def features():
    return {"enterprise": False}
