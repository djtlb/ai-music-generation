"""Development helper routes (NOT for production)."""
from fastapi import APIRouter, HTTPException
import os
from core.security import create_access_token

router = APIRouter()

@router.get("/token")
def dev_token(user_id: str = "demo-user", tier: str = "pro"):
    """Return a short-lived JWT for local testing.
    Disabled automatically when ENV=production.
    """
    if os.getenv("ENV", "development").lower() == "production":  # safety
        raise HTTPException(status_code=403, detail="Dev token endpoint disabled in production")
    token = create_access_token({"sub": user_id, "subscription_tier": tier})
    return {"token": token, "user_id": user_id, "tier": tier}
