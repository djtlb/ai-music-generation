from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import uuid

router = APIRouter()


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """User login endpoint."""
    # Mock authentication - replace with real auth logic
    if request.email and request.password:
        return TokenResponse(
            access_token="mock-jwt-token",
            token_type="bearer",
            user_id=str(uuid.uuid4())
        )
    raise HTTPException(status_code=401, detail="Invalid credentials")


@router.post("/register")
async def register(request: LoginRequest):
    """User registration endpoint."""
    return {"message": "User registered successfully", "user_id": str(uuid.uuid4())}


@router.get("/me")
async def get_current_user():
    """Get current user info."""
    return {"user_id": "mock-user-id", "email": "user@example.com", "premium": True}
