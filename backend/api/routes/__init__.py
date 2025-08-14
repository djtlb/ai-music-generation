"""
API Routes for the AI Music Generation Platform
"""

from .music_generation import router as music_generation_router
from .auth import router as auth_router
from .projects import router as projects_router
from .collaboration import router as collaboration_router
from .marketplace import router as marketplace_router
from .analytics import router as analytics_router
from .enterprise import router as enterprise_router
from .nft import router as nft_router
from .payments import router as payments_router

__all__ = [
    "auth_router",
    "music_generation_router", 
    "projects_router",
    "collaboration_router",
    "marketplace_router",
    "analytics_router",
    "enterprise_router",
    "nft_router",
    "payments_router"
]
