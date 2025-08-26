
from fastapi import APIRouter
from app.api.endpoints import auth, music, projects

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(music.router, prefix="/music", tags=["Music Generation"])
api_router.include_router(projects.router, prefix="/projects", tags=["Projects"])
# The following routers are commented out as they are not currently supported/active:
# from app.api.endpoints import collaboration, marketplace, analytics, enterprise, nft, payments, collab_lab, dev
# api_router.include_router(collaboration.router, prefix="/collaboration", tags=["Collaboration"])
# api_router.include_router(marketplace.router, prefix="/marketplace", tags=["Marketplace"])
# api_router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
# api_router.include_router(enterprise.router, prefix="/enterprise", tags=["Enterprise"])
# api_router.include_router(nft.router, prefix="/nft", tags=["NFT & Blockchain"])
# api_router.include_router(payments.router, prefix="/payments", tags=["Payments"])
# api_router.include_router(collab_lab.router, prefix="/collab-lab", tags=["Collab Lab"])
# api_router.include_router(dev.router, prefix="/dev", tags=["Development"])
