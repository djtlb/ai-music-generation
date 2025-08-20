"""
Database configuration and connection management
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData
from config.settings import get_settings
import logging

logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    """Base class for all database models"""
    metadata = MetaData()

# Database engine
engine = None
async_session_maker = None

async def init_db():
    """Initialize database connection"""
    global engine, async_session_maker
    
    settings = get_settings()
    
    # Create async engine
    engine = create_async_engine(
        settings.database_url,
        echo=settings.database_echo,
        pool_size=20,
        max_overflow=0,
        pool_pre_ping=True,
        pool_recycle=3600
    )
    
    # Create session maker
    async_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    # Auto-create tables if persistence enabled (simple dev path; replace with Alembic for prod)
    from config.settings import get_settings as _gs
    try:
        if _gs().persist_enabled:
            from models.project_models import Project, Stage, ProjectEvent  # noqa: F401
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables ensured (persist_enabled)")
    except Exception as e:
        logger.warning(f"Table creation skipped/failed: {e}")
    logger.info("Database connection initialized")

async def get_db_session() -> AsyncSession:
    """Get database session"""
    if async_session_maker is None:
        await init_db()
    if async_session_maker is None:  # safeguard
        raise RuntimeError("Database session maker not initialized")
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def close_db():
    """Close database connection"""
    global engine
    if engine:
        await engine.dispose()
        logger.info("Database connection closed")
