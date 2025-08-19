"""Repository functions for persisting project, stage and event data.

Thin async helpers; safe to call even in high-frequency progress updates.
All functions swallow and log exceptions to avoid impacting the in-memory pipeline.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select

try:
    from config.settings import get_settings  # type: ignore
except ModuleNotFoundError:
    from backend.config.settings import get_settings  # type: ignore
try:
    from core.database import init_db  # type: ignore
except ModuleNotFoundError:
    from backend.core.database import init_db  # type: ignore
try:
    from models.project_models import Project, Stage, ProjectEvent  # type: ignore
except ModuleNotFoundError:
    from backend.models.project_models import Project, Stage, ProjectEvent  # type: ignore

logger = logging.getLogger(__name__)


async def _ensure_session():
    from core.database import async_session_maker as _maker  # local import to avoid circulars
    if _maker is None:
        await init_db()
    return _maker


async def create_project(project_id: str, name: str, user_id: str, style_config: dict):
    if not get_settings().persist_enabled:
        return
    try:
        maker = await _ensure_session()
        async with maker() as session:  # type: ignore
            project = Project(
                id=project_id,
                name=name,
                user_id=user_id,
                style_config=style_config,
                status="created",
            )
            session.add(project)
            await session.commit()
    except Exception as e:
        logger.warning(f"create_project failed: {e}")


async def update_project(project_id: str, **fields: Any):
    if not get_settings().persist_enabled or not fields:
        return
    try:
        maker = await _ensure_session()
        async with maker() as session:  # type: ignore
            result = await session.execute(select(Project).where(Project.id == project_id))
            project: Optional[Project] = result.scalar_one_or_none()
            if not project:
                return
            for k, v in fields.items():
                if hasattr(project, k):
                    setattr(project, k, v)
            await session.commit()
    except Exception as e:
        logger.warning(f"update_project failed: {e}")


async def stage_start(project_id: str, stage: str):
    if not get_settings().persist_enabled:
        return
    try:
        maker = await _ensure_session()
        async with maker() as session:  # type: ignore
            result = await session.execute(
                select(Stage).where(Stage.project_id == project_id, Stage.name == stage)
            )
            st: Optional[Stage] = result.scalar_one_or_none()
            now = datetime.utcnow()
            if not st:
                st = Stage(project_id=project_id, name=stage, status="in_progress", start=now, progress=0)
                session.add(st)
            else:
                st.status = "in_progress"
                if not st.start:
                    st.start = now
            await session.commit()
            await record_event(project_id, "stage.start", stage=stage, data={})
    except Exception as e:
        logger.warning(f"stage_start failed: {e}")


async def stage_progress(project_id: str, stage: str, progress: int):
    if not get_settings().persist_enabled:
        return
    try:
        maker = await _ensure_session()
        async with maker() as session:  # type: ignore
            result = await session.execute(
                select(Stage).where(Stage.project_id == project_id, Stage.name == stage)
            )
            st: Optional[Stage] = result.scalar_one_or_none()
            if st:
                st.progress = progress
                await session.commit()
    except Exception as e:
        logger.debug(f"stage_progress skipped: {e}")


async def stage_complete(project_id: str, stage: str):
    if not get_settings().persist_enabled:
        return
    try:
        maker = await _ensure_session()
        async with maker() as session:  # type: ignore
            result = await session.execute(
                select(Stage).where(Stage.project_id == project_id, Stage.name == stage)
            )
            st: Optional[Stage] = result.scalar_one_or_none()
            now = datetime.utcnow()
            if st:
                if not st.start:
                    st.start = now
                if not st.end:
                    st.end = now
                st.status = "completed"
                st.progress = 100
                try:
                    st.duration_sec = int((st.end - st.start).total_seconds()) if st.end and st.start else None
                except Exception:
                    pass
                await session.commit()
            await record_event(project_id, "stage.completed", stage=stage, data={})
    except Exception as e:
        logger.warning(f"stage_complete failed: {e}")


async def project_completed(project_id: str):
    await update_project(project_id, status="completed", completed_at=datetime.utcnow())
    await record_event(project_id, "project.completed")


async def project_failed(project_id: str, error: str):
    await update_project(project_id, status="failed", error=error)
    await record_event(project_id, "project.failed", data={"error": error})


async def record_event(project_id: str, event_type: str, stage: Optional[str] = None, data: Optional[dict] = None):
    if not get_settings().persist_enabled:
        return
    try:
        maker = await _ensure_session()
        async with maker() as session:  # type: ignore
            evt = ProjectEvent(project_id=project_id, event_type=event_type, stage=stage, data=data or {})
            session.add(evt)
            await session.commit()
    except Exception as e:
        logger.debug(f"record_event skipped: {e}")
