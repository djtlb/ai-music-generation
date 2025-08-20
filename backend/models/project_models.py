"""Database models for persisting projects, stages, and events."""
from sqlalchemy import String, DateTime, Integer, ForeignKey, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from ..core.database import Base

class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="created", index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    style_config: Mapped[dict] = mapped_column(JSON, default={})
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    stages: Mapped[list["Stage"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    events: Mapped[list["ProjectEvent"]] = relationship(back_populates="project", cascade="all, delete-orphan")

class Stage(Base):
    __tablename__ = "project_stages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="in_progress")
    start: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    end: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_sec: Mapped[int | None] = mapped_column(Integer, nullable=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    meta: Mapped[dict] = mapped_column(JSON, default={})

    project: Mapped[Project] = relationship(back_populates="stages")

class ProjectEvent(Base):
    __tablename__ = "project_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    data: Mapped[dict] = mapped_column(JSON, default={})
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    project: Mapped[Project] = relationship(back_populates="events")
