"""Central logging configuration utilities."""
from __future__ import annotations
import logging
import os

try:
    import structlog  # type: ignore
except Exception:  # pragma: no cover
    structlog = None  # type: ignore

ENV = os.getenv("ENV", "development").lower()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_configured = False

def configure_logging(force: bool=False) -> None:
    global _configured
    if _configured and not force:
        return
    if ENV == "production" and structlog:
        logging.basicConfig(level=LOG_LEVEL, format="%(message)s")
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, LOG_LEVEL, logging.INFO)),
            cache_logger_on_first_use=True,
        )
    else:
        logging.basicConfig(
            level=LOG_LEVEL,
            format="[%(levelname)s] %(asctime)s %(name)s: %(message)s"
        )
    _configured = True
