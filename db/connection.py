"""
db/connection.py

Database connection management for the FDA Clinical Pharmacology Pipeline.

Provides:
- A lazily initialized SQLAlchemy engine built from Settings.
- A session factory (get_session) for use throughout the pipeline.
- A health check utility used by the Pipeline Status UI page.

Connection string is never logged. All credentials come from Settings,
which reads them exclusively from .env.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.engine import Engine

from config.settings import Settings
from utils.logger import get_module_logger

logger = get_module_logger(__name__)

# Module-level engine and session factory, initialized once via initialize_engine().
_engine: Engine | None = None
_session_factory: sessionmaker | None = None


def initialize_engine(settings: Settings) -> None:
    """
    Create and store the SQLAlchemy engine using credentials from Settings.
    Must be called once at application startup before any DB operations.

    Parameters
    ----------
    settings : Settings
        Fully loaded settings object. The connection string is read from
        settings.database.connection_string and is never written to logs.
    """
    global _engine, _session_factory

    if _engine is not None:
        logger.warning("initialize_engine called more than once — skipping re-initialization")
        return

    _engine = create_engine(
        settings.database.connection_string,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        echo=False,
    )

    _session_factory = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    logger.info(
        "Database engine initialized for host=%s db=%s",
        settings.database.host,
        settings.database.database,
    )


def get_engine() -> Engine:
    """
    Return the initialized SQLAlchemy engine.
    Raises RuntimeError if initialize_engine() has not been called.
    """
    if _engine is None:
        raise RuntimeError(
            "Database engine has not been initialized. "
            "Call initialize_engine(settings) at application startup."
        )
    return _engine


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager that yields a SQLAlchemy Session.
    Commits on clean exit; rolls back on any exception.

    Usage
    -----
    with get_session() as session:
        session.add(some_model_instance)

    Raises
    ------
    RuntimeError
        If initialize_engine() has not been called.
    """
    if _session_factory is None:
        raise RuntimeError(
            "Session factory has not been initialized. "
            "Call initialize_engine(settings) at application startup."
        )

    session: Session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception as database_error:
        session.rollback()
        logger.error("Session rolled back due to error: %s", database_error)
        raise
    finally:
        session.close()


def check_database_connection() -> dict:
    """
    Perform a lightweight connectivity check against PostgreSQL.
    Used by the Pipeline Status UI page.

    Returns
    -------
    dict
        {
            "connected": bool,
            "message": str,
            "postgres_version": str or None
        }
    """
    try:
        engine = get_engine()
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version()"))
            postgres_version = result.scalar()
        logger.info("Database connectivity check passed")
        return {
            "connected": True,
            "message": "Connection successful",
            "postgres_version": postgres_version,
        }
    except Exception as connectivity_error:
        logger.error("Database connectivity check failed: %s", connectivity_error)
        return {
            "connected": False,
            "message": str(connectivity_error),
            "postgres_version": None,
        }
