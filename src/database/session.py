"""
Shared SQLAlchemy session context manager.

Every miner (and the model trainer) uses the same commit / rollback / close
semantics, so the context manager lives here rather than being copy-pasted.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy.orm import Session

from src.database.models import SessionLocal


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session with commit / rollback / close semantics."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
