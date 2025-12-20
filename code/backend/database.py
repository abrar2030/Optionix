from typing import Any
import logging
from typing import Any
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from .config import settings
from .models import Base

logger = logging.getLogger(__name__)

# Determine pool class based on database type
database_url = settings.database_url
if database_url.startswith("sqlite"):
    # SQLite needs special pooling
    poolclass = StaticPool if ":memory:" in database_url else NullPool
    connect_args = {"check_same_thread": False}
    pool_pre_ping = False
else:
    poolclass = QueuePool
    connect_args = {}
    pool_pre_ping = True

try:
    engine = create_engine(
        database_url,
        poolclass=poolclass,
        pool_size=settings.database_pool_size if poolclass == QueuePool else None,
        max_overflow=settings.database_max_overflow if poolclass == QueuePool else None,
        pool_pre_ping=pool_pre_ping,
        echo=settings.debug,
        connect_args=connect_args,
    )
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    # Fallback to SQLite in-memory
    logger.warning("Falling back to SQLite in-memory database")
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables() -> Any:
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def get_db() -> Session:
    """
    Dependency to get database session

    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """
    Get a database session for direct use

    Returns:
        Session: SQLAlchemy database session
    """
    return SessionLocal()
