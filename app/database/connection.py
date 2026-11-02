"""
Database connection and session factory for TreadVision
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.config import DATABASE_PATH, DATABASE_URL
from app.database.models import Base

# Ensure parent directory exists
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite in Streamlit multi-threading
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables in the database if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_db_session() -> Session:
    """Yield a database session context."""
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise
