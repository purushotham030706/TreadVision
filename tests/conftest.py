"""
Pytest configuration and isolated database fixtures for TreadVision.
Ensures zero coupling with data/database/treadvision.db.
"""
import pytest
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base
from app.database.connection import init_db
from app.database.seed_data import seed_database


@pytest.fixture(scope="function")
def isolated_engine(tmp_path):
    """Create an isolated, temporary SQLite database for each test."""
    db_file = tmp_path / "isolated_test_treadvision.db"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
        echo=False
    )
    init_db(engine)
    seed_database(target_engine=engine)
    yield engine
    engine.dispose()
    if db_file.exists():
        try:
            db_file.unlink()
        except Exception:
            pass


@pytest.fixture(scope="function")
def isolated_db_session(isolated_engine):
    """Provide a scoped session connected strictly to the isolated test database."""
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=isolated_engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
