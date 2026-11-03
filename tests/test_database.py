"""
Unit tests for TreadVision Database Models, Integrity Verification, and Idempotency
All tests run against isolated temporary SQLite databases.
"""
import pytest
from sqlalchemy import create_engine
from app.database.connection import verify_database_integrity, init_db
from app.database.seed_data import seed_database
from app.database.models import Tyre, TreadPattern, TyreSize, Vehicle, Fitment, TreadFeatures


def test_isolated_database_queries(isolated_db_session):
    """Verify core records in isolated test database."""
    # Check tyres count
    tyres = isolated_db_session.query(Tyre).all()
    assert len(tyres) >= 6

    # Check vehicles count
    vehicles = isolated_db_session.query(Vehicle).all()
    assert len(vehicles) >= 6

    # Check fitments count
    fitments = isolated_db_session.query(Fitment).all()
    assert len(fitments) >= 10

    # Check Michelin Primacy 4
    michelin = isolated_db_session.query(Tyre).filter(
        Tyre.manufacturer == "Michelin", Tyre.model == "Primacy 4"
    ).first()
    assert michelin is not None
    assert michelin.tread_pattern is not None
    assert michelin.features is not None
    assert len(michelin.sizes) > 0

    # Check features deserialization (31-D augmented feature vector)
    vec = michelin.features.get_feature_vector()
    assert len(vec) == 31


def test_database_integrity_verification(isolated_engine, isolated_db_session):
    """Verify that a valid seeded database passes integrity checks."""
    integrity = verify_database_integrity(
        engine_instance=isolated_engine,
        session_instance=isolated_db_session
    )
    assert integrity["valid"] is True
    assert integrity["fk_check_passed"] is True
    assert len(integrity["missing_tables"]) == 0
    assert len(integrity["missing_columns"]) == 0
    assert len(integrity["relationship_errors"]) == 0
    assert integrity["counts"]["tyres"] >= 6
    assert integrity["counts"]["vehicles"] >= 6


def test_database_integrity_detects_incomplete_db(tmp_path):
    """Verify that an empty or incomplete database fails integrity check with descriptive errors."""
    empty_db = tmp_path / "empty.db"
    empty_engine = create_engine(f"sqlite:///{empty_db}")

    integrity = verify_database_integrity(engine_instance=empty_engine)
    assert integrity["valid"] is False
    assert len(integrity["missing_tables"]) > 0
    assert "tyres" in integrity["missing_tables"]
    empty_engine.dispose()


def test_database_idempotent_seeding(isolated_engine, isolated_db_session):
    """Verify that calling seed_database multiple times does not produce duplicate records."""
    count_tyres_before = isolated_db_session.query(Tyre).count()
    count_vehicles_before = isolated_db_session.query(Vehicle).count()
    count_fitments_before = isolated_db_session.query(Fitment).count()

    # Re-run seed with force_reseed=False
    seed_database(target_engine=isolated_engine, force_reseed=False)

    count_tyres_after = isolated_db_session.query(Tyre).count()
    count_vehicles_after = isolated_db_session.query(Vehicle).count()
    count_fitments_after = isolated_db_session.query(Fitment).count()

    assert count_tyres_before == count_tyres_after
    assert count_vehicles_before == count_vehicles_after
    assert count_fitments_before == count_fitments_after
