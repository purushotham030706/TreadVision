"""
Unit tests for TreadVision Database Models and Queries
"""
import pytest
from app.database.connection import get_db_session
from app.database.models import Tyre, TreadPattern, TyreSize, Vehicle, Fitment, TreadFeatures


def test_database_queries():
    session = get_db_session()
    try:
        # Check tyres count
        tyres = session.query(Tyre).all()
        assert len(tyres) >= 6

        # Check vehicles count
        vehicles = session.query(Vehicle).all()
        assert len(vehicles) >= 6

        # Check fitments count
        fitments = session.query(Fitment).all()
        assert len(fitments) >= 10

        # Check Michelin Primacy 4
        michelin = session.query(Tyre).filter(Tyre.manufacturer == "Michelin", Tyre.model == "Primacy 4").first()
        assert michelin is not None
        assert michelin.tread_pattern is not None
        assert michelin.features is not None
        assert len(michelin.sizes) > 0

        # Check features deserialization
        vec = michelin.features.get_feature_vector()
        assert len(vec) == 25
    finally:
        session.close()
