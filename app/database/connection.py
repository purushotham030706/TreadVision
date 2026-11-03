"""
TreadVision: Database Connection, Session Factory, and Integrity Verification
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, Session

from app.config import DATABASE_PATH, DATABASE_URL
from app.database.models import Base, Tyre, TreadPattern, TyreSize, Vehicle, Fitment, TreadFeatures

# Ensure parent directory exists
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(target_engine=None):
    """Create all tables in the database if they do not exist."""
    b_engine = target_engine or engine
    Base.metadata.create_all(bind=b_engine)


def get_db_session(target_engine=None) -> Session:
    """Yield a database session context."""
    if target_engine is not None:
        custom_session_maker = sessionmaker(autocommit=False, autoflush=False, bind=target_engine)
        return custom_session_maker()
    db = SessionLocal()
    return db


def verify_database_integrity(
    engine_instance=None,
    session_instance: Optional[Session] = None
) -> Dict[str, Any]:
    """
    Verify schema integrity, columns, foreign keys, and seed data consistency.
    Returns:
        Dict reporting status, counts, missing tables/columns, and foreign key integrity.
    """
    eng = engine_instance or engine
    close_session = False
    sess = session_instance
    if sess is None:
        sess = get_db_session(eng)
        close_session = True

    required_tables = {
        "tyres": ["id", "manufacturer", "model", "pattern_name", "category", "construction"],
        "tread_patterns": ["id", "tyre_id", "image_path", "pattern_type", "symmetry"],
        "tyre_sizes": ["id", "tyre_id", "width", "aspect_ratio", "rim_diameter"],
        "vehicles": ["id", "manufacturer", "model"],
        "fitments": ["id", "tyre_id", "vehicle_id", "size"],
        "tread_features": ["id", "tyre_id", "edge_features", "texture_features", "shape_features", "feature_vector"]
    }

    inspector = inspect(eng)
    existing_tables = set(inspector.get_table_names())

    missing_tables: List[str] = []
    missing_columns: Dict[str, List[str]] = {}
    tables_checked: Dict[str, bool] = {}

    for tbl, req_cols in required_tables.items():
        if tbl not in existing_tables:
            missing_tables.append(tbl)
            tables_checked[tbl] = False
        else:
            tables_checked[tbl] = True
            existing_cols = {c["name"] for c in inspector.get_columns(tbl)}
            col_diff = [c for c in req_cols if c not in existing_cols]
            if col_diff:
                missing_columns[tbl] = col_diff

    # Row counts and relationship verification if tables exist
    counts: Dict[str, int] = {}
    relationship_errors: List[str] = []
    fk_check_passed = True

    try:
        # Check SQLite foreign key PRAGMA check
        with eng.connect() as conn:
            fk_res = conn.execute(text("PRAGMA foreign_key_check")).fetchall()
            if fk_res:
                fk_check_passed = False
                for row in fk_res:
                    relationship_errors.append(f"Foreign key constraint violation in row {row}")

        if not missing_tables:
            counts["tyres"] = sess.query(Tyre).count()
            counts["tread_patterns"] = sess.query(TreadPattern).count()
            counts["tyre_sizes"] = sess.query(TyreSize).count()
            counts["vehicles"] = sess.query(Vehicle).count()
            counts["fitments"] = sess.query(Fitment).count()
            counts["tread_features"] = sess.query(TreadFeatures).count()

            # Verify relationship references
            tyre_ids = set(t.id for t in sess.query(Tyre.id).all())
            vehicle_ids = set(v.id for v in sess.query(Vehicle.id).all())

            # Check pattern references
            for p in sess.query(TreadPattern).all():
                if p.tyre_id not in tyre_ids:
                    relationship_errors.append(f"TreadPattern id={p.id} references non-existent tyre_id={p.tyre_id}")

            # Check fitment references
            for f in sess.query(Fitment).all():
                if f.tyre_id not in tyre_ids:
                    relationship_errors.append(f"Fitment id={f.id} references non-existent tyre_id={f.tyre_id}")
                if f.vehicle_id not in vehicle_ids:
                    relationship_errors.append(f"Fitment id={f.id} references non-existent vehicle_id={f.vehicle_id}")

            # Check features references
            for feat in sess.query(TreadFeatures).all():
                if feat.tyre_id not in tyre_ids:
                    relationship_errors.append(f"TreadFeatures id={feat.id} references non-existent tyre_id={feat.tyre_id}")
    except Exception as e:
        relationship_errors.append(f"Database query error during verification: {str(e)}")
    finally:
        if close_session:
            sess.close()

    is_valid = (
        len(missing_tables) == 0 and
        len(missing_columns) == 0 and
        len(relationship_errors) == 0 and
        fk_check_passed and
        counts.get("tyres", 0) >= 6 and
        counts.get("vehicles", 0) >= 6 and
        counts.get("fitments", 0) >= 6 and
        counts.get("tread_features", 0) >= 6
    )

    return {
        "valid": is_valid,
        "tables_checked": tables_checked,
        "missing_tables": missing_tables,
        "missing_columns": missing_columns,
        "counts": counts,
        "relationship_errors": relationship_errors,
        "fk_check_passed": fk_check_passed
    }
