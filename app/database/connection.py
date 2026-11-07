"""
TreadVision: Database Connection, Session Factory, and Integrity Verification
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
from sqlalchemy import create_engine, inspect, text, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from app.config import DATABASE_PATH, DATABASE_URL
from app.database.models import (
    Base, Tyre, TreadPattern, TyreSize, Vehicle, Fitment, TreadFeatures,
    Investigation, EvidenceImage, EvidenceFeatures, MatchingRun, MatchResult
)

# Ensure parent directory exists
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """
    Ensure SQLite enforces foreign key constraints on all database connections.
    SQLite ignores foreign keys by default unless PRAGMA foreign_keys = ON is executed.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    except Exception:
        pass
    finally:
        cursor.close()


@event.listens_for(Engine, "checkout")
def set_sqlite_pragma_checkout(dbapi_connection, connection_record, connection_proxy):
    """
    Re-affirm PRAGMA foreign_keys=ON whenever a connection is checked out from the pool.
    Guarantees active foreign key enforcement across connection reuse and test fixtures.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    except Exception:
        pass
    finally:
        cursor.close()


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def migrate_schema(target_engine=None):
    """
    Apply non-destructive schema migrations to maintain backwards compatibility
    with existing databases. Ensures missing columns or tables are created without data loss.
    """
    b_engine = target_engine or engine
    with b_engine.connect() as conn:
        inspector = inspect(b_engine)
        existing_tables = set(inspector.get_table_names())

        # 1. Investigations: add is_archived and archived_at if missing
        if "investigations" in existing_tables:
            inv_cols = {c["name"] for c in inspector.get_columns("investigations")}
            if "is_archived" not in inv_cols:
                conn.execute(text("ALTER TABLE investigations ADD COLUMN is_archived BOOLEAN NOT NULL DEFAULT 0;"))
            if "archived_at" not in inv_cols:
                conn.execute(text("ALTER TABLE investigations ADD COLUMN archived_at DATETIME;"))

        # 2. Ensure evidence_images enforces ON DELETE RESTRICT on investigation_id
        if "evidence_images" in existing_tables:
            ev_sql_row = conn.execute(
                text("SELECT sql FROM sqlite_master WHERE type='table' AND name='evidence_images';")
            ).fetchone()
            norm_ev_sql = ev_sql_row[0].replace(" ", "").lower() if ev_sql_row and ev_sql_row[0] else ""
            if ev_sql_row and "ondeleterestrict" not in norm_ev_sql:
                conn.execute(text("PRAGMA foreign_keys = OFF;"))
                conn.execute(text("DROP VIEW IF EXISTS v_investigation_summaries;"))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS evidence_images_migrated (
                        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                        investigation_id INTEGER NOT NULL REFERENCES investigations(id) ON DELETE RESTRICT,
                        image_path VARCHAR(255) NOT NULL,
                        original_filename VARCHAR(255) NOT NULL,
                        image_format VARCHAR(20) NOT NULL,
                        image_width INTEGER NOT NULL,
                        image_height INTEGER NOT NULL,
                        upload_timestamp DATETIME NOT NULL,
                        image_checksum VARCHAR(64),
                        CONSTRAINT check_positive_dimensions CHECK (image_width > 0 AND image_height > 0)
                    );
                """))
                conn.execute(text("""
                    INSERT INTO evidence_images_migrated (id, investigation_id, image_path, original_filename, image_format, image_width, image_height, upload_timestamp, image_checksum)
                    SELECT id, investigation_id, image_path, original_filename, image_format, image_width, image_height, upload_timestamp, image_checksum
                    FROM evidence_images;
                """))
                conn.execute(text("DROP TABLE evidence_images;"))
                conn.execute(text("ALTER TABLE evidence_images_migrated RENAME TO evidence_images;"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_evidence_images_investigation_id ON evidence_images (investigation_id);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_evidence_images_image_checksum ON evidence_images (image_checksum);"))
                conn.execute(text("PRAGMA foreign_keys = ON;"))

        # 3. Ensure matching_runs table exists and has candidate key (id, evidence_id)
        if "matching_runs" not in existing_tables:
            Base.metadata.tables["matching_runs"].create(b_engine, checkfirst=True)
            existing_tables.add("matching_runs")
        else:
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_matching_runs_id_evidence_id ON matching_runs (id, evidence_id);"))

        # 4. Match Results: add run_id if missing and ensure composite FK to matching_runs(id, evidence_id)
        if "match_results" in existing_tables:
            match_cols = {c["name"] for c in inspector.get_columns("match_results")}
            if "run_id" not in match_cols:
                conn.execute(text("ALTER TABLE match_results ADD COLUMN run_id INTEGER REFERENCES matching_runs(id) ON DELETE CASCADE;"))

            # Backfill any match_results where run_id is null
            unassigned_ev_rows = conn.execute(
                text("SELECT DISTINCT evidence_id FROM match_results WHERE run_id IS NULL;")
            ).fetchall()
            for (ev_id,) in unassigned_ev_rows:
                existing_run = conn.execute(
                    text("SELECT id FROM matching_runs WHERE evidence_id = :ev_id ORDER BY run_number ASC LIMIT 1;"),
                    {"ev_id": ev_id}
                ).fetchone()
                if existing_run:
                    run_id = existing_run[0]
                else:
                    conn.execute(
                        text("""
                            INSERT INTO matching_runs (evidence_id, run_number, matching_method, weights_json, run_timestamp, notes)
                            VALUES (:ev_id, 1, 'Multi-Criteria Fusion v2', '{"edge": 0.3, "texture": 0.25, "shape": 0.25, "local": 0.2}', datetime('now'), 'Initial baseline matching run');
                        """),
                        {"ev_id": ev_id}
                    )
                    run_id = conn.execute(text("SELECT last_insert_rowid();")).fetchone()[0]

                conn.execute(
                    text("UPDATE match_results SET run_id = :run_id WHERE evidence_id = :ev_id AND run_id IS NULL;"),
                    {"run_id": run_id, "ev_id": ev_id}
                )

            # Modernize match_results constraints: ensure composite FK to matching_runs(id, evidence_id)
            tbl_sql_row = conn.execute(
                text("SELECT sql FROM sqlite_master WHERE type='table' AND name='match_results';")
            ).fetchone()
            norm_sql = tbl_sql_row[0].replace(" ", "").lower() if tbl_sql_row and tbl_sql_row[0] else ""
            needs_match_results_migration = (
                tbl_sql_row and (
                    "uq_evidence_pattern_match" in norm_sql or
                    "referencesmatching_runs(id,evidence_id)" not in norm_sql
                )
            )
            if needs_match_results_migration:
                conn.execute(text("PRAGMA foreign_keys = OFF;"))
                conn.execute(text("DROP VIEW IF EXISTS v_investigation_summaries;"))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS match_results_migrated (
                        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                        run_id INTEGER NOT NULL,
                        evidence_id INTEGER NOT NULL,
                        tread_pattern_id INTEGER NOT NULL REFERENCES tread_patterns(id) ON DELETE RESTRICT,
                        similarity_score FLOAT NOT NULL,
                        rank INTEGER NOT NULL,
                        matching_method VARCHAR(100) NOT NULL,
                        match_details TEXT,
                        match_timestamp DATETIME NOT NULL,
                        FOREIGN KEY (run_id, evidence_id) REFERENCES matching_runs(id, evidence_id) ON DELETE CASCADE,
                        FOREIGN KEY (evidence_id) REFERENCES evidence_images(id) ON DELETE CASCADE,
                        CONSTRAINT check_valid_similarity_score CHECK (similarity_score >= 0.0 AND similarity_score <= 100.0),
                        CONSTRAINT check_valid_rank CHECK (rank >= 1),
                        CONSTRAINT uq_run_pattern_match UNIQUE (run_id, tread_pattern_id),
                        CONSTRAINT uq_run_rank UNIQUE (run_id, rank)
                    );
                """))
                conn.execute(text("""
                    INSERT INTO match_results_migrated (id, run_id, evidence_id, tread_pattern_id, similarity_score, rank, matching_method, match_details, match_timestamp)
                    SELECT id, COALESCE(run_id, 1), evidence_id, tread_pattern_id, similarity_score, rank, matching_method, match_details, match_timestamp
                    FROM match_results;
                """))
                conn.execute(text("DROP TABLE match_results;"))
                conn.execute(text("ALTER TABLE match_results_migrated RENAME TO match_results;"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_match_results_evidence_id ON match_results (evidence_id);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_match_results_run_id ON match_results (run_id);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_match_results_run_rank ON match_results (run_id, rank);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_match_results_evidence_rank ON match_results (evidence_id, rank);"))
                conn.execute(text("PRAGMA foreign_keys = ON;"))

        conn.execute(text("PRAGMA foreign_keys = ON;"))
        conn.commit()


def create_triggers(target_engine=None):
    """Create database triggers enforcing forensic integrity rules in SQLite."""
    b_engine = target_engine or engine
    trigger_statements = [
        """
        CREATE TRIGGER IF NOT EXISTS trg_validate_match_result_evidence_id
        BEFORE INSERT ON match_results
        FOR EACH ROW
        WHEN NEW.evidence_id != (SELECT evidence_id FROM matching_runs WHERE id = NEW.run_id)
        BEGIN
            SELECT RAISE(ABORT, 'Integrity violation: match_results.evidence_id does not match matching_runs.evidence_id');
        END;
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_validate_match_result_evidence_id_update
        BEFORE UPDATE OF run_id, evidence_id ON match_results
        FOR EACH ROW
        WHEN NEW.evidence_id != (SELECT evidence_id FROM matching_runs WHERE id = NEW.run_id)
        BEGIN
            SELECT RAISE(ABORT, 'Integrity violation: match_results.evidence_id does not match matching_runs.evidence_id');
        END;
        """
    ]
    with b_engine.connect() as conn:
        for stmt in trigger_statements:
            conn.execute(text(stmt.strip()))
        conn.commit()


def create_views(target_engine=None):
    """Create analytical and forensic reporting views in the database."""
    b_engine = target_engine or engine
    view_sql = """
    DROP VIEW IF EXISTS v_investigation_summaries;
    CREATE VIEW v_investigation_summaries AS
    SELECT
        i.id AS investigation_id,
        i.case_number,
        i.title,
        i.status,
        i.is_archived,
        i.investigation_date,
        COUNT(DISTINCT e.id) AS evidence_count,
        COUNT(DISTINCT r.id) AS total_matching_runs,
        COUNT(DISTINCT m.id) AS total_matches_recorded,
        COALESCE(MAX(m.similarity_score), 0.0) AS highest_similarity_score,
        (
            SELECT tp.pattern_type
            FROM match_results m2
            JOIN evidence_images e2 ON m2.evidence_id = e2.id
            JOIN tread_patterns tp ON m2.tread_pattern_id = tp.id
            WHERE e2.investigation_id = i.id AND m2.rank = 1
            ORDER BY m2.similarity_score DESC
            LIMIT 1
        ) AS top_pattern_type,
        (
            SELECT t.manufacturer || ' ' || t.model
            FROM match_results m3
            JOIN evidence_images e3 ON m3.evidence_id = e3.id
            JOIN tread_patterns tp2 ON m3.tread_pattern_id = tp2.id
            JOIN tyres t ON tp2.tyre_id = t.id
            WHERE e3.investigation_id = i.id AND m3.rank = 1
            ORDER BY m3.similarity_score DESC
            LIMIT 1
        ) AS top_candidate_tyre
    FROM investigations i
    LEFT JOIN evidence_images e ON i.id = e.investigation_id
    LEFT JOIN matching_runs r ON e.id = r.evidence_id
    LEFT JOIN match_results m ON e.id = m.evidence_id
    GROUP BY i.id, i.case_number, i.title, i.status, i.is_archived, i.investigation_date;
    """
    with b_engine.connect() as conn:
        for stmt in view_sql.strip().split(";"):
            if stmt.strip():
                conn.execute(text(stmt.strip() + ";"))
        conn.commit()


def init_db(target_engine=None):
    """Create all tables, analytical views, and forensic integrity triggers in the database if they do not exist."""
    b_engine = target_engine or engine
    Base.metadata.create_all(bind=b_engine)
    migrate_schema(b_engine)
    create_views(b_engine)
    create_triggers(b_engine)


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
        "tread_features": ["id", "tyre_id", "edge_features", "texture_features", "shape_features", "feature_vector"],
        "investigations": ["id", "case_number", "title", "status", "investigation_date", "is_archived"],
        "evidence_images": ["id", "investigation_id", "image_path", "original_filename"],
        "evidence_features": ["id", "evidence_id", "edge_features", "texture_features", "feature_vector"],
        "matching_runs": ["id", "evidence_id", "run_number", "weights_json", "run_timestamp"],
        "match_results": ["id", "run_id", "evidence_id", "tread_pattern_id", "similarity_score", "rank"]
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
            counts["investigations"] = sess.query(Investigation).count()
            counts["evidence_images"] = sess.query(EvidenceImage).count()
            counts["evidence_features"] = sess.query(EvidenceFeatures).count()
            counts["match_results"] = sess.query(MatchResult).count()
            counts["matching_runs"] = sess.query(MatchingRun).count()

            # Verify relationship references
            tyre_ids = set(t.id for t in sess.query(Tyre.id).all())
            vehicle_ids = set(v.id for v in sess.query(Vehicle.id).all())
            investigation_ids = set(i.id for i in sess.query(Investigation.id).all())
            evidence_ids = set(e.id for e in sess.query(EvidenceImage.id).all())
            run_ids = set(r.id for r in sess.query(MatchingRun.id).all())
            pattern_ids = set(p.id for p in sess.query(TreadPattern.id).all())

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

            # Check evidence references
            for e in sess.query(EvidenceImage).all():
                if e.investigation_id not in investigation_ids:
                    relationship_errors.append(f"EvidenceImage id={e.id} references non-existent investigation_id={e.investigation_id}")

            # Check evidence features references
            for ef in sess.query(EvidenceFeatures).all():
                if ef.evidence_id not in evidence_ids:
                    relationship_errors.append(f"EvidenceFeatures id={ef.id} references non-existent evidence_id={ef.evidence_id}")

            # Check matching_runs references
            for r in sess.query(MatchingRun).all():
                if r.evidence_id not in evidence_ids:
                    relationship_errors.append(f"MatchingRun id={r.id} references non-existent evidence_id={r.evidence_id}")

            # Check match results references
            for m in sess.query(MatchResult).all():
                if m.run_id not in run_ids:
                    relationship_errors.append(f"MatchResult id={m.id} references non-existent run_id={m.run_id}")
                if m.evidence_id not in evidence_ids:
                    relationship_errors.append(f"MatchResult id={m.id} references non-existent evidence_id={m.evidence_id}")
                if m.tread_pattern_id not in pattern_ids:
                    relationship_errors.append(f"MatchResult id={m.id} references non-existent tread_pattern_id={m.tread_pattern_id}")
                if m.matching_run and m.evidence_id != m.matching_run.evidence_id:
                    relationship_errors.append(f"MatchResult id={m.id} evidence_id={m.evidence_id} desynchronized with MatchingRun evidence_id={m.matching_run.evidence_id}")
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
