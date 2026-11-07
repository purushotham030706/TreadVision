"""
Unit and Integration Tests for TreadVision Forensic DBMS Architecture
Validates:
1. Investigation CRUD & Unique Case Number Constraints
2. Evidence Association & Cascade Deletion
3. 31-D Feature Persistence and Deserialization
4. Match Result Persistence, Ranking, and Bounds
5. Foreign Key Constraint Enforcement (PRAGMA foreign_keys = ON)
6. Check Constraints (Status, Similarity Score [0-100], Rank >= 1, Dimensions > 0)
7. Relational SQL Joins (Evidence -> Pattern -> Tyre -> Fitment -> Vehicle)
8. Aggregate Queries (Manufacturer Candidate Frequency)
9. SQL View (v_investigation_summaries)
10. Atomic Transaction Handling and Rollback on Failure
11. Persistence Across Session Reinitialization
12. Zero Regression of Reference Catalogue
"""
from datetime import date, datetime
from pathlib import Path
import json
import pytest
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.config import FORENSIC_IMPRESSIONS_DIR, BASE_DIR
from app.database.connection import init_db, verify_database_integrity, get_db_session
from app.database.models import (
    Base, Tyre, TreadPattern, Vehicle, Fitment, TyreSize, TreadFeatures,
    Investigation, EvidenceImage, EvidenceFeatures, MatchingRun, MatchResult
)
from app.database.forensic_repository import (
    get_or_create_investigation,
    list_investigations,
    archive_investigation,
    save_forensic_match_transaction,
    query_investigation_summaries_view,
    get_evidence_details_with_matches,
    get_manufacturer_match_aggregates,
    store_evidence_file,
    verify_evidence_storage_integrity,
    cleanup_orphan_evidence_files
)
from app.image_processing.forensic_matcher import ForensicMatcher, ForensicMatchCandidate
from app.image_processing.pipeline import PipelineResult


def test_investigation_creation_and_retrieval(isolated_db_session):
    """Verify investigation creation, field assignment, and retrieval by case_number."""
    inv, created = get_or_create_investigation(
        session=isolated_db_session,
        case_number="CASE-TEST-001",
        title="Test Investigation Alpha",
        investigation_date=date(2026, 11, 4),
        description="Detailed test case for unit validation.",
        location="Sector 12 Test Facility",
        status="Active"
    )
    isolated_db_session.commit()
    assert created is True
    assert inv.id is not None
    assert inv.case_number == "CASE-TEST-001"

    # Query back
    queried = isolated_db_session.query(Investigation).filter(
        Investigation.case_number == "CASE-TEST-001"
    ).first()
    assert queried is not None
    assert queried.title == "Test Investigation Alpha"
    assert queried.location == "Sector 12 Test Facility"
    assert str(queried.investigation_date) == "2026-11-04"
    assert queried.status == "Active"


def test_investigation_unique_case_number_constraint(isolated_db_session):
    """Verify that duplicate case numbers are strictly rejected by UNIQUE constraint."""
    inv1 = Investigation(
        case_number="CASE-DUP-999",
        title="First Case",
        investigation_date=date.today(),
        status="Active"
    )
    isolated_db_session.add(inv1)
    isolated_db_session.commit()

    inv2 = Investigation(
        case_number="CASE-DUP-999",
        title="Second Duplicate Case",
        investigation_date=date.today(),
        status="Active"
    )
    isolated_db_session.add(inv2)
    with pytest.raises(IntegrityError):
        isolated_db_session.commit()
    isolated_db_session.rollback()


def test_investigation_status_check_constraint(isolated_db_session):
    """Verify that invalid investigation status values violate CHECK constraint."""
    inv = Investigation(
        case_number="CASE-STATUS-ERR",
        title="Invalid Status Case",
        investigation_date=date.today(),
        status="NonExistentStatus"  # Violates ('Active', 'Under Review', 'Closed', 'Archived')
    )
    isolated_db_session.add(inv)
    with pytest.raises(IntegrityError):
        isolated_db_session.commit()
    isolated_db_session.rollback()


def test_investigation_deletion_blocked_when_containing_evidence_and_cascade_from_evidence(isolated_db_session):
    """Verify that an investigation containing evidence cannot be deleted (ON DELETE RESTRICT), and deleting evidence cascades to runs/matches."""
    inv = Investigation(
        case_number="CASE-CASCADE-01",
        title="Cascade Test Case",
        investigation_date=date.today(),
        status="Active"
    )
    isolated_db_session.add(inv)
    isolated_db_session.flush()

    pattern = isolated_db_session.query(TreadPattern).first()
    assert pattern is not None

    evidence = EvidenceImage(
        investigation_id=inv.id,
        image_path="data/raw/forensic/test_evidence.jpg",
        original_filename="test_evidence.jpg",
        image_format="JPEG",
        image_width=512,
        image_height=512,
        image_checksum="abcd1234abcd1234"
    )
    isolated_db_session.add(evidence)
    isolated_db_session.flush()

    features = EvidenceFeatures(
        evidence_id=evidence.id,
        edge_features=json.dumps({"density": 0.25}),
        texture_features=json.dumps({"contrast": 1.5}),
        shape_features=json.dumps({"solidity": 0.85}),
        feature_vector=json.dumps([0.1] * 31),
        extraction_method="Test Pipeline"
    )
    isolated_db_session.add(features)

    run = MatchingRun(
        evidence_id=evidence.id,
        run_number=1,
        matching_method="Multi-Criteria Test",
        weights_json=json.dumps({"edge": 0.3, "texture": 0.25, "shape": 0.25, "local": 0.2}),
        run_timestamp=datetime.now()
    )
    isolated_db_session.add(run)
    isolated_db_session.flush()

    match = MatchResult(
        run_id=run.id,
        evidence_id=evidence.id,
        tread_pattern_id=pattern.id,
        similarity_score=92.5,
        rank=1,
        matching_method="Multi-Criteria Test"
    )
    isolated_db_session.add(match)
    isolated_db_session.commit()

    ev_id = evidence.id
    # Assert records exist
    assert isolated_db_session.query(EvidenceImage).filter(EvidenceImage.id == ev_id).count() == 1
    assert isolated_db_session.query(EvidenceFeatures).filter(EvidenceFeatures.evidence_id == ev_id).count() == 1
    assert isolated_db_session.query(MatchingRun).filter(MatchingRun.evidence_id == ev_id).count() == 1
    assert isolated_db_session.query(MatchResult).filter(MatchResult.evidence_id == ev_id).count() == 1

    # 1. Direct deletion of investigation containing evidence must be BLOCKED by ON DELETE RESTRICT
    isolated_db_session.delete(inv)
    with pytest.raises(IntegrityError):
        isolated_db_session.commit()
    isolated_db_session.rollback()

    # Re-fetch after rollback
    evidence_to_delete = isolated_db_session.query(EvidenceImage).filter(EvidenceImage.id == ev_id).first()
    assert evidence_to_delete is not None

    # 2. Deleting the evidence image cascades down to features, runs, and matches
    isolated_db_session.delete(evidence_to_delete)
    isolated_db_session.commit()

    assert isolated_db_session.query(EvidenceImage).filter(EvidenceImage.id == ev_id).count() == 0
    assert isolated_db_session.query(EvidenceFeatures).filter(EvidenceFeatures.evidence_id == ev_id).count() == 0
    assert isolated_db_session.query(MatchingRun).filter(MatchingRun.evidence_id == ev_id).count() == 0
    assert isolated_db_session.query(MatchResult).filter(MatchResult.evidence_id == ev_id).count() == 0

    # 3. Once empty of evidence, the investigation can be safely removed
    inv_now_empty = isolated_db_session.query(Investigation).filter(Investigation.id == inv.id).first()
    isolated_db_session.delete(inv_now_empty)
    isolated_db_session.commit()
    assert isolated_db_session.query(Investigation).filter(Investigation.id == inv.id).count() == 0


def test_feature_persistence_31d_and_json(isolated_db_session):
    """Verify exact 31-D feature vector storage, float fidelity, and JSON descriptors."""
    inv = Investigation(case_number="CASE-FEAT-31D", title="Feature Test", status="Active")
    isolated_db_session.add(inv)
    isolated_db_session.flush()

    evidence = EvidenceImage(
        investigation_id=inv.id,
        image_path="test_path.jpg",
        original_filename="test.jpg",
        image_format="JPEG",
        image_width=512,
        image_height=512
    )
    isolated_db_session.add(evidence)
    isolated_db_session.flush()

    vector_31d = [round(float(i * 0.03125), 5) for i in range(31)]
    edge_dict = {"total_edge_density": 0.185, "horizontal_edge_density": 0.092}
    tex_dict = {"contrast": 2.45, "homogeneity": 0.81}
    shape_dict = {"solidity": 0.74, "hu_moment_1": 0.0023}

    feat = EvidenceFeatures(
        evidence_id=evidence.id,
        edge_features=json.dumps(edge_dict),
        texture_features=json.dumps(tex_dict),
        shape_features=json.dumps(shape_dict),
        feature_vector=json.dumps(vector_31d),
        extraction_method="TreadVision 31D Adaptive Pipeline"
    )
    isolated_db_session.add(feat)
    isolated_db_session.commit()

    retrieved = isolated_db_session.query(EvidenceFeatures).filter(
        EvidenceFeatures.evidence_id == evidence.id
    ).first()
    assert retrieved is not None
    vec = retrieved.get_feature_vector()
    assert len(vec) == 31
    assert vec == vector_31d
    d = retrieved.to_dict()
    assert d["edge_features"]["total_edge_density"] == 0.185
    assert d["texture_features"]["contrast"] == 2.45


def test_match_result_persistence_and_ranking(isolated_db_session):
    """Verify ordered storage and retrieval of multi-candidate match rankings."""
    inv = Investigation(case_number="CASE-RANK-TEST", title="Rank Test", status="Active")
    isolated_db_session.add(inv)
    isolated_db_session.flush()

    evidence = EvidenceImage(
        investigation_id=inv.id,
        image_path="test.jpg",
        original_filename="test.jpg",
        image_format="JPEG",
        image_width=512,
        image_height=512
    )
    isolated_db_session.add(evidence)
    isolated_db_session.flush()

    patterns = isolated_db_session.query(TreadPattern).limit(3).all()
    assert len(patterns) >= 3

    run = MatchingRun(
        evidence_id=evidence.id,
        run_number=1,
        matching_method="Multi-Criteria Fusion v2",
        weights_json=json.dumps({"edge": 0.3, "texture": 0.25, "shape": 0.25, "local": 0.2}),
        run_timestamp=datetime.now()
    )
    isolated_db_session.add(run)
    isolated_db_session.flush()

    scores = [94.2, 88.7, 81.0]
    for idx, (pat, score) in enumerate(zip(patterns, scores), start=1):
        m = MatchResult(
            run_id=run.id,
            evidence_id=evidence.id,
            tread_pattern_id=pat.id,
            similarity_score=score,
            rank=idx,
            matching_method="Multi-Criteria Fusion v2",
            match_details=json.dumps({"edge_similarity": score})
        )
        isolated_db_session.add(m)
    isolated_db_session.commit()

    matches = isolated_db_session.query(MatchResult).filter(
        MatchResult.evidence_id == evidence.id
    ).order_by(MatchResult.rank.asc()).all()

    assert len(matches) == 3
    assert matches[0].rank == 1
    assert matches[0].similarity_score == 94.2
    assert matches[1].rank == 2
    assert matches[1].similarity_score == 88.7
    assert matches[2].rank == 3
    assert matches[2].similarity_score == 81.0


def test_foreign_key_enforcement(isolated_engine, isolated_db_session):
    """Verify that foreign keys reject orphan evidence and orphan match rows under PRAGMA foreign_keys = ON."""
    # Orphan evidence referencing non-existent investigation_id = 99999
    orphan_evidence = EvidenceImage(
        investigation_id=99999,
        image_path="orphan.jpg",
        original_filename="orphan.jpg",
        image_format="JPEG",
        image_width=512,
        image_height=512
    )
    isolated_db_session.add(orphan_evidence)
    with pytest.raises(IntegrityError):
        isolated_db_session.commit()
    isolated_db_session.rollback()

    # Orphan match result referencing non-existent tread_pattern_id = 99999
    inv = Investigation(case_number="CASE-FK-TEST", title="FK Test", status="Active")
    isolated_db_session.add(inv)
    isolated_db_session.flush()

    evidence = EvidenceImage(
        investigation_id=inv.id,
        image_path="test.jpg",
        original_filename="test.jpg",
        image_format="JPEG",
        image_width=512,
        image_height=512
    )
    isolated_db_session.add(evidence)
    isolated_db_session.flush()

    run = MatchingRun(
        evidence_id=evidence.id,
        run_number=1,
        matching_method="FK Test Run",
        weights_json=json.dumps({"edge": 0.3, "texture": 0.25, "shape": 0.25, "local": 0.2}),
        run_timestamp=datetime.now()
    )
    isolated_db_session.add(run)
    isolated_db_session.flush()

    orphan_match = MatchResult(
        run_id=run.id,
        evidence_id=evidence.id,
        tread_pattern_id=99999,  # Non-existent FK violation
        similarity_score=85.0,
        rank=1
    )
    isolated_db_session.add(orphan_match)
    with pytest.raises(IntegrityError):
        isolated_db_session.commit()
    isolated_db_session.rollback()

    # Orphan match result referencing non-existent run_id = 99999
    valid_pattern = isolated_db_session.query(TreadPattern).first()
    orphan_run_match = MatchResult(
        run_id=99999,  # Non-existent run FK violation
        evidence_id=evidence.id,
        tread_pattern_id=valid_pattern.id,
        similarity_score=85.0,
        rank=1
    )
    isolated_db_session.add(orphan_run_match)
    with pytest.raises(IntegrityError):
        isolated_db_session.commit()
    isolated_db_session.rollback()


def test_check_constraints_similarity_and_rank(isolated_db_session):
    """Verify CHECK constraints on similarity score bounds [0.0, 100.0] and rank >= 1."""
    inv = Investigation(case_number="CASE-CHECK-BOUNDS", title="Bounds Test", status="Active")
    isolated_db_session.add(inv)
    isolated_db_session.flush()

    evidence = EvidenceImage(
        investigation_id=inv.id,
        image_path="test.jpg",
        original_filename="test.jpg",
        image_format="JPEG",
        image_width=512,
        image_height=512
    )
    isolated_db_session.add(evidence)
    isolated_db_session.flush()

    run = MatchingRun(
        evidence_id=evidence.id,
        run_number=1,
        matching_method="Bounds Test Run",
        weights_json=json.dumps({"edge": 0.3, "texture": 0.25, "shape": 0.25, "local": 0.2}),
        run_timestamp=datetime.now()
    )
    isolated_db_session.add(run)
    isolated_db_session.flush()

    pat = isolated_db_session.query(TreadPattern).first()

    # 1. Similarity score > 100.0
    m_high = MatchResult(run_id=run.id, evidence_id=evidence.id, tread_pattern_id=pat.id, similarity_score=105.0, rank=1)
    isolated_db_session.add(m_high)
    with pytest.raises(IntegrityError):
        isolated_db_session.commit()
    isolated_db_session.rollback()

    # 2. Similarity score < 0.0
    m_neg = MatchResult(run_id=run.id, evidence_id=evidence.id, tread_pattern_id=pat.id, similarity_score=-5.0, rank=1)
    isolated_db_session.add(m_neg)
    with pytest.raises(IntegrityError):
        isolated_db_session.commit()
    isolated_db_session.rollback()

    # 3. Rank < 1
    m_rank_zero = MatchResult(run_id=run.id, evidence_id=evidence.id, tread_pattern_id=pat.id, similarity_score=80.0, rank=0)
    isolated_db_session.add(m_rank_zero)
    with pytest.raises(IntegrityError):
        isolated_db_session.commit()
    isolated_db_session.rollback()


def test_sql_joins_and_multi_table_traceability(isolated_db_session):
    """
    Verify complex SQL multi-table join:
    evidence_images -> investigations -> evidence_features -> match_results ->
    tread_patterns -> tyres -> fitments -> vehicles & tyre_sizes.
    """
    inv = isolated_db_session.query(Investigation).filter(Investigation.case_number == "CASE-2026-001").first()
    assert inv is not None
    assert len(inv.evidence_images) > 0
    ev = inv.evidence_images[0]

    details = get_evidence_details_with_matches(isolated_db_session, ev.id)
    assert details is not None
    assert details["case_number"] == "CASE-2026-001"
    assert details["evidence_id"] == ev.id
    assert details["image_checksum"] is not None
    assert details["features"] is not None
    assert len(details["features"]["feature_vector"]) == 31
    assert len(details["matches"]) == 5

    # Check top match details joined from tyres, patterns, fitments, vehicles
    top_cand = details["matches"][0]
    assert top_cand["rank"] == 1
    assert top_cand["similarity_score"] > 80.0
    assert "tyre" in top_cand
    assert top_cand["tyre"]["manufacturer"] is not None
    assert top_cand["tyre"]["model"] is not None
    assert len(top_cand["tyre"]["available_sizes"]) > 0
    assert len(top_cand["tyre"]["compatible_vehicles"]) > 0


def test_aggregate_query_manufacturer_attribution(isolated_db_session):
    """Verify SQL aggregate query grouping match results by manufacturer."""
    mfg_stats = get_manufacturer_match_aggregates(isolated_db_session)
    assert len(mfg_stats) > 0
    for stat in mfg_stats:
        assert "manufacturer" in stat
        assert stat["appearance_count"] >= 1
        assert 0.0 <= stat["avg_similarity"] <= 100.0
        assert 0.0 <= stat["max_similarity"] <= 100.0


def test_sql_view_investigation_summaries(isolated_db_session):
    """Verify that SQL view v_investigation_summaries compiles and returns accurate aggregates."""
    summaries = query_investigation_summaries_view(isolated_db_session)
    assert len(summaries) >= 1
    s = summaries[0]
    assert s["case_number"] == "CASE-2026-001"
    assert s["evidence_count"] >= 1
    assert s["total_matches_recorded"] >= 5
    assert s["highest_similarity_score"] > 80.0
    assert s["top_candidate_tyre"] != "N/A"


def test_transaction_atomic_rollback_on_failure(isolated_db_session, tmp_path):
    """Verify that a failure during forensic match saving completely rolls back the transaction."""
    inv_count_before = isolated_db_session.query(Investigation).count()
    ev_count_before = isolated_db_session.query(EvidenceImage).count()
    feat_count_before = isolated_db_session.query(EvidenceFeatures).count()
    match_count_before = isolated_db_session.query(MatchResult).count()

    # Create dummy pipeline result
    dummy_pipe = PipelineResult(
        success=True,
        stages={"Enhanced (CLAHE)": np.zeros((512, 512), dtype=np.uint8)},
        features={"edge": {}, "texture": {}, "shape": {}},
        feature_vector=np.zeros((31,), dtype=np.float32)
    )

    # Corrupted candidate with invalid similarity score > 100 to trigger CHECK constraint failure
    corrupt_candidate = ForensicMatchCandidate(
        rank=1,
        tyre_id=1,
        manufacturer="CorruptMfg",
        model="CorruptModel",
        pattern_name="CorruptPattern",
        pattern_type="Asymmetric",
        symmetry="Asymmetric",
        similarity_score=150.0,  # VIOLATES CHECK CONSTRAINT
        confidence_level="High",
        component_scores={},
        reference_image_path="none.jpg",
        available_sizes=[],
        compatible_vehicles=[],
        description="Corrupt"
    )

    with pytest.raises(IntegrityError):
        save_forensic_match_transaction(
            case_number="CASE-FAIL-TEST",
            image_input=np.zeros((512, 512, 3), dtype=np.uint8),
            original_filename="fail_test.jpg",
            pipeline_result=dummy_pipe,
            candidates=[corrupt_candidate],
            session_instance=isolated_db_session
        )

    # Verify atomic rollback — no phantom records created
    assert isolated_db_session.query(Investigation).filter(Investigation.case_number == "CASE-FAIL-TEST").count() == 0
    assert isolated_db_session.query(EvidenceImage).count() == ev_count_before
    assert isolated_db_session.query(EvidenceFeatures).count() == feat_count_before
    assert isolated_db_session.query(MatchResult).count() == match_count_before


def test_persistence_after_connection_reinitialization(tmp_path):
    """Verify that committed forensic records persist cleanly after closing and reopening database."""
    test_db_path = tmp_path / "persist_verify.db"
    eng1 = create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False})
    init_db(eng1)

    SessionMaker1 = sessionmaker(bind=eng1)
    sess1 = SessionMaker1()

    inv = Investigation(
        case_number="CASE-PERSIST-100",
        title="Session Persistence Test",
        investigation_date=date(2026, 11, 4),
        status="Active"
    )
    sess1.add(inv)
    sess1.flush()

    evidence = EvidenceImage(
        investigation_id=inv.id,
        image_path="data/raw/forensic/evidence_uploads/persist_sample.jpg",
        original_filename="persist_sample.jpg",
        image_format="JPEG",
        image_width=512,
        image_height=512,
        image_checksum="sha256_mock_hash_for_persistence_test"
    )
    sess1.add(evidence)
    sess1.commit()
    sess1.close()
    eng1.dispose()

    # Reconnect with fresh engine and session
    eng2 = create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False})
    SessionMaker2 = sessionmaker(bind=eng2)
    sess2 = SessionMaker2()

    reloaded_inv = sess2.query(Investigation).filter(Investigation.case_number == "CASE-PERSIST-100").first()
    assert reloaded_inv is not None
    assert reloaded_inv.title == "Session Persistence Test"
    assert len(reloaded_inv.evidence_images) == 1
    assert reloaded_inv.evidence_images[0].image_checksum == "sha256_mock_hash_for_persistence_test"

    sess2.close()
    eng2.dispose()


def test_reference_catalogue_regression_and_integrity(isolated_engine, isolated_db_session):
    """Verify that all 10 tables exist and pass foreign key checks without regressions."""
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
    assert integrity["counts"]["fitments"] >= 10
    assert integrity["counts"]["investigations"] >= 1
    assert integrity["counts"]["evidence_images"] >= 1
    assert integrity["counts"]["match_results"] >= 5


def test_matching_run_entity_and_multiple_runs_on_same_evidence(isolated_db_session):
    """Verify that multiple matching runs with different weights can be executed and stored on the same evidence."""
    inv = Investigation(case_number="CASE-MULTI-RUN", title="Multi Run Test", status="Active")
    isolated_db_session.add(inv)
    isolated_db_session.flush()

    evidence = EvidenceImage(
        investigation_id=inv.id,
        image_path="test_multi.jpg",
        original_filename="test_multi.jpg",
        image_format="JPEG",
        image_width=512,
        image_height=512,
        image_checksum="multi_checksum_123"
    )
    isolated_db_session.add(evidence)
    isolated_db_session.flush()

    patterns = isolated_db_session.query(TreadPattern).limit(3).all()
    assert len(patterns) >= 3

    # Run 1: Standard Weights
    weights_1 = {"edge": 0.35, "texture": 0.25, "shape": 0.25, "local": 0.15}
    run1 = MatchingRun(
        evidence_id=evidence.id,
        run_number=1,
        matching_method="Multi-Criteria Fusion v2",
        weights_json=json.dumps(weights_1),
        execution_time_ms=45.2,
        notes="Standard Weights Run",
        run_timestamp=datetime.now()
    )
    isolated_db_session.add(run1)
    isolated_db_session.flush()

    for idx, pat in enumerate(patterns, start=1):
        isolated_db_session.add(MatchResult(
            run_id=run1.id,
            evidence_id=evidence.id,
            tread_pattern_id=pat.id,
            similarity_score=90.0 - idx,
            rank=idx,
            matching_method="Multi-Criteria Fusion v2"
        ))
    isolated_db_session.commit()

    # Run 2: Edge-Dominant Weights on SAME evidence image
    weights_2 = {"edge": 0.60, "texture": 0.15, "shape": 0.15, "local": 0.10}
    run2 = MatchingRun(
        evidence_id=evidence.id,
        run_number=2,
        matching_method="Multi-Criteria Fusion v2",
        weights_json=json.dumps(weights_2),
        execution_time_ms=42.1,
        notes="Edge-Dominant Run",
        run_timestamp=datetime.now()
    )
    isolated_db_session.add(run2)
    isolated_db_session.flush()

    for idx, pat in enumerate(reversed(patterns), start=1):
        isolated_db_session.add(MatchResult(
            run_id=run2.id,
            evidence_id=evidence.id,
            tread_pattern_id=pat.id,
            similarity_score=95.0 - (idx * 2),
            rank=idx,
            matching_method="Multi-Criteria Fusion v2"
        ))
    isolated_db_session.commit()

    # Verify both runs are queryable for this evidence
    runs = isolated_db_session.query(MatchingRun).filter(
        MatchingRun.evidence_id == evidence.id
    ).order_by(MatchingRun.run_number.asc()).all()
    assert len(runs) == 2
    assert runs[0].run_number == 1
    assert runs[1].run_number == 2
    assert len(runs[0].matches) == 3
    assert len(runs[1].matches) == 3

    # Verify UniqueConstraint on (evidence_id, run_number)
    dup_run = MatchingRun(
        evidence_id=evidence.id,
        run_number=1,  # Duplicate run_number
        matching_method="Duplicate Test",
        run_timestamp=datetime.now()
    )
    isolated_db_session.add(dup_run)
    with pytest.raises(IntegrityError):
        isolated_db_session.commit()
    isolated_db_session.rollback()


def test_reference_catalogue_on_delete_restrict(isolated_db_session):
    """Verify that ON DELETE RESTRICT prevents deleting reference patterns cited in forensic match history."""
    # Find a pattern cited in seed matches
    match = isolated_db_session.query(MatchResult).first()
    assert match is not None
    cited_pattern_id = match.tread_pattern_id
    pattern = isolated_db_session.query(TreadPattern).filter(TreadPattern.id == cited_pattern_id).first()
    assert pattern is not None

    # Attempt to delete the cited reference tread pattern
    isolated_db_session.delete(pattern)
    with pytest.raises(IntegrityError):
        isolated_db_session.commit()
    isolated_db_session.rollback()

    # Pattern must still exist intact
    rechecked = isolated_db_session.query(TreadPattern).filter(TreadPattern.id == cited_pattern_id).first()
    assert rechecked is not None


def test_investigation_soft_archival(isolated_db_session):
    """Verify soft archival of investigations preserving audit trail and data accessibility."""
    inv = Investigation(
        case_number="CASE-ARCHIVE-TEST",
        title="Soft Archival Verification",
        investigation_date=date.today(),
        status="Active",
        is_archived=False
    )
    isolated_db_session.add(inv)
    isolated_db_session.commit()

    assert inv.is_archived is False
    assert inv.archived_at is None

    # Archive the investigation
    archive_investigation(isolated_db_session, inv.id, archive=True)
    assert inv.is_archived is True
    assert inv.archived_at is not None

    # Verify filtering
    active_cases = list_investigations(isolated_db_session, include_archived=False)
    assert not any(i.case_number == "CASE-ARCHIVE-TEST" for i in active_cases)

    all_cases = list_investigations(isolated_db_session, include_archived=True)
    assert any(i.case_number == "CASE-ARCHIVE-TEST" for i in all_cases)

    # Unarchive
    archive_investigation(isolated_db_session, inv.id, archive=False)
    assert inv.is_archived is False
    assert inv.archived_at is None
    active_cases_after = list_investigations(isolated_db_session, include_archived=False)
    assert any(i.case_number == "CASE-ARCHIVE-TEST" for i in active_cases_after)


def test_live_sha256_cryptographic_verification(isolated_db_session, tmp_path):
    """Verify cryptographic SHA-256 integrity verification: clean, tampered, and missing file states."""
    import hashlib
    original_bytes = b"EXACT_EVIDENCE_IMAGE_BINARY_DATA_FOR_SHA256"
    expected_hash = hashlib.sha256(original_bytes).hexdigest()

    inv = Investigation(case_number="CASE-CRYPTO-TEST", title="Crypto Audit", status="Active")
    isolated_db_session.add(inv)
    isolated_db_session.flush()

    # Create file relative to BASE_DIR
    evidence_rel = f"data/raw/forensic/evidence_uploads/test_crypto_{inv.id}.jpg"
    dest_path = BASE_DIR / evidence_rel
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(original_bytes)

    evidence = EvidenceImage(
        investigation_id=inv.id,
        image_path=evidence_rel,
        original_filename="crypto_test.jpg",
        image_format="JPEG",
        image_width=512,
        image_height=512,
        image_checksum=expected_hash
    )
    isolated_db_session.add(evidence)
    isolated_db_session.commit()

    try:
        # 1. Clean verification
        details_clean = get_evidence_details_with_matches(isolated_db_session, evidence.id)
        assert details_clean["checksum_status"] == "VERIFIED"
        assert details_clean["checksum_verified"] is True
        assert details_clean["live_checksum"] == expected_hash

        # 2. Tampered file verification (alter byte content)
        dest_path.write_bytes(b"MODIFIED_EVIDENCE_DATA_TAMPER_SIMULATION")
        details_tampered = get_evidence_details_with_matches(isolated_db_session, evidence.id)
        assert details_tampered["checksum_status"] == "TAMPER_DETECTED"
        assert details_tampered["checksum_verified"] is False
        assert details_tampered["live_checksum"] != expected_hash

        # 3. Missing file verification (delete file)
        dest_path.unlink()
        details_missing = get_evidence_details_with_matches(isolated_db_session, evidence.id)
        assert details_missing["checksum_status"] == "FILE_MISSING"
        assert details_missing["checksum_verified"] is False
    finally:
        if dest_path.exists():
            dest_path.unlink()


def test_storage_integrity_audit_and_orphan_cleanup(isolated_db_session):
    """Verify that storage integrity audit detects untracked orphan files and cleanup purges them."""
    from app.config import EVIDENCE_UPLOADS_DIR
    EVIDENCE_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    orphan_file = EVIDENCE_UPLOADS_DIR / "untracked_orphan_sample_file.jpg"
    orphan_file.write_bytes(b"ORPHAN_IMAGE_NOT_IN_SQLITE")

    try:
        audit_before = verify_evidence_storage_integrity(isolated_db_session)
        assert any(orphan_file.name in orphan for orphan in audit_before["orphan_files"])

        removed_count = cleanup_orphan_evidence_files(isolated_db_session)
        assert removed_count >= 1
        assert not orphan_file.exists()

        audit_after = verify_evidence_storage_integrity(isolated_db_session)
        assert not any(orphan_file.name in orphan for orphan in audit_after["orphan_files"])
    finally:
        if orphan_file.exists():
            orphan_file.unlink()


def test_match_results_evidence_id_enforcement_against_matching_run(isolated_db_session):
    """Verify that match_results.evidence_id must strictly match matching_runs.evidence_id (composite FK and trigger)."""
    inv = Investigation(case_number="CASE-SYNC-TEST", title="Sync Test", status="Active")
    isolated_db_session.add(inv)
    isolated_db_session.flush()

    # Create two distinct evidence records
    ev1 = EvidenceImage(
        investigation_id=inv.id,
        image_path="test_ev1.jpg",
        original_filename="test_ev1.jpg",
        image_format="JPEG",
        image_width=512,
        image_height=512
    )
    ev2 = EvidenceImage(
        investigation_id=inv.id,
        image_path="test_ev2.jpg",
        original_filename="test_ev2.jpg",
        image_format="JPEG",
        image_width=512,
        image_height=512
    )
    isolated_db_session.add_all([ev1, ev2])
    isolated_db_session.flush()

    # Create a matching run tied to ev1
    run1 = MatchingRun(
        evidence_id=ev1.id,
        run_number=1,
        matching_method="Multi-Criteria Fusion v2",
        weights_json=json.dumps({"edge": 0.35, "texture": 0.25, "shape": 0.25, "local": 0.15}),
        run_timestamp=datetime.now()
    )
    isolated_db_session.add(run1)
    isolated_db_session.commit()

    pat = isolated_db_session.query(TreadPattern).first()
    assert pat is not None

    # 1. Attempt to insert MatchResult with run1 (ev1) but specifying evidence_id = ev2.id -> MUST FAIL
    invalid_match = MatchResult(
        run_id=run1.id,
        evidence_id=ev2.id,  # MISMATCH: run1 belongs to ev1, not ev2!
        tread_pattern_id=pat.id,
        similarity_score=91.0,
        rank=1,
        matching_method="Multi-Criteria Fusion v2"
    )
    isolated_db_session.add(invalid_match)
    with pytest.raises(IntegrityError):
        isolated_db_session.commit()
    isolated_db_session.rollback()

    # 2. Insert valid MatchResult with matching evidence_id = ev1.id -> MUST SUCCEED
    valid_match = MatchResult(
        run_id=run1.id,
        evidence_id=ev1.id,  # MATCHES run1.evidence_id
        tread_pattern_id=pat.id,
        similarity_score=91.0,
        rank=1,
        matching_method="Multi-Criteria Fusion v2"
    )
    isolated_db_session.add(valid_match)
    isolated_db_session.commit()
    assert valid_match.id is not None

    # 3. Attempt to UPDATE evidence_id to a mismatching value -> MUST FAIL
    valid_match.evidence_id = ev2.id
    with pytest.raises(IntegrityError):
        isolated_db_session.commit()
    isolated_db_session.rollback()
