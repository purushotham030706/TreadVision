"""
TreadVision: Forensic DBMS Repository and Transaction Manager
Encapsulates ACID transactions, evidence ingestion, multi-table joins,
aggregate queries, cryptographic integrity audits, and analytical view access.
"""
from pathlib import Path
import hashlib
import json
import uuid
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image

from sqlalchemy import text, func, desc
from sqlalchemy.orm import Session

from app.config import EVIDENCE_UPLOADS_DIR, BASE_DIR
from app.database.connection import get_db_session
from app.database.models import (
    Investigation, EvidenceImage, EvidenceFeatures, MatchingRun, MatchResult,
    TreadPattern, Tyre, Vehicle, Fitment, TyreSize
)
from app.image_processing.pipeline import PipelineResult
from app.image_processing.forensic_matcher import ForensicMatchCandidate


def calculate_image_checksum(image_bytes: bytes) -> str:
    """Compute SHA-256 cryptographic checksum of image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()


def store_evidence_file(
    image_input: Union[str, Path, Image.Image, np.ndarray, bytes],
    original_filename: str,
    target_dir: Optional[Path] = None
) -> Tuple[str, str, int, int, str]:
    """
    Store an uploaded or selected evidence image into managed relative storage.
    Returns:
        Tuple of (relative_image_path, checksum, width, height, format_str)
    """
    storage_dir = target_dir or EVIDENCE_UPLOADS_DIR
    storage_dir.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex[:12]
    safe_name = Path(original_filename).name or "evidence_impression.jpg"
    stem = Path(safe_name).stem
    dest_filename = f"{file_id}_{stem}.jpg"
    dest_path = storage_dir / dest_filename

    # Process and write file
    if isinstance(image_input, (str, Path)):
        src = Path(image_input)
        img_bytes = src.read_bytes()
        with Image.open(src) as pil_img:
            width, height = pil_img.size
            img_format = (pil_img.format or "JPEG").upper()
        dest_path.write_bytes(img_bytes)
        checksum = calculate_image_checksum(img_bytes)

    elif isinstance(image_input, Image.Image):
        width, height = image_input.size
        img_format = (image_input.format or "JPEG").upper()
        rgb_img = image_input.convert("RGB")
        rgb_img.save(dest_path, format="JPEG", quality=95)
        img_bytes = dest_path.read_bytes()
        checksum = calculate_image_checksum(img_bytes)

    elif isinstance(image_input, np.ndarray):
        height, width = image_input.shape[:2]
        img_format = "JPEG"
        if len(image_input.shape) == 2:
            cv2.imwrite(str(dest_path), image_input)
        elif image_input.shape[2] == 3:
            cv2.imwrite(str(dest_path), image_input)
        else:
            cv2.imwrite(str(dest_path), cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR))
        img_bytes = dest_path.read_bytes()
        checksum = calculate_image_checksum(img_bytes)

    elif isinstance(image_input, bytes):
        img_bytes = image_input
        dest_path.write_bytes(img_bytes)
        with Image.open(dest_path) as pil_img:
            width, height = pil_img.size
            img_format = (pil_img.format or "JPEG").upper()
        checksum = calculate_image_checksum(img_bytes)

    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Deterministic relative path for database portability
    try:
        relative_path = str(dest_path.relative_to(BASE_DIR)).replace("\\", "/")
    except ValueError:
        relative_path = str(dest_path).replace("\\", "/")

    return relative_path, checksum, width, height, img_format


def get_or_create_investigation(
    session: Session,
    case_number: str,
    title: Optional[str] = None,
    investigation_date: Optional[date] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    status: str = "Active"
) -> Tuple[Investigation, bool]:
    """
    Retrieve existing investigation by case_number or create a new one.
    Returns: (Investigation, created: bool)
    """
    case_num = case_number.strip().upper()
    existing = session.query(Investigation).filter(
        func.upper(Investigation.case_number) == case_num
    ).first()

    if existing:
        return existing, False

    inv = Investigation(
        case_number=case_num,
        title=title or f"Investigation {case_num}",
        investigation_date=investigation_date or date.today(),
        description=description or "Forensic tyre impression investigation.",
        location=location or "Forensic Scene Evidence",
        status=status,
        is_archived=False,
        created_at=datetime.now()
    )
    session.add(inv)
    session.flush()
    return inv, True


def list_investigations(session: Session, include_archived: bool = True) -> List[Investigation]:
    """List investigations ordered by latest creation."""
    query = session.query(Investigation)
    if not include_archived:
        query = query.filter(Investigation.is_archived == False)
    return query.order_by(desc(Investigation.created_at)).all()


def archive_investigation(
    session: Session,
    investigation_id: int,
    archive: bool = True
) -> Optional[Investigation]:
    """Soft-archive or unarchive an investigation to control case visibility without data deletion."""
    inv = session.query(Investigation).filter(Investigation.id == investigation_id).first()
    if inv:
        if archive:
            inv.archive()
        else:
            inv.is_archived = False
            inv.archived_at = None
        session.commit()
    return inv


def verify_evidence_storage_integrity(session: Session) -> Dict[str, Any]:
    """
    Comprehensive cryptographic storage audit:
    1. Cross-references all EvidenceImage records with files on disk.
    2. Computes live SHA-256 hash and compares with recorded image_checksum.
    3. Detects any unindexed orphan files in EVIDENCE_UPLOADS_DIR.
    """
    EVIDENCE_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    evidence_rows = session.query(EvidenceImage).all()

    verified_count = 0
    tampered = []
    missing = []

    indexed_filenames = set()
    for ev in evidence_rows:
        file_path = BASE_DIR / ev.image_path
        indexed_filenames.add(file_path.name)
        if not file_path.exists():
            missing.append({
                "id": ev.id,
                "path": ev.image_path,
                "filename": ev.original_filename
            })
        else:
            live_hash = calculate_image_checksum(file_path.read_bytes())
            if ev.image_checksum and live_hash != ev.image_checksum:
                tampered.append({
                    "id": ev.id,
                    "path": ev.image_path,
                    "expected_checksum": ev.image_checksum,
                    "actual_checksum": live_hash
                })
            else:
                verified_count += 1

    # Check for orphan files
    orphan_files = []
    for p in EVIDENCE_UPLOADS_DIR.iterdir():
        if p.is_file() and p.name != ".gitkeep" and p.name not in indexed_filenames:
            orphan_files.append(str(p.name))

    status = "VERIFIED_SECURE"
    if missing or tampered:
        status = "INTEGRITY_WARNING"
    elif orphan_files:
        status = "ORPHAN_FILES_DETECTED"

    return {
        "status": status,
        "total_evidence_records": len(evidence_rows),
        "verified_intact": verified_count,
        "missing_files": missing,
        "tampered_files": tampered,
        "orphan_files": orphan_files
    }


def cleanup_orphan_evidence_files(session: Session) -> int:
    """Remove untracked orphan image files in evidence upload directory."""
    evidence_rows = session.query(EvidenceImage).all()
    indexed_names = {Path(ev.image_path).name for ev in evidence_rows}
    removed = 0
    if EVIDENCE_UPLOADS_DIR.exists():
        for p in EVIDENCE_UPLOADS_DIR.iterdir():
            if p.is_file() and p.name != ".gitkeep" and p.name not in indexed_names:
                try:
                    p.unlink()
                    removed += 1
                except Exception:
                    pass
    return removed


def save_forensic_match_transaction(
    case_number: str,
    image_input: Any,
    original_filename: str,
    pipeline_result: PipelineResult,
    candidates: List[ForensicMatchCandidate],
    title: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    notes: Optional[str] = None,
    evidence_id: Optional[int] = None,
    execution_time_ms: Optional[float] = None,
    target_dir: Optional[Path] = None,
    session_instance: Optional[Session] = None
) -> Dict[str, Any]:
    """
    Atomic DBMS Transaction:
    Persists Investigation -> EvidenceImage -> EvidenceFeatures -> MatchingRun -> MatchResult (top candidates)
    with complete atomicity and rollback on any failure.
    Supports multiple matching runs per evidence image without duplicating the image on disk.
    """
    close_session = False
    session = session_instance
    if session is None:
        session = get_db_session()
        close_session = True

    created_filepath: Optional[Path] = None

    try:
        # Case 1: Re-running matching on existing evidence image
        if evidence_id is not None:
            evidence = session.query(EvidenceImage).filter(EvidenceImage.id == evidence_id).first()
            if not evidence:
                raise ValueError(f"EvidenceImage id={evidence_id} not found.")
            inv = evidence.investigation
            rel_path = evidence.image_path
            checksum = evidence.image_checksum
        else:
            # Case 2: New evidence upload
            rel_path, checksum, width, height, img_fmt = store_evidence_file(
                image_input, original_filename, target_dir=target_dir
            )
            created_filepath = BASE_DIR / rel_path

            inv, _ = get_or_create_investigation(
                session=session,
                case_number=case_number,
                title=title,
                description=description,
                location=location
            )

            evidence = EvidenceImage(
                investigation_id=inv.id,
                image_path=rel_path,
                original_filename=Path(original_filename).name,
                image_format=img_fmt,
                image_width=width,
                image_height=height,
                image_checksum=checksum,
                upload_timestamp=datetime.now()
            )
            session.add(evidence)
            session.flush()

            # Persist 31-D features
            vec_list = pipeline_result.feature_vector.tolist() if hasattr(pipeline_result.feature_vector, "tolist") else list(pipeline_result.feature_vector)
            edge_json = json.dumps(pipeline_result.features.get("edge", {}))
            tex_json = json.dumps(pipeline_result.features.get("texture", {}))
            shape_json = json.dumps(pipeline_result.features.get("shape", {}))

            features_rec = EvidenceFeatures(
                evidence_id=evidence.id,
                edge_features=edge_json,
                texture_features=tex_json,
                shape_features=shape_json,
                feature_vector=json.dumps(vec_list),
                extraction_method="TreadVision 31D Adaptive Pipeline",
                extraction_timestamp=datetime.now()
            )
            session.add(features_rec)

        # Create MatchingRun entity
        existing_runs_count = session.query(MatchingRun).filter(MatchingRun.evidence_id == evidence.id).count()
        run_number = existing_runs_count + 1
        weights_payload = json.dumps(weights or {
            "edge": 0.30, "texture": 0.25, "shape": 0.25, "local": 0.20
        })

        matching_run = MatchingRun(
            evidence_id=evidence.id,
            run_number=run_number,
            matching_method="Multi-Criteria Fusion v2",
            weights_json=weights_payload,
            run_timestamp=datetime.now(),
            execution_time_ms=execution_time_ms or getattr(pipeline_result, "execution_time_ms", 0.0),
            notes=notes or f"Matching Run #{run_number}"
        )
        session.add(matching_run)
        session.flush()

        # Persist ranked candidate matches for this run
        tyre_ids = [c.tyre_id for c in candidates]
        pattern_rows = session.query(TreadPattern.id, TreadPattern.tyre_id).filter(
            TreadPattern.tyre_id.in_(tyre_ids)
        ).all()
        tyre_to_pattern = {tyre_id: pat_id for pat_id, tyre_id in pattern_rows}

        for cand in candidates:
            pat_id = tyre_to_pattern.get(cand.tyre_id)
            if pat_id is None:
                pat = session.query(TreadPattern).filter(TreadPattern.tyre_id == cand.tyre_id).first()
                if pat:
                    pat_id = pat.id
                    tyre_to_pattern[cand.tyre_id] = pat_id

            if pat_id is not None:
                match_rec = MatchResult(
                    run_id=matching_run.id,
                    evidence_id=evidence.id,
                    tread_pattern_id=pat_id,
                    similarity_score=float(cand.similarity_score),
                    rank=int(cand.rank),
                    matching_method="Multi-Criteria Fusion v2",
                    match_details=json.dumps(cand.component_scores),
                    match_timestamp=datetime.now()
                )
                session.add(match_rec)

        session.commit()

        return {
            "success": True,
            "investigation_id": inv.id,
            "case_number": inv.case_number,
            "evidence_id": evidence.id,
            "run_id": matching_run.id,
            "run_number": run_number,
            "image_path": rel_path,
            "checksum": checksum,
            "matches_stored": len(candidates)
        }

    except Exception as e:
        session.rollback()
        if created_filepath and created_filepath.exists():
            try:
                created_filepath.unlink()
            except Exception:
                pass
        raise e
    finally:
        if close_session:
            session.close()


def query_investigation_summaries_view(session: Session) -> List[Dict[str, Any]]:
    """Query the SQL analytical reporting view v_investigation_summaries."""
    sql = text("""
        SELECT
            investigation_id,
            case_number,
            title,
            status,
            is_archived,
            investigation_date,
            evidence_count,
            total_matching_runs,
            total_matches_recorded,
            highest_similarity_score,
            top_pattern_type,
            top_candidate_tyre
        FROM v_investigation_summaries
        ORDER BY investigation_date DESC, investigation_id DESC
    """)
    rows = session.execute(sql).fetchall()
    results = []
    for r in rows:
        results.append({
            "investigation_id": r[0],
            "case_number": r[1],
            "title": r[2],
            "status": r[3],
            "is_archived": bool(r[4]),
            "investigation_date": str(r[5]),
            "evidence_count": r[6],
            "total_matching_runs": r[7],
            "total_matches_recorded": r[8],
            "highest_similarity_score": round(float(r[9]), 1),
            "top_pattern_type": r[10] or "N/A",
            "top_candidate_tyre": r[11] or "N/A"
        })
    return results


def get_evidence_details_with_matches(
    session: Session,
    evidence_id: int,
    run_id: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Execute comprehensive multi-table join across:
    evidence_images -> investigations -> evidence_features -> matching_runs -> match_results ->
    tread_patterns -> tyres -> fitments -> vehicles & tyre_sizes.
    Includes active cryptographic SHA-256 integrity verification against the file on disk.
    """
    evidence = session.query(EvidenceImage).filter(EvidenceImage.id == evidence_id).first()
    if not evidence:
        return None

    # Cryptographic file verification
    file_on_disk = BASE_DIR / evidence.image_path
    if not file_on_disk.exists():
        checksum_status = "FILE_MISSING"
        checksum_verified = False
        live_checksum = "N/A"
    else:
        live_checksum = calculate_image_checksum(file_on_disk.read_bytes())
        if evidence.image_checksum and live_checksum == evidence.image_checksum:
            checksum_status = "VERIFIED"
            checksum_verified = True
        else:
            checksum_status = "TAMPER_DETECTED"
            checksum_verified = False

    # Retrieve all runs for this evidence
    all_runs = session.query(MatchingRun).filter(
        MatchingRun.evidence_id == evidence_id
    ).order_by(MatchingRun.run_number.asc()).all()

    # Determine active run
    active_run = None
    if run_id:
        active_run = next((r for r in all_runs if r.id == run_id), None)
    if active_run is None and all_runs:
        active_run = all_runs[-1]  # Latest run by default

    # Load matches for active run
    ranked_matches = []
    if active_run:
        matches_query = session.query(
            MatchResult, TreadPattern, Tyre
        ).join(
            TreadPattern, MatchResult.tread_pattern_id == TreadPattern.id
        ).join(
            Tyre, TreadPattern.tyre_id == Tyre.id
        ).filter(
            MatchResult.run_id == active_run.id
        ).order_by(
            MatchResult.rank.asc()
        ).all()

        for match, pattern, tyre in matches_query:
            fitment_rows = session.query(Fitment, Vehicle).join(
                Vehicle, Fitment.vehicle_id == Vehicle.id
            ).filter(
                Fitment.tyre_id == tyre.id
            ).all()
            vehicles = sorted(list(set(f"{v.manufacturer} {v.model}" for f, v in fitment_rows)))
            sizes = [s.formatted_size for s in tyre.sizes]

            details = match.get_match_details()
            ranked_matches.append({
                "rank": match.rank,
                "similarity_score": match.similarity_score,
                "matching_method": match.matching_method,
                "match_timestamp": match.match_timestamp.isoformat(),
                "component_scores": details,
                "tyre": {
                    "id": tyre.id,
                    "manufacturer": tyre.manufacturer,
                    "model": tyre.model,
                    "pattern_name": tyre.pattern_name,
                    "category": tyre.category,
                    "description": tyre.description,
                    "pattern_type": pattern.pattern_type,
                    "symmetry": pattern.symmetry,
                    "groove_count": pattern.groove_count,
                    "image_path": pattern.image_path,
                    "available_sizes": sizes,
                    "compatible_vehicles": vehicles
                }
            })

    features_dict = evidence.features.to_dict() if evidence.features else None

    return {
        "evidence_id": evidence.id,
        "investigation_id": evidence.investigation_id,
        "case_number": evidence.investigation.case_number if evidence.investigation else "UNKNOWN",
        "case_title": evidence.investigation.title if evidence.investigation else "UNKNOWN",
        "case_status": evidence.investigation.status if evidence.investigation else "Active",
        "is_archived": evidence.investigation.is_archived if evidence.investigation else False,
        "image_path": evidence.image_path,
        "original_filename": evidence.original_filename,
        "image_format": evidence.image_format,
        "image_width": evidence.image_width,
        "image_height": evidence.image_height,
        "image_checksum": evidence.image_checksum,
        "live_checksum": live_checksum,
        "checksum_status": checksum_status,
        "checksum_verified": checksum_verified,
        "upload_timestamp": evidence.upload_timestamp.isoformat(),
        "available_runs": [r.to_dict() for r in all_runs],
        "active_run": active_run.to_dict() if active_run else None,
        "features": features_dict,
        "matches": ranked_matches
    }


def get_manufacturer_match_aggregates(session: Session) -> List[Dict[str, Any]]:
    """
    Execute SQL aggregate query grouping match candidate occurrences by tyre manufacturer.
    Computes count of appearances, average similarity, and maximum similarity.
    """
    sql = text("""
        SELECT
            t.manufacturer,
            COUNT(m.id) AS appearance_count,
            ROUND(AVG(m.similarity_score), 2) AS avg_similarity,
            ROUND(MAX(m.similarity_score), 2) AS max_similarity
        FROM match_results m
        JOIN tread_patterns tp ON m.tread_pattern_id = tp.id
        JOIN tyres t ON tp.tyre_id = t.id
        GROUP BY t.manufacturer
        ORDER BY appearance_count DESC, avg_similarity DESC
    """)
    rows = session.execute(sql).fetchall()
    return [
        {
            "manufacturer": r[0],
            "appearance_count": r[1],
            "avg_similarity": float(r[2]),
            "max_similarity": float(r[3])
        }
        for r in rows
    ]
