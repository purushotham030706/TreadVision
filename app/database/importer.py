"""
TreadVision: CSV and JSON Batch Importer with Schema Validation
Enables extending tyre and fitment database from external verified records.
"""
from typing import Dict, Any, List, Tuple
from pathlib import Path
import csv
import json
from app.database.connection import SessionLocal
from app.database.models import Tyre, TreadPattern, TyreSize, Vehicle, Fitment

REQUIRED_TYRE_FIELDS = ["manufacturer", "model", "pattern_name", "category"]


def validate_tyre_row(row: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate mandatory fields in CSV/dict row."""
    for f in REQUIRED_TYRE_FIELDS:
        val = row.get(f)
        if not val or not str(val).strip():
            return False, f"Missing required field: '{f}'"
    return True, "Valid"


def import_tyres_from_csv(csv_path: str) -> Dict[str, Any]:
    """
    Import and validate tyre records from a CSV file.
    Expected CSV columns:
    manufacturer,model,pattern_name,category,construction,description,image_path,pattern_type,sizes
    """
    path = Path(csv_path)
    if not path.exists():
        return {"success": False, "message": f"CSV file not found: {csv_path}", "imported_count": 0}

    session = SessionLocal()
    imported_count = 0
    errors = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for line_no, row in enumerate(reader, start=2):
                is_valid, err = validate_tyre_row(row)
                if not is_valid:
                    errors.append(f"Line {line_no}: {err}")
                    continue

                # Check for existing duplicate
                existing = session.query(Tyre).filter(
                    Tyre.manufacturer == row["manufacturer"].strip(),
                    Tyre.model == row["model"].strip()
                ).first()

                if existing:
                    errors.append(f"Line {line_no}: {row['manufacturer']} {row['model']} already exists in database.")
                    continue

                tyre = Tyre(
                    manufacturer=row["manufacturer"].strip(),
                    model=row["model"].strip(),
                    pattern_name=row["pattern_name"].strip(),
                    category=row.get("category", "Touring").strip(),
                    construction=row.get("construction", "Radial").strip(),
                    description=row.get("description", "").strip()
                )
                session.add(tyre)
                session.flush()

                # Optional pattern
                img_path = row.get("image_path", "").strip()
                if img_path:
                    pattern = TreadPattern(
                        tyre_id=tyre.id,
                        image_path=img_path,
                        pattern_type=row.get("pattern_type", "Asymmetric").strip(),
                        symmetry=row.get("symmetry", "Asymmetric").strip(),
                        groove_count=int(row.get("groove_count", 4)),
                        description=row.get("pattern_description", "")
                    )
                    session.add(pattern)

                # Optional sizes (comma-separated, e.g. "205/55R16,225/45R17")
                sizes_raw = row.get("sizes", "").strip()
                if sizes_raw:
                    for s in sizes_raw.split(","):
                        s = s.strip()
                        # Simple parse "205/55R16" or "205/55 R16"
                        try:
                            clean_s = s.replace(" ", "").upper()
                            parts = clean_s.split("/")
                            width = int(parts[0])
                            rem = parts[1].split("R")
                            aspect = int(rem[0])
                            rim = int(rem[1][:2])
                            ts = TyreSize(tyre_id=tyre.id, width=width, aspect_ratio=aspect, rim_diameter=rim)
                            session.add(ts)
                        except Exception:
                            continue

                imported_count += 1

        session.commit()
        return {
            "success": True,
            "imported_count": imported_count,
            "errors": errors,
            "message": f"Successfully imported {imported_count} tyre records."
        }
    except Exception as e:
        session.rollback()
        return {"success": False, "message": f"Import error: {str(e)}", "imported_count": 0}
    finally:
        session.close()
