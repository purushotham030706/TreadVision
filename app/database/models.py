"""
SQLAlchemy ORM Models for TreadVision Database
"""
import json
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Float, Text, ForeignKey, ForeignKeyConstraint, create_engine,
    DateTime, Date, CheckConstraint, UniqueConstraint, Index, Boolean
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Tyre(Base):
    __tablename__ = "tyres"

    id = Column(Integer, primary_key=True, autoincrement=True)
    manufacturer = Column(String(100), nullable=False, index=True)
    model = Column(String(150), nullable=False, index=True)
    pattern_name = Column(String(150), nullable=False)
    category = Column(String(80), nullable=False, index=True)  # Touring, Ultra High Performance, All-Season, Eco
    construction = Column(String(50), default="Radial")
    description = Column(Text, nullable=True)

    # Relationships
    tread_pattern = relationship("TreadPattern", back_populates="tyre", uselist=False, cascade="all, delete-orphan")
    sizes = relationship("TyreSize", back_populates="tyre", cascade="all, delete-orphan")
    fitments = relationship("Fitment", back_populates="tyre", cascade="all, delete-orphan")
    features = relationship("TreadFeatures", back_populates="tyre", uselist=False, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "pattern_name": self.pattern_name,
            "category": self.category,
            "construction": self.construction,
            "description": self.description,
            "pattern": self.tread_pattern.to_dict() if self.tread_pattern else None,
            "sizes": [s.to_dict() for s in self.sizes],
            "vehicles": [f.vehicle.to_dict() for f in self.fitments if f.vehicle]
        }


class TreadPattern(Base):
    __tablename__ = "tread_patterns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tyre_id = Column(Integer, ForeignKey("tyres.id"), nullable=False, unique=True)
    image_path = Column(String(255), nullable=False)
    pattern_type = Column(String(80), default="Asymmetric")  # Symmetrical, Asymmetric, Directional
    symmetry = Column(String(50), default="Asymmetric")
    groove_count = Column(Integer, default=4)
    description = Column(Text, nullable=True)

    tyre = relationship("Tyre", back_populates="tread_pattern")
    matches = relationship("MatchResult", back_populates="tread_pattern", passive_deletes=True)

    def to_dict(self):
        return {
            "id": self.id,
            "tyre_id": self.tyre_id,
            "image_path": self.image_path,
            "pattern_type": self.pattern_type,
            "symmetry": self.symmetry,
            "groove_count": self.groove_count,
            "description": self.description
        }


class TyreSize(Base):
    __tablename__ = "tyre_sizes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tyre_id = Column(Integer, ForeignKey("tyres.id"), nullable=False, index=True)
    width = Column(Integer, nullable=False)           # e.g., 205
    aspect_ratio = Column(Integer, nullable=False)    # e.g., 55
    rim_diameter = Column(Integer, nullable=False)    # e.g., 16
    load_index = Column(String(20), default="91")     # e.g., 91
    speed_rating = Column(String(10), default="V")    # e.g., V, W, Y

    tyre = relationship("Tyre", back_populates="sizes")

    @property
    def formatted_size(self) -> str:
        return f"{self.width}/{self.aspect_ratio} R{self.rim_diameter} {self.load_index}{self.speed_rating}"

    def to_dict(self):
        return {
            "id": self.id,
            "tyre_id": self.tyre_id,
            "width": self.width,
            "aspect_ratio": self.aspect_ratio,
            "rim_diameter": self.rim_diameter,
            "load_index": self.load_index,
            "speed_rating": self.speed_rating,
            "formatted": self.formatted_size
        }


class Vehicle(Base):
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    manufacturer = Column(String(100), nullable=False, index=True)  # e.g., BMW, Volkswagen
    model = Column(String(120), nullable=False, index=True)         # e.g., 3 Series, Golf
    generation = Column(String(80), nullable=True)                  # e.g., G20, Mk7
    year_from = Column(Integer, nullable=True)
    year_to = Column(Integer, nullable=True)

    fitments = relationship("Fitment", back_populates="vehicle", cascade="all, delete-orphan")

    def to_dict(self):
        year_str = ""
        if self.year_from:
            year_str = f" ({self.year_from}-{self.year_to or 'Present'})"
        return {
            "id": self.id,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "generation": self.generation,
            "year_from": self.year_from,
            "year_to": self.year_to,
            "full_name": f"{self.manufacturer} {self.model} {self.generation or ''}{year_str}".strip()
        }


class Fitment(Base):
    __tablename__ = "fitments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tyre_id = Column(Integer, ForeignKey("tyres.id"), nullable=False, index=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=False, index=True)
    size = Column(String(50), nullable=False)               # e.g., "205/55 R16"
    fitment_type = Column(String(50), default="Standard")   # Standard, Optional OEM, Plus 1
    source = Column(String(100), default="Verified OEM / Public Spec")

    tyre = relationship("Tyre", back_populates="fitments")
    vehicle = relationship("Vehicle", back_populates="fitments")

    def to_dict(self):
        return {
            "id": self.id,
            "tyre_id": self.tyre_id,
            "vehicle_id": self.vehicle_id,
            "size": self.size,
            "fitment_type": self.fitment_type,
            "source": self.source
        }


class TreadFeatures(Base):
    __tablename__ = "tread_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tyre_id = Column(Integer, ForeignKey("tyres.id"), nullable=False, unique=True)
    edge_features = Column(Text, nullable=False)       # JSON string
    texture_features = Column(Text, nullable=False)    # JSON string
    shape_features = Column(Text, nullable=False)      # JSON string
    feature_vector = Column(Text, nullable=False)      # JSON string of normalized float array

    tyre = relationship("Tyre", back_populates="features")

    def get_feature_vector(self) -> List[float]:
        try:
            return json.loads(self.feature_vector)
        except Exception:
            return []

    def to_dict(self):
        return {
            "id": self.id,
            "tyre_id": self.tyre_id,
            "edge_features": json.loads(self.edge_features) if self.edge_features else {},
            "texture_features": json.loads(self.texture_features) if self.texture_features else {},
            "shape_features": json.loads(self.shape_features) if self.shape_features else {},
            "feature_vector": self.get_feature_vector()
        }


class Investigation(Base):
    __tablename__ = "investigations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_number = Column(String(64), nullable=False, unique=True, index=True)
    title = Column(String(150), nullable=False)
    investigation_date = Column(Date, nullable=False, default=date.today)
    description = Column(Text, nullable=True)
    location = Column(String(200), nullable=True)
    status = Column(String(50), nullable=False, default="Active")
    is_archived = Column(Boolean, nullable=False, default=False)
    archived_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)

    __table_args__ = (
        CheckConstraint(
            "status IN ('Active', 'Under Review', 'Closed', 'Archived')",
            name="check_investigation_status"
        ),
    )

    # Relationships
    evidence_images = relationship(
        "EvidenceImage",
        back_populates="investigation",
        cascade="save-update, merge",
        passive_deletes=True
    )

    def archive(self):
        """Soft-delete / archive investigation to prevent irreversible evidence loss."""
        self.status = "Archived"
        self.is_archived = True
        self.archived_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "case_number": self.case_number,
            "title": self.title,
            "investigation_date": str(self.investigation_date),
            "description": self.description,
            "location": self.location,
            "status": self.status,
            "is_archived": self.is_archived,
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "evidence_count": len(self.evidence_images) if self.evidence_images else 0
        }


class EvidenceImage(Base):
    __tablename__ = "evidence_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    investigation_id = Column(
        Integer,
        ForeignKey("investigations.id", ondelete="RESTRICT"),
        nullable=False,
        index=True
    )
    image_path = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    image_format = Column(String(20), nullable=False)
    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)
    upload_timestamp = Column(DateTime, nullable=False, default=datetime.now)
    image_checksum = Column(String(64), nullable=True, index=True)

    __table_args__ = (
        CheckConstraint(
            "image_width > 0 AND image_height > 0",
            name="check_positive_dimensions"
        ),
    )

    # Relationships
    investigation = relationship("Investigation", back_populates="evidence_images")
    features = relationship(
        "EvidenceFeatures",
        back_populates="evidence_image",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True
    )
    matching_runs = relationship(
        "MatchingRun",
        back_populates="evidence_image",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="MatchingRun.run_number"
    )
    matches = relationship(
        "MatchResult",
        back_populates="evidence_image",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="MatchResult.rank"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "investigation_id": self.investigation_id,
            "image_path": self.image_path,
            "original_filename": self.original_filename,
            "image_format": self.image_format,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "upload_timestamp": self.upload_timestamp.isoformat() if self.upload_timestamp else None,
            "image_checksum": self.image_checksum,
            "has_features": self.features is not None,
            "runs_count": len(self.matching_runs) if self.matching_runs else 0,
            "match_count": len(self.matches) if self.matches else 0
        }


class EvidenceFeatures(Base):
    __tablename__ = "evidence_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evidence_id = Column(
        Integer,
        ForeignKey("evidence_images.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )
    edge_features = Column(Text, nullable=False)       # JSON string
    texture_features = Column(Text, nullable=False)    # JSON string
    shape_features = Column(Text, nullable=False)      # JSON string
    feature_vector = Column(Text, nullable=False)      # JSON string of 31-D float vector
    extraction_method = Column(String(100), nullable=False, default="TreadVision 31D Adaptive Pipeline")
    extraction_timestamp = Column(DateTime, nullable=False, default=datetime.now)

    # Relationships
    evidence_image = relationship("EvidenceImage", back_populates="features")

    def get_feature_vector(self) -> List[float]:
        try:
            return json.loads(self.feature_vector)
        except Exception:
            return []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "evidence_id": self.evidence_id,
            "edge_features": json.loads(self.edge_features) if self.edge_features else {},
            "texture_features": json.loads(self.texture_features) if self.texture_features else {},
            "shape_features": json.loads(self.shape_features) if self.shape_features else {},
            "feature_vector": self.get_feature_vector(),
            "extraction_method": self.extraction_method,
            "extraction_timestamp": self.extraction_timestamp.isoformat() if self.extraction_timestamp else None
        }


class MatchingRun(Base):
    __tablename__ = "matching_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evidence_id = Column(
        Integer,
        ForeignKey("evidence_images.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    run_number = Column(Integer, nullable=False, default=1)
    matching_method = Column(String(100), nullable=False, default="Multi-Criteria Fusion v2")
    weights_json = Column(Text, nullable=False)
    run_timestamp = Column(DateTime, nullable=False, default=datetime.now)
    execution_time_ms = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("evidence_id", "run_number", name="uq_evidence_run_number"),
        UniqueConstraint("id", "evidence_id", name="uq_matching_runs_id_evidence_id"),
        CheckConstraint("run_number >= 1", name="check_valid_run_number"),
    )

    # Relationships
    evidence_image = relationship("EvidenceImage", back_populates="matching_runs")
    matches = relationship(
        "MatchResult",
        back_populates="matching_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="MatchResult.rank",
        overlaps="evidence_image,matches"
    )

    def get_weights(self) -> Dict[str, float]:
        try:
            return json.loads(self.weights_json) if self.weights_json else {}
        except Exception:
            return {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "evidence_id": self.evidence_id,
            "run_number": self.run_number,
            "matching_method": self.matching_method,
            "weights": self.get_weights(),
            "run_timestamp": self.run_timestamp.isoformat() if self.run_timestamp else None,
            "execution_time_ms": self.execution_time_ms,
            "notes": self.notes,
            "matches_count": len(self.matches) if self.matches else 0,
            "match_count": len(self.matches) if self.matches else 0
        }


class MatchResult(Base):
    __tablename__ = "match_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(
        Integer,
        nullable=False,
        index=True
    )
    evidence_id = Column(
        Integer,
        ForeignKey("evidence_images.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    tread_pattern_id = Column(
        Integer,
        ForeignKey("tread_patterns.id", ondelete="RESTRICT"),
        nullable=False,
        index=True
    )
    similarity_score = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)
    matching_method = Column(String(100), nullable=False, default="Multi-Criteria Fusion v2")
    match_details = Column(Text, nullable=True)         # JSON string of component scores
    match_timestamp = Column(DateTime, nullable=False, default=datetime.now)

    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "evidence_id"],
            ["matching_runs.id", "matching_runs.evidence_id"],
            ondelete="CASCADE",
            name="fk_match_results_run_evidence"
        ),
        CheckConstraint(
            "similarity_score >= 0.0 AND similarity_score <= 100.0",
            name="check_valid_similarity_score"
        ),
        CheckConstraint("rank >= 1", name="check_valid_rank"),
        UniqueConstraint("run_id", "tread_pattern_id", name="uq_run_pattern_match"),
        UniqueConstraint("run_id", "rank", name="uq_run_rank_match"),
        Index("ix_match_results_run_rank", "run_id", "rank"),
        Index("ix_match_results_evidence_rank", "evidence_id", "rank"),
    )

    # Relationships
    matching_run = relationship("MatchingRun", back_populates="matches", foreign_keys=[run_id, evidence_id], overlaps="evidence_image,matches")
    evidence_image = relationship("EvidenceImage", back_populates="matches", foreign_keys=[evidence_id], overlaps="matching_run,matches")
    tread_pattern = relationship("TreadPattern", back_populates="matches")

    def get_match_details(self) -> Dict[str, Any]:
        try:
            return json.loads(self.match_details) if self.match_details else {}
        except Exception:
            return {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "evidence_id": self.evidence_id,
            "tread_pattern_id": self.tread_pattern_id,
            "similarity_score": self.similarity_score,
            "rank": self.rank,
            "matching_method": self.matching_method,
            "match_details": self.get_match_details(),
            "match_timestamp": self.match_timestamp.isoformat() if self.match_timestamp else None
        }
