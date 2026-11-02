"""
SQLAlchemy ORM Models for TreadVision Database
"""
import json
from typing import List, Optional
from sqlalchemy import (
    Column, Integer, String, Float, Text, ForeignKey, create_engine
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
