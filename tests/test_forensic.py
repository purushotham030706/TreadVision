"""
Unit tests for TreadVision Forensic Similarity Matcher
Validates mathematical bounds, weight renormalization on ORB failure, and vehicle traceability.
"""
import pytest
import numpy as np
from app.image_processing.forensic_matcher import ForensicMatcher
from app.image_processing.pipeline import ImageProcessingPipeline
from app.database.models import Tyre


@pytest.fixture
def dummy_pipeline_result():
    pipeline = ImageProcessingPipeline(target_size=(256, 256))
    dummy_img = np.full((256, 256, 3), 50, dtype=np.uint8)
    return pipeline.process(dummy_img)


def test_forensic_matcher_initialization():
    matcher = ForensicMatcher()
    assert matcher.weights["edge"] == 0.30
    assert matcher.weights["texture"] == 0.25
    assert matcher.weights["shape"] == 0.25
    assert matcher.weights["local"] == 0.20
    assert "Forensic Analysis Notice" in matcher.FORENSIC_DISCLAIMER


def test_forensic_matching_ranking(dummy_pipeline_result):
    matcher = ForensicMatcher()

    db_records = [
        {
            "id": 1,
            "manufacturer": "Michelin",
            "model": "Primacy 4",
            "pattern_name": "Primacy 4 Asymmetric",
            "pattern_type": "Asymmetric",
            "symmetry": "Asymmetric",
            "features": dummy_pipeline_result.features,
            "image_path": "non_existent.jpg",
            "sizes": ["205/55 R16"],
            "vehicles": ["BMW 3 Series", "VW Golf"],
            "description": "High performance touring tyre"
        },
        {
            "id": 2,
            "manufacturer": "Continental",
            "model": "PremiumContact 6",
            "pattern_name": "PC6",
            "pattern_type": "Asymmetric",
            "symmetry": "Asymmetric",
            "features": dummy_pipeline_result.features,
            "image_path": "non_existent.jpg",
            "sizes": ["225/45 R17"],
            "vehicles": ["Audi A4"],
            "description": "Sport touring tyre"
        }
    ]

    candidates = matcher.match(dummy_pipeline_result, db_records, top_k=2)
    assert len(candidates) == 2
    assert candidates[0].rank == 1
    assert candidates[1].rank == 2
    assert candidates[0].similarity_score >= candidates[1].similarity_score
    assert candidates[0].component_scores is not None
    assert "edge_similarity" in candidates[0].component_scores


def test_orb_failure_handling_and_weight_renormalization(dummy_pipeline_result):
    """
    Verify that when local ORB keypoints are unavailable (e.g. non-existent image or zero keypoints):
    1. Local feature similarity is None (never 0.5)
    2. Local feature status is 'INSUFFICIENT_LOCAL_FEATURES'
    3. Overall score is strictly renormalized across available edge, texture, and shape components.
    """
    matcher = ForensicMatcher()
    query_gray = dummy_pipeline_result.stages.get("Grayscale")

    overall, comp_scores = matcher.calculate_similarity(
        query_features=dummy_pipeline_result.features,
        query_gray=query_gray,
        db_tyre_features=dummy_pipeline_result.features,
        ref_image_path="non_existent_file.jpg"
    )

    # Local feature must be None, NOT an arbitrary 0.5 or 50%
    assert comp_scores["local_pattern_similarity"] is None
    assert comp_scores["local_feature_status"] == "INSUFFICIENT_LOCAL_FEATURES"

    # Verify weight renormalization:
    # S_overall = (w_e * S_e + w_t * S_t + w_s * S_s) / (w_e + w_t + w_s)
    w_e = matcher.weights["edge"]
    w_t = matcher.weights["texture"]
    w_s = matcher.weights["shape"]
    w_sum = w_e + w_t + w_s

    expected_overall = (
        w_e * comp_scores["edge_similarity"] +
        w_t * comp_scores["texture_similarity"] +
        w_s * comp_scores["shape_similarity"]
    ) / w_sum

    assert abs(overall - round(expected_overall, 1)) <= 0.1


def test_identical_match_mathematical_bounds(dummy_pipeline_result):
    """Verify that identical features return mathematical 100% component scores."""
    matcher = ForensicMatcher()
    query_gray = dummy_pipeline_result.stages.get("Grayscale")

    _, comp_scores = matcher.calculate_similarity(
        query_features=dummy_pipeline_result.features,
        query_gray=query_gray,
        db_tyre_features=dummy_pipeline_result.features,
        ref_image_path=""
    )

    assert comp_scores["edge_similarity"] == 100.0
    assert comp_scores["texture_similarity"] == 100.0
    assert comp_scores["shape_similarity"] == 100.0


def test_vehicle_fitment_traceability_from_database(isolated_db_session):
    """
    Verify that vehicle applications originate strictly from database relationships
    (Tyre -> Fitment -> Vehicle) rather than hardcoded strings.
    """
    michelin = isolated_db_session.query(Tyre).filter(
        Tyre.manufacturer == "Michelin", Tyre.model == "Primacy 4"
    ).first()

    assert michelin is not None
    assert len(michelin.fitments) > 0

    fitment_vehicles = [f"{f.vehicle.manufacturer} {f.vehicle.model}" for f in michelin.fitments if f.vehicle]
    assert "Volkswagen Golf" in fitment_vehicles
    assert "BMW 3 Series" in fitment_vehicles
