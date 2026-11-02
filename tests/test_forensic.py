"""
Unit tests for TreadVision Forensic Similarity Matcher
"""
import pytest
import numpy as np
from app.image_processing.forensic_matcher import ForensicMatcher
from app.image_processing.pipeline import ImageProcessingPipeline


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

    # Create dummy database records
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
