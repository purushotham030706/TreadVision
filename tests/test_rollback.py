"""
Rollback Verification Test Suite
Confirms that the archived 25-D classifier and matching scaler can be restored,
loaded into WearInferenceEngine, and operated with full backward compatibility.
"""
from pathlib import Path
import pytest
import joblib
import numpy as np

from app.config import BASE_DIR, WEAR_CLASSES
from app.ml.inference import WearInferenceEngine

LEGACY_25D_DIR = BASE_DIR / "models_legacy_25d"
BASELINE_DIR = BASE_DIR / "models_baseline"


def test_archived_legacy_25d_artifacts_exist():
    """Verify that all required 25-D artifacts exist in the legacy archive."""
    archive_dir = LEGACY_25D_DIR if LEGACY_25D_DIR.exists() else BASELINE_DIR
    assert archive_dir.exists(), "No legacy 25-D model archive directory found!"

    required_files = [
        "wear_classifier_rf.joblib",
        "scaler.joblib",
        "label_encoder.joblib",
        "metrics_rf.json"
    ]
    for fname in required_files:
        assert (archive_dir / fname).exists(), f"Missing archived file: {fname} in {archive_dir}"


def test_legacy_25d_scaler_and_model_dimensions():
    """Verify that the archived scaler and model expect exactly 25 features."""
    archive_dir = LEGACY_25D_DIR if LEGACY_25D_DIR.exists() else BASELINE_DIR
    scaler = joblib.load(archive_dir / "scaler.joblib")
    model = joblib.load(archive_dir / "wear_classifier_rf.joblib")

    assert scaler.n_features_in_ == 25, f"Expected 25 features in archived scaler, got {scaler.n_features_in_}"
    assert model.n_features_in_ == 25, f"Expected 25 features in archived model, got {model.n_features_in_}"


def test_rollback_inference_with_25d_engine():
    """Verify that WearInferenceEngine cleanly operates in 25-D mode when pointed to the archive directory."""
    archive_dir = LEGACY_25D_DIR if LEGACY_25D_DIR.exists() else BASELINE_DIR
    engine = WearInferenceEngine(models_dir=archive_dir)

    assert engine.is_model_trained() is True
    assert engine.model_type == "rf"

    # Simulate 25-D dummy feature vector
    vec_25d = np.full(25, 0.05, dtype=np.float32)
    res = engine.predict(vec_25d)

    assert res["is_fallback"] is False
    assert res["predicted_class"] in WEAR_CLASSES
    assert "25D" in res["engine"]
    assert "probabilities" in res
    assert len(res["probabilities"]) == len(WEAR_CLASSES)


def test_dimension_mismatch_guard_on_25d_engine():
    """Verify that passing an incompatible 31-D vector to a 25-D engine is rejected."""
    archive_dir = LEGACY_25D_DIR if LEGACY_25D_DIR.exists() else BASELINE_DIR
    engine = WearInferenceEngine(models_dir=archive_dir)

    # 31-D vector
    vec_31d = np.full(31, 0.05, dtype=np.float32)
    res = engine.predict(vec_31d)

    # Should gracefully catch the dimension mismatch and fall back to CV feature rule engine
    assert res["is_fallback"] is True
    assert "Deterministic CV Feature Engine" in res["engine"]
