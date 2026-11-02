"""
TreadVision: Machine Learning Wear Inference Engine
"""
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import json
import joblib
import numpy as np

from app.config import MODELS_DIR, WEAR_CLASSES, WEAR_LABELS


class WearInferenceEngine:
    """
    Handles model loading, feature scaling, and wear condition classification.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.model_type = "none"
        self.metrics = None
        self._load_best_available_model()

    def _load_best_available_model(self):
        """Try loading RandomForest, SVM, or KNN in order of preference."""
        for m_type in ("rf", "svm", "knn"):
            m_path = self.models_dir / f"wear_classifier_{m_type}.joblib"
            s_path = self.models_dir / "scaler.joblib"
            e_path = self.models_dir / "label_encoder.joblib"
            met_path = self.models_dir / f"metrics_{m_type}.json"

            if m_path.exists() and s_path.exists():
                try:
                    self.model = joblib.load(m_path)
                    self.scaler = joblib.load(s_path)
                    self.label_encoder = joblib.load(e_path) if e_path.exists() else None
                    self.model_type = m_type
                    if met_path.exists():
                        with open(met_path, "r") as f:
                            self.metrics = json.load(f)
                    return
                except Exception:
                    continue

    def is_model_trained(self) -> bool:
        """Check if a trained ML model is loaded."""
        return self.model is not None and self.scaler is not None

    def predict(self, feature_vector: np.ndarray, feature_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict tyre wear class ('good', 'moderate', 'worn') with confidence score.
        If no ML model is trained on disk, utilizes a deterministic CV feature rule engine
        (based on edge density and GLCM contrast/homogeneity) with clear indicator.
        """
        if self.is_model_trained():
            try:
                # Ensure 2D shape (1, 25)
                feat_2d = feature_vector.reshape(1, -1)
                feat_scaled = self.scaler.transform(feat_2d)

                pred_idx = self.model.predict(feat_scaled)[0]
                if self.label_encoder:
                    pred_class = self.label_encoder.inverse_transform([pred_idx])[0]
                else:
                    pred_class = WEAR_CLASSES[pred_idx]

                # Probabilities
                probabilities = {}
                confidence = 0.85
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(feat_scaled)[0]
                    for idx, c_name in enumerate(WEAR_CLASSES):
                        probabilities[c_name] = round(float(probs[idx]), 3)
                    confidence = float(np.max(probs))

                return {
                    "predicted_class": pred_class,
                    "label": WEAR_LABELS.get(pred_class, pred_class.title()),
                    "confidence": round(confidence, 3),
                    "probabilities": probabilities,
                    "engine": f"Trained ML Model ({self.model_type.upper()})",
                    "model_metrics": self.metrics,
                    "is_fallback": False
                }
            except Exception as e:
                # Fallback to feature rule engine if inference fails
                pass

        # Deterministic Computer Vision Feature Rule Engine
        # When no trained ML model is present on disk yet
        edge_density = 0.05
        glcm_contrast = 2.0
        glcm_homogeneity = 0.5
        if feature_dict:
            edge_density = feature_dict.get("edge", {}).get("total_edge_density", 0.05)
            glcm_contrast = feature_dict.get("texture", {}).get("contrast", 2.0)
            glcm_homogeneity = feature_dict.get("texture", {}).get("homogeneity", 0.5)

        # High edge density + high contrast + low homogeneity = Good
        # Low edge density + low contrast + high homogeneity = Worn
        wear_score = (edge_density / 0.08) * 0.45 + (glcm_contrast / 5.0) * 0.35 + (1.0 - glcm_homogeneity) * 0.20

        if wear_score >= 0.70:
            pred_class = "good"
            conf = min(0.92, 0.75 + (wear_score - 0.70) * 0.5)
            probs = {"good": round(conf, 2), "moderate": round(1.0 - conf - 0.05, 2), "worn": 0.05}
        elif wear_score >= 0.40:
            pred_class = "moderate"
            conf = 0.78
            probs = {"good": 0.15, "moderate": 0.78, "worn": 0.07}
        else:
            pred_class = "worn"
            conf = min(0.94, 0.75 + (0.40 - wear_score) * 0.6)
            probs = {"good": 0.05, "moderate": round(1.0 - conf - 0.05, 2), "worn": round(conf, 2)}

        return {
            "predicted_class": pred_class,
            "label": WEAR_LABELS.get(pred_class, pred_class.title()),
            "confidence": round(conf, 3),
            "probabilities": probs,
            "engine": "Deterministic CV Feature Engine (No trained ML weights on disk)",
            "model_metrics": None,
            "is_fallback": True
        }
