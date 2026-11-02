"""
TreadVision: Training Pipeline and Evaluation Suite
Computes accuracy, precision, recall, f1-score, cross-validation, and confusion matrix.
"""
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from app.config import MODELS_DIR, WEAR_CLASSES
from app.ml.dataset import WearDataset
from app.ml.models import get_model


class ModelTrainer:
    """
    Manages the complete training workflow from feature normalization
    to multi-metric evaluation and model artifact persistence.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_manager = WearDataset()
        self.scaler = StandardScaler()

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        model_type: str = "rf"
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Scale features, fit specified model, compute evaluation metrics,
        and save model artifacts.
        """
        # Fit scaler on training data only to prevent data leakage
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Instantiate model
        model = get_model(model_type)

        # Train
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None

        # Calculate metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Cross-validation score on training fold if sample size permits
        cv_scores = []
        if len(y_train) >= 9:
            cv = StratifiedKFold(n_splits=min(3, np.min(np.bincount(y_train))))
            scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
            cv_scores = [round(float(s), 4) for s in scores]

        # Per-class metrics
        class_report = classification_report(
            y_test, y_pred,
            target_names=[WEAR_CLASSES[i] for i in sorted(list(set(y_test) | set(y_pred)))],
            output_dict=True,
            zero_division=0
        )

        metrics = {
            "model_type": model_type,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "cv_scores": cv_scores,
            "cv_mean": round(float(np.mean(cv_scores)), 4) if cv_scores else None,
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "classification_report": class_report,
            "classes": WEAR_CLASSES
        }

        # Save artifacts
        model_path = self.models_dir / f"wear_classifier_{model_type}.joblib"
        scaler_path = self.models_dir / "scaler.joblib"
        encoder_path = self.models_dir / "label_encoder.joblib"
        metrics_path = self.models_dir / f"metrics_{model_type}.json"

        joblib.dump(model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.dataset_manager.label_encoder, encoder_path)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return model, metrics

    def run_training_pipeline(self, model_type: str = "rf") -> Dict[str, Any]:
        """
        Full end-to-end training pipeline on available dataset.
        """
        records = self.dataset_manager.scan_dataset()
        if len(records) < 6:
            return {
                "success": False,
                "message": "Model training requires labelled tyre images in data/raw/tyre_wear/ (minimum 6 samples needed)."
            }

        X, y, groups, _ = self.dataset_manager.build_feature_matrix(records)
        if len(X) < 6:
            return {
                "success": False,
                "message": "Insufficient valid features extracted from images."
            }

        X_train, X_test, y_train, y_test = self.dataset_manager.split_data(X, y, groups)
        _, metrics = self.train_and_evaluate(X_train, X_test, y_train, y_test, model_type=model_type)

        metrics["success"] = True
        return metrics
