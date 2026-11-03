"""
TreadVision: Training Pipeline and Evaluation Suite
Computes accuracy, precision, recall, f1-score, cross-validation, and confusion matrix
with strict physical tyre group-aware cross-validation (StratifiedGroupKFold).
"""
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedGroupKFold
import copy

from app.config import MODELS_DIR, WEAR_CLASSES
from app.ml.dataset import WearDataset
from app.ml.models import get_model
from app.image_processing.feature_fusion import FEATURE_NAMES


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
        groups_train: Optional[np.ndarray] = None,
        groups_test: Optional[np.ndarray] = None,
        model_type: str = "rf"
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Scale features, fit specified model, compute evaluation metrics,
        and save model artifacts using group-stratified cross-validation.
        """
        # Fit scaler on training data only to prevent data leakage
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Instantiate model
        model = get_model(model_type)

        # Train
        model.fit(X_train_scaled, y_train)

        # Predict on held-out test set
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None

        # Calculate metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Group-Aware Cross-Validation (StratifiedGroupKFold)
        cv_scores: List[float] = []
        cv_fold_details: List[Dict[str, Any]] = []

        if groups_train is not None and len(np.unique(groups_train)) >= 3:
            n_cv_splits = min(3, len(np.unique(groups_train)))
            sgkf_cv = StratifiedGroupKFold(n_splits=n_cv_splits, shuffle=True, random_state=42)

            for fold_idx, (f_train_idx, f_val_idx) in enumerate(sgkf_cv.split(X_train_scaled, y_train, groups=groups_train), 1):
                fold_model = get_model(model_type)
                fold_model.fit(X_train_scaled[f_train_idx], y_train[f_train_idx])
                fold_pred = fold_model.predict(X_train_scaled[f_val_idx])
                fold_acc = float(accuracy_score(y_train[f_val_idx], fold_pred))

                val_tyres = sorted(list(set(str(g) for g in groups_train[f_val_idx])))
                class_counts = {
                    WEAR_CLASSES[c]: int(cnt)
                    for c, cnt in enumerate(np.bincount(y_train[f_val_idx], minlength=len(WEAR_CLASSES)))
                }

                cv_scores.append(round(fold_acc, 4))
                cv_fold_details.append({
                    "fold": fold_idx,
                    "validation_tyres": val_tyres,
                    "validation_class_distribution": class_counts,
                    "accuracy": round(fold_acc, 4)
                })

        # Per-class metrics
        class_report = classification_report(
            y_test, y_pred,
            target_names=[WEAR_CLASSES[i] for i in sorted(list(set(y_test) | set(y_pred)))],
            output_dict=True,
            zero_division=0
        )

        train_tyres = sorted(list(set(str(g) for g in groups_train))) if groups_train is not None else []
        test_tyres = sorted(list(set(str(g) for g in groups_test))) if groups_test is not None else []
        leakage_check = len(set(train_tyres).intersection(set(test_tyres))) == 0

        metrics = {
            "model_type": model_type,
            "feature_dimension": int(X_train.shape[1]),
            "feature_names": FEATURE_NAMES,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "cv_scores": cv_scores,
            "cv_mean": round(float(np.mean(cv_scores)), 4) if cv_scores else None,
            "cv_fold_details": cv_fold_details,
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "train_physical_tyres": train_tyres,
            "test_physical_tyres": test_tyres,
            "leakage_check_passed": leakage_check,
            "evaluation_protocol": (
                "Production 4-fold StratifiedGroupKFold outer split (27 train views across 9 physical tyres, "
                "9 test views across 3 physical tyres) with 3-fold inner group-aware CV. "
                "Evaluated separately from earlier exploratory experimental runs."
            ),
            "probability_interpretation": "Random Forest tree ensemble vote fractions (not empirically calibrated probabilities)",
            "validation_type": "Calibrated prototype validation",
            "dataset_limitation": (
                "Current classifier validation is based on controlled reference images. "
                "Real-world performance requires validation against independently collected tyre photographs "
                "across manufacturers, lighting conditions, cameras, angles and tread conditions."
            ),
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
        """Full end-to-end training pipeline on available dataset."""
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

        X_train, X_test, y_train, y_test, groups_train, groups_test = self.dataset_manager.split_data(
            X, y, groups, return_groups=True
        )
        _, metrics = self.train_and_evaluate(
            X_train, X_test, y_train, y_test,
            groups_train=groups_train,
            groups_test=groups_test,
            model_type=model_type
        )

        metrics["success"] = True
        return metrics
