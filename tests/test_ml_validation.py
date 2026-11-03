"""
Unit tests for Machine Learning Group-Stratified Holdout and Cross-Validation
Ensures zero data leakage between physical tyre units across all train/test splits and CV folds.
"""
import pytest
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from app.config import WEAR_CLASSES
from app.ml.dataset import WearDataset
from app.ml.train import ModelTrainer


@pytest.fixture(scope="module")
def wear_data():
    dataset_mgr = WearDataset()
    records = dataset_mgr.scan_dataset()
    X, y, groups, paths = dataset_mgr.build_feature_matrix(records)
    return dataset_mgr, X, y, groups


def test_group_holdout_zero_leakage(wear_data):
    """
    Verify that the holdout train/test split satisfies:
    1. train_groups INTERSECT test_groups == EMPTY (zero physical tyre leakage)
    2. Every class (Good, Moderate, Worn) is represented in both train and test sets
    """
    dataset_mgr, X, y, groups = wear_data
    if len(X) < 12:
        pytest.skip("Dataset requires at least 12 samples for group holdout verification.")

    X_train, X_test, y_train, y_test, groups_train, groups_test = dataset_mgr.split_data(
        X, y, groups, test_size=0.25, random_state=42, return_groups=True
    )

    train_tyres = set(str(g) for g in groups_train)
    test_tyres = set(str(g) for g in groups_test)

    # 1. Zero leakage check
    overlap = train_tyres.intersection(test_tyres)
    assert len(overlap) == 0, f"Data leakage detected! Overlapping tyres: {overlap}"

    # 2. Class representation in test set
    test_classes = set(y_test)
    assert len(test_classes) == len(WEAR_CLASSES), f"Test set missing classes: {WEAR_CLASSES}"

    # 3. Class representation in train set
    train_classes = set(y_train)
    assert len(train_classes) == len(WEAR_CLASSES), f"Train set missing classes: {WEAR_CLASSES}"


def test_stratified_group_kfold_cross_validation_leakage(wear_data):
    """
    Verify that in every CV fold:
    1. No physical tyre appears in both train and validation folds
    2. All classes are represented in validation fold
    """
    dataset_mgr, X, y, groups = wear_data
    if len(X) < 12:
        pytest.skip("Dataset requires at least 12 samples for CV verification.")

    X_train, X_test, y_train, y_test, groups_train, groups_test = dataset_mgr.split_data(
        X, y, groups, test_size=0.25, random_state=42, return_groups=True
    )

    sgkf_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    for fold_idx, (f_train, f_val) in enumerate(sgkf_cv.split(X_train, y_train, groups=groups_train), 1):
        fold_train_tyres = set(str(g) for g in groups_train[f_train])
        fold_val_tyres = set(str(g) for g in groups_train[f_val])

        # Overlap must be empty
        leakage = fold_train_tyres.intersection(fold_val_tyres)
        assert len(leakage) == 0, f"CV Fold {fold_idx} leaked tyres: {leakage}"

        # Class distribution check
        val_classes = set(y_train[f_val])
        assert len(val_classes) == len(WEAR_CLASSES), f"CV Fold {fold_idx} missing classes in validation"


def test_model_training_and_metrics_generation(wear_data, tmp_path):
    """Verify that ModelTrainer computes all required metrics and records physical tyre groups."""
    dataset_mgr, X, y, groups = wear_data
    if len(X) < 12:
        pytest.skip("Dataset requires at least 12 samples for training verification.")

    X_train, X_test, y_train, y_test, groups_train, groups_test = dataset_mgr.split_data(
        X, y, groups, test_size=0.25, random_state=42, return_groups=True
    )

    trainer = ModelTrainer(models_dir=tmp_path)
    _, metrics = trainer.train_and_evaluate(
        X_train, X_test, y_train, y_test,
        groups_train=groups_train,
        groups_test=groups_test,
        model_type="rf"
    )

    assert metrics["leakage_check_passed"] is True
    assert len(metrics["train_physical_tyres"]) > 0
    assert len(metrics["test_physical_tyres"]) > 0
    assert len(metrics["cv_fold_details"]) == 3
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["validation_type"] == "Calibrated prototype validation"
