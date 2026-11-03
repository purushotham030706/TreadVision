"""
TreadVision: Machine Learning Wear Model Training & Evaluation Report CLI
Executes anti-leakage group-aware train/test split, group-stratified cross-validation,
trains Random Forest, SVM, and KNN models, evaluates performance, and persists artifacts.
"""
import sys
from pathlib import Path
import json
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import WEAR_CLASSES
from app.ml.dataset import WearDataset
from app.ml.train import ModelTrainer


def main():
    print("=" * 70)
    print("TREADVISION WEAR CLASSIFIER - TRAINING & EVALUATION REPORT")
    print("=" * 70)

    dataset_mgr = WearDataset()
    records = dataset_mgr.scan_dataset()

    X, y, groups, paths = dataset_mgr.build_feature_matrix(records)

    unique_groups = sorted(list(set(str(g) for g in groups)))
    samples_per_class = {c: int(np.sum(y == idx)) for idx, c in enumerate(WEAR_CLASSES)}
    aug_factor = len(records) // len(unique_groups) if unique_groups else 0

    print("DATASET SUMMARY:")
    print(f"  Total samples:         {len(records)}")
    print(f"  Physical tyre groups:  {len(unique_groups)}")
    print(f"  Classes:               {', '.join([c.capitalize() for c in WEAR_CLASSES])}")
    print(f"  Samples per class:     {samples_per_class}")
    print(f"  Augmentation factor:   {aug_factor}x per physical tyre")
    print(f"  Validation type:       Calibrated prototype validation")
    print(f"  Feature dimensions:    {X.shape[1]} classical CV descriptors (edges, texture, moments)")

    # Anti-leakage group-stratified holdout split
    X_train, X_test, y_train, y_test, groups_train, groups_test = dataset_mgr.split_data(
        X, y, groups, test_size=0.25, random_state=42, return_groups=True
    )

    train_tyres = sorted(list(set(str(g) for g in groups_train)))
    test_tyres = sorted(list(set(str(g) for g in groups_test)))
    overlap = set(train_tyres).intersection(set(test_tyres))
    leakage_status = "PASSED (0 overlapping tyres)" if len(overlap) == 0 else f"FAILED ({len(overlap)} overlapping tyres)"

    print("\nGROUP-STRATIFIED SPLIT:")
    print(f"  Train samples:         {len(X_train)} ({len(train_tyres)} physical tyres)")
    print(f"  Test samples:          {len(X_test)} ({len(test_tyres)} physical tyres)")
    print(f"  Train physical tyres:  {train_tyres}")
    print(f"  Test physical tyres:   {test_tyres}")
    print(f"  Leakage check:         train_groups INTERSECT test_groups == EMPTY -> {leakage_status}")

    trainer = ModelTrainer()
    models_to_train = [("rf", "Random Forest"), ("svm", "Support Vector Machine"), ("knn", "K-Nearest Neighbors")]

    for m_code, m_name in models_to_train:
        print(f"\n{'-'*70}")
        print(f"MODEL: {m_name.upper()} ({m_code.upper()})")
        print(f"{'-'*70}")

        _, metrics = trainer.train_and_evaluate(
            X_train, X_test, y_train, y_test,
            groups_train=groups_train,
            groups_test=groups_test,
            model_type=m_code
        )

        print(f"CROSS-VALIDATION (StratifiedGroupKFold, K=3):")
        for f in metrics.get("cv_fold_details", []):
            print(f"  Fold {f['fold']}: accuracy = {f['accuracy']*100:.2f}% | val tyres = {f['validation_tyres']} | class dist = {f['validation_class_distribution']}")
        print(f"  Mean CV accuracy:      {metrics.get('cv_mean', 0.0)*100:.2f}%\n")

        print("HELD-OUT TEST SET EVALUATION:")
        print(f"  Accuracy:              {metrics['accuracy'] * 100:.2f}%")
        print(f"  Weighted Precision:    {metrics['precision'] * 100:.2f}%")
        print(f"  Weighted Recall:       {metrics['recall'] * 100:.2f}%")
        print(f"  Weighted F1-Score:     {metrics['f1_score'] * 100:.2f}%")
        print("  Confusion Matrix (Rows: True, Cols: Pred):")
        cm = metrics["confusion_matrix"]
        print(f"    Good:     {cm[0]}")
        print(f"    Moderate: {cm[1]}")
        print(f"    Worn:     {cm[2]}")

        print("\n  Per-Class Metrics:")
        cr = metrics.get("classification_report", {})
        for c in WEAR_CLASSES:
            if c in cr:
                c_data = cr[c]
                print(f"    {c.capitalize():<10} Precision: {c_data['precision']*100:.1f}%  Recall: {c_data['recall']*100:.1f}%  F1: {c_data['f1-score']*100:.1f}%  (Support: {c_data['support']})")

    print("\n" + "=" * 70)
    print("ALL MODELS, SCALERS, AND EVALUATION METRICS SAVED TO 'models/'")
    print("=" * 70)


if __name__ == "__main__":
    main()
