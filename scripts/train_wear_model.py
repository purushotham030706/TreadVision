"""
TreadVision: Machine Learning Wear Model Training CLI
Executes anti-leakage group-aware train/test split, trains Random Forest, SVM,
and KNN models, evaluates performance, and persists artifacts.
"""
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.dataset import WearDataset
from app.ml.train import ModelTrainer


def main():
    print("==================================================")
    print("  TreadVision: Machine Learning Training Pipeline  ")
    print("==================================================")

    dataset_mgr = WearDataset()
    records = dataset_mgr.scan_dataset()
    print(f"\n[1] Discovered {len(records)} labelled tyre images across classes.")

    print("\n[2] Extracting 25 classical computer vision features per image...")
    def progress(cur, tot, p):
        if cur % 6 == 0 or cur == tot:
            print(f"    Processed {cur}/{tot} images...")

    X, y, groups, paths = dataset_mgr.build_feature_matrix(records, progress_callback=progress)
    print(f"    Feature matrix shape: {X.shape}")
    print(f"    Unique physical tyre groups: {len(set(groups))}")

    print("\n[3] Splitting dataset using physical tyre group stratification (Anti-Leakage)...")
    X_train, X_test, y_train, y_test = dataset_mgr.split_data(X, y, groups, test_size=0.25, random_state=42)
    print(f"    Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    trainer = ModelTrainer()

    for m_type in ("rf", "svm", "knn"):
        print(f"\n--- Training {m_type.upper()} Classifier ---")
        _, metrics = trainer.train_and_evaluate(X_train, X_test, y_train, y_test, model_type=m_type)
        print(f"    Accuracy:  {metrics['accuracy'] * 100:.2f}%")
        print(f"    Precision: {metrics['precision'] * 100:.2f}%")
        print(f"    Recall:    {metrics['recall'] * 100:.2f}%")
        print(f"    F1-Score:  {metrics['f1_score'] * 100:.2f}%")
        if metrics.get("cv_scores"):
            print(f"    Cross-Validation Scores: {metrics['cv_scores']} (Mean: {metrics['cv_mean']*100:.2f}%)")
        print(f"    Confusion Matrix:\n      {metrics['confusion_matrix']}")

    print("\n[4] All models, scalers, and evaluation metrics saved to 'models/' directory.")
    print("Training pipeline finished successfully.")


if __name__ == "__main__":
    main()
