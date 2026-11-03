"""
TreadVision: Phase 2 Wear-Specific Feature Experiment
Evaluates Configurations A (25-D Baseline), B (31-D Augmented), and C (6-D Wear-Only)
under identical StratifiedGroupKFold and physical-tyre held-out evaluation.
"""
import sys
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, r"c:\Users\puru\Desktop\IP-Event")
from app.config import WEAR_DATA_DIR, WEAR_CLASSES
from app.ml.dataset import WearDataset
from app.ml.models import get_model
from app.image_processing.pipeline import ImageProcessingPipeline

sys.path.insert(0, r"C:\Users\puru\.gemini\antigravity\brain\2ab4a37c-e2ff-463d-a77e-4f68bf3e8933\scratch")
from wear_features_diagnostic import extract_wear_specific_features

WEAR_FEATURE_NAMES = [
    "groove_to_land_contrast",
    "groove_visibility_ratio",
    "sipe_visibility_freq",
    "local_groove_continuity",
    "local_texture_contrast",
    "groove_wall_gradient"
]

PATTERN_MAP = {
    "tyre_g01": "Primacy 4", "tyre_g02": "PremiumContact 6", "tyre_g03": "Turanza T005", "tyre_g04": "EfficientGrip",
    "tyre_m01": "Primacy 4", "tyre_m02": "PremiumContact 6", "tyre_m03": "Turanza T005", "tyre_m04": "EfficientGrip",
    "tyre_w01": "Primacy 4", "tyre_w02": "PremiumContact 6", "tyre_w03": "Turanza T005", "tyre_w04": "EfficientGrip"
}

def cohen_d(x1, x2):
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    pooled = np.sqrt((s1**2 + s2**2) / 2.0)
    return abs(np.mean(x1) - np.mean(x2)) / (pooled if pooled > 0 else 1.0)


def run_experiment():
    print("=" * 80)
    print("PHASE 2: WEAR-SPECIFIC FEATURE EXPERIMENT")
    print("=" * 80)

    pipe = ImageProcessingPipeline()
    ds = WearDataset()
    records = ds.scan_dataset()

    data_rows = []
    for r in records:
        fpath = r["file_path"]
        t_id = r["tyre_id"]
        label = r["label"]
        bgr = cv2.imread(fpath)

        res = pipe.process(bgr)
        vec_25d = res.feature_vector  # 25-D numpy array

        wear_dict = extract_wear_specific_features(bgr)
        vec_wear_6d = np.array([wear_dict[k] for k in WEAR_FEATURE_NAMES], dtype=np.float32)

        vec_31d = np.concatenate([vec_25d, vec_wear_6d])

        item = {
            "file_path": fpath,
            "filename": Path(fpath).name,
            "tyre_id": t_id,
            "label": label,
            "pattern": PATTERN_MAP[t_id],
            "vec_25d": vec_25d,
            "vec_wear_6d": vec_wear_6d,
            "vec_31d": vec_31d,
        }
        item.update(wear_dict)
        data_rows.append(item)

    df_all = pd.DataFrame(data_rows)

    # Encode labels
    label_map = {"good": 0, "moderate": 1, "worn": 2}
    y_all = np.array([label_map[r["label"]] for r in data_rows])
    groups_all = np.array([r["tyre_id"] for r in data_rows])

    # Anti-leakage split (identical to production)
    sgkf_split = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf_split.split(data_rows, y_all, groups=groups_all))

    train_groups = sorted(list(set(groups_all[train_idx])))
    test_groups = sorted(list(set(groups_all[test_idx])))

    print(f"\nTraining Physical Tyres ({len(train_groups)}): {train_groups}")
    print(f"Held-Out Physical Tyres ({len(test_groups)}): {test_groups}")

    # Prepare feature matrices for A, B, C
    X_A_train = np.stack(df_all.iloc[train_idx]["vec_25d"].values)
    X_A_test  = np.stack(df_all.iloc[test_idx]["vec_25d"].values)

    X_B_train = np.stack(df_all.iloc[train_idx]["vec_31d"].values)
    X_B_test  = np.stack(df_all.iloc[test_idx]["vec_31d"].values)

    X_C_train = np.stack(df_all.iloc[train_idx]["vec_wear_6d"].values)
    X_C_test  = np.stack(df_all.iloc[test_idx]["vec_wear_6d"].values)

    y_train = y_all[train_idx]
    y_test  = y_all[test_idx]
    g_train = groups_all[train_idx]
    g_test  = groups_all[test_idx]
    test_files = [df_all.iloc[i]["filename"] for i in test_idx]

    configs = {
        "A (Baseline 25-D)": (X_A_train, X_A_test),
        "B (Augmented 31-D)": (X_B_train, X_B_test),
        "C (Wear-Only 6-D)": (X_C_train, X_C_test),
    }

    results = {}
    models_to_test = ["rf", "svm", "knn"]

    for cfg_name, (X_tr, X_te) in configs.items():
        results[cfg_name] = {}
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        for m_name in models_to_test:
            # 1. Fit on full train split
            clf = get_model(m_name)
            clf.fit(X_tr_s, y_train)

            # Holdout predictions
            y_pred = clf.predict(X_te_s)
            y_proba = clf.predict_proba(X_te_s) if hasattr(clf, "predict_proba") else None

            acc = float(accuracy_score(y_test, y_pred))
            prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
            rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
            f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
            report = classification_report(y_test, y_pred, target_names=WEAR_CLASSES, output_dict=True, zero_division=0)

            # 2. 3-fold CV on train split
            sgkf_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
            cv_accs = []
            cv_f1s = []

            for f_idx, (f_tr_idx, f_val_idx) in enumerate(sgkf_cv.split(X_tr_s, y_train, groups=g_train), 1):
                f_clf = get_model(m_name)
                f_clf.fit(X_tr_s[f_tr_idx], y_train[f_tr_idx])
                f_pred = f_clf.predict(X_tr_s[f_val_idx])
                f_acc = float(accuracy_score(y_train[f_val_idx], f_pred))
                f_f1 = float(f1_score(y_train[f_val_idx], f_pred, average="macro", zero_division=0))
                cv_accs.append(f_acc)
                cv_f1s.append(f_f1)

            sample_preds = []
            for i, fname in enumerate(test_files):
                true_lbl = WEAR_CLASSES[y_test[i]]
                pred_lbl = WEAR_CLASSES[y_pred[i]]
                prob = float(np.max(y_proba[i])) if y_proba is not None else 1.0
                sample_preds.append({
                    "filename": fname,
                    "tyre_id": g_test[i],
                    "true_label": true_lbl,
                    "pred_label": pred_lbl,
                    "confidence": prob,
                    "correct": bool(y_test[i] == y_pred[i])
                })

            results[cfg_name][m_name] = {
                "holdout_acc": acc,
                "macro_prec": prec_macro,
                "macro_rec": rec_macro,
                "macro_f1": f1_macro,
                "confusion_matrix": cm.tolist(),
                "report": report,
                "cv_accs": cv_accs,
                "cv_acc_mean": float(np.mean(cv_accs)),
                "cv_acc_std": float(np.std(cv_accs)),
                "cv_f1s": cv_f1s,
                "cv_f1_mean": float(np.mean(cv_f1s)),
                "cv_f1_std": float(np.std(cv_f1s)),
                "sample_preds": sample_preds
            }

    # Print summary tables
    print("\n" + "=" * 80)
    print("A vs B vs C PERFORMANCE COMPARISON (PRIMARY: RANDOM FOREST)")
    print("=" * 80)

    header = f"{'Metric':<28} | {'Config A (25-D)':<16} | {'Config B (31-D)':<16} | {'Config C (6-D)':<16}"
    print(header)
    print("-" * len(header))

    metrics_to_print = [
        ("Holdout Accuracy", lambda r: f"{r['holdout_acc']:.4f}"),
        ("Holdout Macro Precision", lambda r: f"{r['macro_prec']:.4f}"),
        ("Holdout Macro Recall", lambda r: f"{r['macro_rec']:.4f}"),
        ("Holdout Macro F1", lambda r: f"{r['macro_f1']:.4f}"),
        ("Per-Class F1: Good", lambda r: f"{r['report']['good']['f1-score']:.4f}"),
        ("Per-Class F1: Moderate", lambda r: f"{r['report']['moderate']['f1-score']:.4f}"),
        ("Per-Class F1: Worn", lambda r: f"{r['report']['worn']['f1-score']:.4f}"),
        ("3-Fold CV Mean Acc", lambda r: f"{r['cv_acc_mean']:.4f} +/- {r['cv_acc_std']:.4f}"),
        ("3-Fold CV Acc Scores", lambda r: str([round(x, 4) for x in r['cv_accs']])),
        ("3-Fold CV Mean Macro F1", lambda r: f"{r['cv_f1_mean']:.4f} +/- {r['cv_f1_std']:.4f}"),
        ("3-Fold CV F1 Scores", lambda r: str([round(x, 4) for x in r['cv_f1s']])),
    ]

    for label_str, fn in metrics_to_print:
        val_a = fn(results["A (Baseline 25-D)"]["rf"])
        val_b = fn(results["B (Augmented 31-D)"]["rf"])
        val_c = fn(results["C (Wear-Only 6-D)"]["rf"])
        print(f"{label_str:<28} | {val_a:<16} | {val_b:<16} | {val_c:<16}")

    print("\n" + "=" * 80)
    print("CONFUSION MATRICES (RANDOM FOREST)")
    print("=" * 80)
    for cfg in configs.keys():
        cm = np.array(results[cfg]["rf"]["confusion_matrix"])
        print(f"\nConfiguration: {cfg}")
        print(f"               Pred Good   Pred Mod   Pred Worn")
        print(f"  True Good        {cm[0,0]:<10} {cm[0,1]:<10} {cm[0,2]:<10}")
        print(f"  True Moderate    {cm[1,0]:<10} {cm[1,1]:<10} {cm[1,2]:<10}")
        print(f"  True Worn        {cm[2,0]:<10} {cm[2,1]:<10} {cm[2,2]:<10}")

    print("\n" + "=" * 80)
    print("SVM & KNN COMPARISON")
    print("=" * 80)
    for m in ["svm", "knn"]:
        print(f"\nModel: {m.upper()}")
        print(f"{'Metric':<28} | {'Config A':<16} | {'Config B':<16} | {'Config C':<16}")
        print("-" * 80)
        for label_str, fn in [
            ("Holdout Accuracy", lambda r: f"{r['holdout_acc']:.4f}"),
            ("Holdout Macro F1", lambda r: f"{r['macro_f1']:.4f}"),
            ("CV Mean Accuracy", lambda r: f"{r['cv_acc_mean']:.4f} +/- {r['cv_acc_std']:.4f}"),
            ("CV Mean Macro F1", lambda r: f"{r['cv_f1_mean']:.4f} +/- {r['cv_f1_std']:.4f}"),
        ]:
            val_a = fn(results["A (Baseline 25-D)"][m])
            val_b = fn(results["B (Augmented 31-D)"][m])
            val_c = fn(results["C (Wear-Only 6-D)"][m])
            print(f"{label_str:<28} | {val_a:<16} | {val_b:<16} | {val_c:<16}")

    print("\n" + "=" * 80)
    print("PREDICTIONS FOR HELD-OUT PHYSICAL TYRES (FOCUS: TYRE_G04)")
    print("=" * 80)
    for i, fname in enumerate(test_files):
        t_id = g_test[i]
        true_lbl = WEAR_CLASSES[y_test[i]]
        pred_a = results["A (Baseline 25-D)"]["rf"]["sample_preds"][i]
        pred_b = results["B (Augmented 31-D)"]["rf"]["sample_preds"][i]
        pred_c = results["C (Wear-Only 6-D)"]["rf"]["sample_preds"][i]

        print(f"File: {fname:<25} (Tyre: {t_id}, True: {true_lbl.upper():<8}) | A: {pred_a['pred_label'].upper():<8} (conf={pred_a['confidence']:.2f}) | B: {pred_b['pred_label'].upper():<8} (conf={pred_b['confidence']:.2f}) | C: {pred_c['pred_label'].upper():<8} (conf={pred_c['confidence']:.2f})")

    # Feature Level Diagnostics
    print("\n" + "=" * 80)
    print("FEATURE-LEVEL ANALYSIS OF THE SIX WEAR FEATURES")
    print("=" * 80)
    for f in WEAR_FEATURE_NAMES:
        g = df_all[df_all["label"] == "good"][f].values
        m = df_all[df_all["label"] == "moderate"][f].values
        w = df_all[df_all["label"] == "worn"][f].values
        d_gm = cohen_d(g, m)
        d_mw = cohen_d(m, w)
        d_gw = cohen_d(g, w)
        is_mono = (np.mean(g) > np.mean(m) > np.mean(w)) or (np.mean(g) < np.mean(m) < np.mean(w))
        print(f"Feature: {f}")
        print(f"   Good: {np.mean(g):.4f} +/- {np.std(g):.4f} | Mod: {np.mean(m):.4f} +/- {np.std(m):.4f} | Worn: {np.mean(w):.4f} +/- {np.std(w):.4f}")
        print(f"   Cohen's d: Good vs Mod = {d_gm:.3f} | Mod vs Worn = {d_mw:.3f} | Good vs Worn = {d_gw:.3f}")
        print(f"   Monotonic: {is_mono} | Variance: {np.var(df_all[f]):.6f}")

    # Feature correlation matrix
    print("\n" + "=" * 80)
    print("INTER-FEATURE CORRELATION MATRIX (SIX WEAR FEATURES)")
    print("=" * 80)
    corr_matrix = df_all[WEAR_FEATURE_NAMES].corr()
    print(corr_matrix.round(3).to_string())

    # Save artifacts in dedicated experiment folder
    exp_dir = Path(r"C:\Users\puru\.gemini\antigravity\brain\2ab4a37c-e2ff-463d-a77e-4f68bf3e8933\scratch\phase2_artifacts")
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "phase2_results.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n[+] Experimental results saved to {exp_dir / 'phase2_results.json'}")

if __name__ == "__main__":
    run_experiment()
