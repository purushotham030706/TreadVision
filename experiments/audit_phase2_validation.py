"""
TreadVision: Comprehensive Phase 2 Validation Audit Script
Compares:
1. Exact Old vs Regenerated dataset files, hashes, pixel stats, and 25-D vectors for held-out tyres
2. Exact models, pipelines, seeds, splits between Old (66.67%) and Phase 2 (100%)
3. Verification of holdout isolation and leakage checks
4. Exact feature vectors for tyre_g04 before vs after
5. Feature ordering and composition audit for Configuration B
"""
import sys
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sys.path.insert(0, r"c:\Users\puru\Desktop\IP-Event")
from app.config import WEAR_CLASSES
from app.ml.dataset import WearDataset
from app.ml.models import get_model
from app.image_processing.pipeline import ImageProcessingPipeline

sys.path.insert(0, r"C:\Users\puru\.gemini\antigravity\brain\2ab4a37c-e2ff-463d-a77e-4f68bf3e8933\scratch")
from wear_features_diagnostic import extract_wear_specific_features
from phase2_experiment import WEAR_FEATURE_NAMES

def run_audit():
    pipe = ImageProcessingPipeline()

    p_base = Path(r"data\raw\tyre_wear_baseline")
    p_curr = Path(r"data\raw\tyre_wear")

    test_tyres = ["tyre_g04", "tyre_m02", "tyre_w03"]
    classes = {"tyre_g04": "good", "tyre_m02": "moderate", "tyre_w03": "worn"}

    print("=" * 80)
    print("SECTION 1: DATASET COMPARISON (OLD BASELINE VS REGENERATED DATASET)")
    print("=" * 80)

    audit_rows = []
    for tid in test_tyres:
        cls = classes[tid]
        for view in [1, 2, 3]:
            fn = f"{tid}_view{view}.jpg"
            f_old = p_base / cls / fn
            f_new = p_curr / cls / fn

            h_old = hashlib.md5(f_old.read_bytes()).hexdigest()
            h_new = hashlib.md5(f_new.read_bytes()).hexdigest()

            img_old = cv2.imread(str(f_old))
            img_new = cv2.imread(str(f_new))

            res_old = pipe.process(img_old)
            res_new = pipe.process(img_new)

            audit_rows.append({
                "tyre_id": tid,
                "label": cls,
                "view": f"view{view}",
                "old_md5": h_old[:8],
                "new_md5": h_new[:8],
                "identical_bytes": (h_old == h_new),
                "old_shape": img_old.shape,
                "new_shape": img_new.shape,
                "old_pixel_mean": round(float(np.mean(img_old)), 2),
                "new_pixel_mean": round(float(np.mean(img_new)), 2),
                "old_pixel_std": round(float(np.std(img_old)), 2),
                "new_pixel_std": round(float(np.std(img_new)), 2),
                "old_edge_density": round(float(res_old.features["edge"]["total_edge_density"]), 4),
                "new_edge_density": round(float(res_new.features["edge"]["total_edge_density"]), 4),
                "old_glcm_contrast": round(float(res_old.features["texture"]["contrast"]), 2),
                "new_glcm_contrast": round(float(res_new.features["texture"]["contrast"]), 2),
                "old_glcm_homog": round(float(res_old.features["texture"]["homogeneity"]), 4),
                "new_glcm_homog": round(float(res_new.features["texture"]["homogeneity"]), 4),
                "old_canny_edges": int(np.count_nonzero(res_old.stages["Canny Edges"])),
                "new_canny_edges": int(np.count_nonzero(res_new.stages["Canny Edges"])),
                "vec_old": res_old.feature_vector,
                "vec_new": res_new.feature_vector
            })

    df_test_audit = pd.DataFrame(audit_rows)
    cols_display = [
        "tyre_id", "view", "identical_bytes",
        "old_pixel_mean", "new_pixel_mean",
        "old_edge_density", "new_edge_density",
        "old_glcm_contrast", "new_glcm_contrast",
        "old_canny_edges", "new_canny_edges"
    ]
    print(df_test_audit[cols_display].to_string())

    print("\n" + "=" * 80)
    print("SECTION 2: EVALUATION OF OLD BASELINE DATASET (VERIFYING 66.67%)")
    print("=" * 80)

    # Load entire OLD baseline dataset
    ds_old = WearDataset()
    # Temporarily point to baseline dir
    records_old = []
    for cls in ["good", "moderate", "worn"]:
        for f in (p_base / cls).glob("*.jpg"):
            parts = f.name.split("_")
            tid = f"{parts[0]}_{parts[1]}"
            records_old.append({"file_path": str(f), "label": cls, "tyre_id": tid})

    X_old_list, y_old_list, g_old_list, fn_old_list = [], [], [], []
    for r in records_old:
        bgr = cv2.imread(r["file_path"])
        res = pipe.process(bgr)
        X_old_list.append(res.feature_vector)
        y_old_list.append({"good": 0, "moderate": 1, "worn": 2}[r["label"]])
        g_old_list.append(r["tyre_id"])
        fn_old_list.append(Path(r["file_path"]).name)

    X_old = np.array(X_old_list, dtype=np.float32)
    y_old = np.array(y_old_list)
    g_old = np.array(g_old_list)

    # Split using same StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    tr_idx_old, te_idx_old = next(sgkf.split(X_old, y_old, groups=g_old))

    print(f"Old Dataset Train Tyres: {sorted(list(set(g_old[tr_idx_old])))}")
    print(f"Old Dataset Test Tyres:  {sorted(list(set(g_old[te_idx_old])))}")

    scaler_old = StandardScaler()
    X_tr_old_s = scaler_old.fit_transform(X_old[tr_idx_old])
    X_te_old_s = scaler_old.transform(X_old[te_idx_old])

    for m_name in ["rf", "svm", "knn"]:
        clf_old = get_model(m_name)
        clf_old.fit(X_tr_old_s, y_old[tr_idx_old])
        pred_old = clf_old.predict(X_te_old_s)
        acc_old = accuracy_score(y_old[te_idx_old], pred_old)
        print(f"Old Baseline {m_name.upper()} Holdout Accuracy: {acc_old * 100:.2f}%")
        cm_old = confusion_matrix(y_old[te_idx_old], pred_old, labels=[0, 1, 2])
        print(f"  Confusion Matrix:\n{cm_old}")

    print("\nOld Dataset Predictions for held-out tyres:")
    rf_old = get_model("rf")
    rf_old.fit(X_tr_old_s, y_old[tr_idx_old])
    rf_old_preds = rf_old.predict(X_te_old_s)
    for i, idx in enumerate(te_idx_old):
        true_name = WEAR_CLASSES[y_old[idx]]
        pred_name = WEAR_CLASSES[rf_old_preds[i]]
        print(f"  {fn_old_list[idx]:<25} ({g_old[idx]}, True: {true_name.upper():<8}) -> Pred: {pred_name.upper():<8}")

    print("\n" + "=" * 80)
    print("SECTION 3: HOLDOUT ISOLATION & LEAKAGE CHECK")
    print("=" * 80)
    train_tyres_set = set(g_old[tr_idx_old])
    test_tyres_set = set(g_old[te_idx_old])
    overlap = train_tyres_set.intersection(test_tyres_set)
    print(f"Train Physical Tyres: {train_tyres_set}")
    print(f"Test Physical Tyres:  {test_tyres_set}")
    print(f"Physical Tyre Intersection: {overlap} (Zero leakage verified: {len(overlap) == 0})")

    print("\n" + "=" * 80)
    print("SECTION 4: FEATURE CONFIGURATION B AUDIT (FEATURE NAMES AND ORDERING)")
    print("=" * 80)
    from app.image_processing.feature_fusion import FEATURE_NAMES as BASE_FEATURE_NAMES
    full_31d_names = BASE_FEATURE_NAMES + WEAR_FEATURE_NAMES
    print(f"Base 25-D Feature Count: {len(BASE_FEATURE_NAMES)}")
    print(f"Appended Wear Feature Count: {len(WEAR_FEATURE_NAMES)}")
    print(f"Total Vector Dimension: {len(full_31d_names)}")
    print("Feature Ordering (First 5 and Last 6):")
    print(f"  [0..4]:   {full_31d_names[:5]}")
    print(f"  [25..30]: {full_31d_names[25:]}")

if __name__ == "__main__":
    run_audit()
