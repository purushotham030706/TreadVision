"""
TreadVision: Phase 4 Pattern-Conditioned Normalization Experiment
Investigates whether pattern-conditioned normalization reduces tread-pattern-specific
variation while preserving monotonic wear trajectories across Good -> Moderate -> Worn.

Evaluates:
  1. Pre-normalization baseline (Configuration B: 31-D Augmented)
  2. Strategy 1: Pattern Reference Normalization (Catalogue/Unworn Template Ratio)
  3. Strategy 2: Pattern In-Fold Centering & Scaling (Training Pattern Mean/Std)
  4. Strategy 3: Pattern Indicator Feature Conditioning (One-Hot Augmentation with Unknown Fallback)
  5. Pattern dependence metrics: Between-pattern variance at constant wear vs within-pattern wear separation
  6. Robustness under Pattern Uncertainty / Unknown Pattern fallback
"""
import sys
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

sys.path.insert(0, r"c:\Users\puru\Desktop\IP-Event")
from app.config import WEAR_CLASSES
from app.ml.dataset import WearDataset
from app.ml.models import get_model
from app.image_processing.feature_fusion import FeatureFusion

sys.path.insert(0, r"C:\Users\puru\.gemini\antigravity\brain\2ab4a37c-e2ff-463d-a77e-4f68bf3e8933\scratch")
from wear_features_diagnostic import extract_wear_specific_features
from phase2_experiment import PATTERN_MAP, WEAR_FEATURE_NAMES
from phase3_experiment import process_with_canny_mode

PATTERNS = ["Primacy 4", "PremiumContact 6", "Turanza T005", "EfficientGrip"]

def extract_all_phase4_data():
    ds = WearDataset()
    records = ds.scan_dataset()
    label_map = {"good": 0, "moderate": 1, "worn": 2}

    items = []
    for r in records:
        fpath = r["file_path"]
        tid = r["tyre_id"]
        lbl = r["label"]
        y_val = label_map[lbl]
        pat = PATTERN_MAP[tid]

        # Use Phase 3 adaptive Canny pipeline
        res = process_with_canny_mode(fpath, mode="adaptive")

        item = {
            "file_path": fpath,
            "filename": Path(fpath).name,
            "tyre_id": tid,
            "label": lbl,
            "y": y_val,
            "pattern": pat,
            "vec_31d": res["vec_31d"],
            "vec_25d": res["vec_25d"],
            "vec_wear_6d": res["vec_wear_6d"],
            "edge_density": res["edge_density"],
            "features": res["features"],
            "wear_dict": res["wear_dict"]
        }
        items.append(item)

    return items


def compute_pattern_dependence_metrics(X: np.ndarray, labels: list, patterns: list):
    """
    Computes:
      1. Between-pattern variance at constant wear level (Good, Moderate, Worn)
      2. Within-pattern wear separation (Good vs Moderate, Moderate vs Worn)
      3. Pattern Dependence Ratio (Between-Pattern Var / Total Wear Var)
    """
    unique_labels = sorted(list(set(labels)))
    unique_patterns = sorted(list(set(patterns)))

    # Between-pattern variance for each wear class
    between_pat_vars = {}
    for l in unique_labels:
        mask_l = np.array(labels) == l
        pat_means = []
        for p in unique_patterns:
            mask_lp = mask_l & (np.array(patterns) == p)
            if np.any(mask_lp):
                pat_means.append(np.mean(X[mask_lp], axis=0))
        if len(pat_means) > 1:
            # variance across pattern centroids
            between_pat_vars[l] = float(np.mean(np.var(pat_means, axis=0)))
        else:
            between_pat_vars[l] = 0.0

    # Within-pattern wear separation: distance between Good and Moderate, Moderate and Worn centroids per pattern
    gm_dists = []
    mw_dists = []
    for p in unique_patterns:
        mask_p = np.array(patterns) == p
        mask_g = mask_p & (np.array(labels) == 0)
        mask_m = mask_p & (np.array(labels) == 1)
        mask_w = mask_p & (np.array(labels) == 2)
        if np.any(mask_g) and np.any(mask_m):
            mu_g = np.mean(X[mask_g], axis=0)
            mu_m = np.mean(X[mask_m], axis=0)
            gm_dists.append(float(np.linalg.norm(mu_g - mu_m)))
        if np.any(mask_m) and np.any(mask_w):
            mu_m = np.mean(X[mask_m], axis=0)
            mu_w = np.mean(X[mask_w], axis=0)
            mw_dists.append(float(np.linalg.norm(mu_m - mu_w)))

    # Overall class separability (Bhattacharyya or distance between class means)
    mu_g_all = np.mean(X[np.array(labels) == 0], axis=0)
    mu_m_all = np.mean(X[np.array(labels) == 1], axis=0)
    mu_w_all = np.mean(X[np.array(labels) == 2], axis=0)

    dist_gm_all = float(np.linalg.norm(mu_g_all - mu_m_all))
    dist_mw_all = float(np.linalg.norm(mu_m_all - mu_w_all))

    return {
        "between_pat_var_good": between_pat_vars.get(0, 0.0),
        "between_pat_var_moderate": between_pat_vars.get(1, 0.0),
        "between_pat_var_worn": between_pat_vars.get(2, 0.0),
        "mean_between_pat_var": float(np.mean(list(between_pat_vars.values()))),
        "within_pat_gm_dist": float(np.mean(gm_dists)) if gm_dists else 0.0,
        "within_pat_mw_dist": float(np.mean(mw_dists)) if mw_dists else 0.0,
        "global_gm_dist": dist_gm_all,
        "global_mw_dist": dist_mw_all
    }


def run_phase4():
    print("=" * 80)
    print("PHASE 4: PATTERN-CONDITIONED NORMALIZATION & FINAL MODEL SELECTION")
    print("=" * 80)

    dataset_items = extract_all_phase4_data()
    n_samples = len(dataset_items)
    print(f"Loaded {n_samples} dataset items across 12 physical tyres and 4 pattern families.")

    y_all = np.array([item["y"] for item in dataset_items])
    groups_all = np.array([item["tyre_id"] for item in dataset_items])
    patterns_all = [item["pattern"] for item in dataset_items]
    X_31d_all = np.stack([item["vec_31d"] for item in dataset_items])

    # Standard StratifiedGroupKFold split
    sgkf_split = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf_split.split(dataset_items, y_all, groups=groups_all))

    y_train, y_test = y_all[train_idx], y_all[test_idx]
    g_train, g_test = groups_all[train_idx], groups_all[test_idx]
    pat_train = [patterns_all[i] for i in train_idx]
    pat_test = [patterns_all[i] for i in test_idx]

    print(f"\nTrain Tyres ({len(train_idx)} views, 9 tyres): {sorted(list(set(g_train)))}")
    print(f"Test Tyres  ({len(test_idx)} views, 3 tyres): {sorted(list(set(g_test)))}")

    # -------------------------------------------------------------------------
    # NORMALIZATION STRATEGIES DEFINITION
    # -------------------------------------------------------------------------
    # Strategy 0: Pre-normalization Baseline (Configuration B: 31-D Augmented)
    # Scaled with standard scaler fitted strictly on train fold.

    # Strategy 1: Pattern Reference Prototype Normalization (Forensic Catalogue Reference)
    # Each pattern has a known nominal unworn reference prototype R_p.
    # To avoid test leakage, R_p is calculated strictly from unworn (Good) samples in training set,
    # or if missing (EfficientGrip has no Good in train!), it falls back to the pattern's earliest available wear state
    # or population unworn prototype.
    # Let's inspect unworn reference prototype construction:
    unworn_refs = {}
    for p in PATTERNS:
        # Check if Good is in train for pattern p
        mask_pg = [train_idx[i] for i, (pt, yt) in enumerate(zip(pat_train, y_train)) if pt == p and yt == 0]
        if len(mask_pg) > 0:
            unworn_refs[p] = np.mean(X_31d_all[mask_pg], axis=0)
        else:
            # For EfficientGrip, no Good in train!
            # Realistic scenario: In real world, reference catalog has OEM unworn scan.
            # In purely data-driven training set: Fallback to global unworn prototype or nearest train state.
            pass

    # Compute global unworn reference from training set Good samples
    train_good_idx = [train_idx[i] for i, yt in enumerate(y_train) if yt == 0]
    global_unworn_ref = np.mean(X_31d_all[train_good_idx], axis=0)

    # For any pattern missing Good in train, set fallback
    for p in PATTERNS:
        if p not in unworn_refs:
            unworn_refs[p] = global_unworn_ref.copy()

    def normalize_by_reference(X_in, pat_list):
        X_out = np.zeros_like(X_in)
        for i, (row, p) in enumerate(zip(X_in, pat_list)):
            ref = unworn_refs.get(p, global_unworn_ref)
            denom = np.where(np.abs(ref) > 1e-4, ref, 1.0)
            X_out[i] = row / denom
        return X_out

    # Strategy 2: Pattern In-Fold Centering and Scaling
    # Normalizes each pattern by the mean and std of that pattern in the training set.
    def fit_transform_pattern_zscore(X_tr, pat_tr, X_te, pat_te):
        pat_means = {}
        pat_stds = {}
        global_mean = np.mean(X_tr, axis=0)
        global_std = np.std(X_tr, axis=0) + 1e-6
        for p in PATTERNS:
            mask = np.array(pat_tr) == p
            if np.sum(mask) > 1:
                pat_means[p] = np.mean(X_tr[mask], axis=0)
                pat_stds[p] = np.std(X_tr[mask], axis=0) + 1e-6
            else:
                pat_means[p] = global_mean
                pat_stds[p] = global_std

        X_tr_norm = np.zeros_like(X_tr)
        for i, (row, p) in enumerate(zip(X_tr, pat_tr)):
            X_tr_norm[i] = (row - pat_means.get(p, global_mean)) / pat_stds.get(p, global_std)

        X_te_norm = np.zeros_like(X_te)
        for i, (row, p) in enumerate(zip(X_te, pat_te)):
            X_te_norm[i] = (row - pat_means.get(p, global_mean)) / pat_stds.get(p, global_std)

        return X_tr_norm, X_te_norm

    # Strategy 3: Pattern Indicator Feature Augmentation (One-Hot Conditioning)
    # Augments 31-D with 4-D one-hot vector for pattern identity, with fallback [0,0,0,0] for unknown pattern.
    def augment_pattern_onehot(X_in, pat_list, zero_out=False):
        onehots = []
        for p in pat_list:
            oh = np.zeros(len(PATTERNS), dtype=np.float32)
            if not zero_out and p in PATTERNS:
                oh[PATTERNS.index(p)] = 1.0
            onehots.append(oh)
        return np.hstack([X_in, np.array(onehots)])

    # Strategy 4: External Perfect Catalogue Reference (OEM Reference)
    # Suppose every pattern has its true unworn prototype available in an external forensic database.
    oem_unworn_refs = {}
    for p in PATTERNS:
        mask_pg = [i for i, item in enumerate(dataset_items) if item["pattern"] == p and item["y"] == 0]
        oem_unworn_refs[p] = np.mean(X_31d_all[mask_pg], axis=0)

    def normalize_by_oem_catalog(X_in, pat_list):
        X_out = np.zeros_like(X_in)
        for i, (row, p) in enumerate(zip(X_in, pat_list)):
            ref = oem_unworn_refs.get(p, global_unworn_ref)
            denom = np.where(np.abs(ref) > 1e-4, ref, 1.0)
            X_out[i] = row / denom
        return X_out

    # -------------------------------------------------------------------------
    # EVALUATION SUITE
    # -------------------------------------------------------------------------
    configs = {
        "Config B (31-D Unnormalized Baseline)": {
            "type": "baseline",
            "desc": "Standard 31-D Augmented with standard train-set z-score"
        },
        "Config D1 (In-Fold Pattern Centering / Z-Score)": {
            "type": "pattern_zscore",
            "desc": "Centering by pattern mean and std within training set"
        },
        "Config D2 (Data-Driven Reference Ratio)": {
            "type": "ref_ratio_data",
            "desc": "Ratio to training-set unworn reference (global fallback for missing)"
        },
        "Config D3 (OEM Catalogue Reference Ratio)": {
            "type": "ref_ratio_oem",
            "desc": "Ratio to OEM external catalogue unworn reference prototype"
        },
        "Config D4 (Pattern Indicator Augmentation)": {
            "type": "one_hot",
            "desc": "31-D concatenated with 4-D pattern one-hot indicator"
        },
        "Config D4-Unknown (Unknown Pattern Fallback)": {
            "type": "one_hot_unknown",
            "desc": "31-D + one-hot where test set pattern is UNKNOWN (all zeros)"
        }
    }

    results = {}

    print("\n" + "=" * 80)
    print("EVALUATING CONFIGURATIONS ACROSS HOLDOUT AND 3-FOLD GROUP CV")
    print("=" * 80)

    for cfg_name, cfg_info in configs.items():
        cfg_type = cfg_info["type"]
        
        # Prepare Train & Test feature matrices
        if cfg_type == "baseline":
            X_tr = X_31d_all[train_idx]
            X_te = X_31d_all[test_idx]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
        elif cfg_type == "pattern_zscore":
            X_tr_raw = X_31d_all[train_idx]
            X_te_raw = X_31d_all[test_idx]
            X_tr_s, X_te_s = fit_transform_pattern_zscore(X_tr_raw, pat_train, X_te_raw, pat_test)
        elif cfg_type == "ref_ratio_data":
            X_tr_norm = normalize_by_reference(X_31d_all[train_idx], pat_train)
            X_te_norm = normalize_by_reference(X_31d_all[test_idx], pat_test)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_norm)
            X_te_s = scaler.transform(X_te_norm)
        elif cfg_type == "ref_ratio_oem":
            X_tr_norm = normalize_by_oem_catalog(X_31d_all[train_idx], pat_train)
            X_te_norm = normalize_by_oem_catalog(X_31d_all[test_idx], pat_test)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_norm)
            X_te_s = scaler.transform(X_te_norm)
        elif cfg_type == "one_hot":
            X_tr_oh = augment_pattern_onehot(X_31d_all[train_idx], pat_train, zero_out=False)
            X_te_oh = augment_pattern_onehot(X_31d_all[test_idx], pat_test, zero_out=False)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_oh)
            X_te_s = scaler.transform(X_te_oh)
        elif cfg_type == "one_hot_unknown":
            X_tr_oh = augment_pattern_onehot(X_31d_all[train_idx], pat_train, zero_out=False)
            X_te_oh = augment_pattern_onehot(X_31d_all[test_idx], pat_test, zero_out=True)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_oh)
            X_te_s = scaler.transform(X_te_oh)

        # Train Random Forest
        rf = get_model("rf")
        rf.fit(X_tr_s, y_train)

        y_pred = rf.predict(X_te_s)
        probs = rf.predict_proba(X_te_s)
        acc = float(accuracy_score(y_test, y_pred))
        f1_m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        rep = classification_report(y_test, y_pred, target_names=WEAR_CLASSES, output_dict=True, zero_division=0)

        # 3-Fold Stratified Group CV on Training Set
        sgkf_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        cv_accs, cv_f1s = [], []
        for f_tr, f_val in sgkf_cv.split(train_idx, y_train, groups=g_train):
            val_pts = [pat_train[k] for k in f_val]
            tr_pts = [pat_train[k] for k in f_tr]
            
            if cfg_type == "baseline":
                s_cv = StandardScaler()
                f_tr_s = s_cv.fit_transform(X_31d_all[train_idx[f_tr]])
                f_val_s = s_cv.transform(X_31d_all[train_idx[f_val]])
            elif cfg_type == "pattern_zscore":
                f_tr_s, f_val_s = fit_transform_pattern_zscore(
                    X_31d_all[train_idx[f_tr]], tr_pts,
                    X_31d_all[train_idx[f_val]], val_pts
                )
            elif cfg_type == "ref_ratio_data":
                f_tr_n = normalize_by_reference(X_31d_all[train_idx[f_tr]], tr_pts)
                f_val_n = normalize_by_reference(X_31d_all[train_idx[f_val]], val_pts)
                s_cv = StandardScaler()
                f_tr_s = s_cv.fit_transform(f_tr_n)
                f_val_s = s_cv.transform(f_val_n)
            elif cfg_type == "ref_ratio_oem":
                f_tr_n = normalize_by_oem_catalog(X_31d_all[train_idx[f_tr]], tr_pts)
                f_val_n = normalize_by_oem_catalog(X_31d_all[train_idx[f_val]], val_pts)
                s_cv = StandardScaler()
                f_tr_s = s_cv.fit_transform(f_tr_n)
                f_val_s = s_cv.transform(f_val_n)
            elif cfg_type in ["one_hot", "one_hot_unknown"]:
                f_tr_oh = augment_pattern_onehot(X_31d_all[train_idx[f_tr]], tr_pts, zero_out=False)
                f_val_oh = augment_pattern_onehot(X_31d_all[train_idx[f_val]], val_pts, zero_out=(cfg_type == "one_hot_unknown"))
                s_cv = StandardScaler()
                f_tr_s = s_cv.fit_transform(f_tr_oh)
                f_val_s = s_cv.transform(f_val_oh)

            m_cv = get_model("rf")
            m_cv.fit(f_tr_s, y_train[f_tr])
            pv = m_cv.predict(f_val_s)
            cv_accs.append(float(accuracy_score(y_train[f_val], pv)))
            cv_f1s.append(float(f1_score(y_train[f_val], pv, average="macro", zero_division=0)))

        # Also test SVM and KNN on CV
        svm_accs, knn_accs = [], []
        for f_tr, f_val in sgkf_cv.split(train_idx, y_train, groups=g_train):
            val_pts = [pat_train[k] for k in f_val]
            tr_pts = [pat_train[k] for k in f_tr]
            if cfg_type == "baseline":
                s_cv = StandardScaler()
                f_tr_s = s_cv.fit_transform(X_31d_all[train_idx[f_tr]])
                f_val_s = s_cv.transform(X_31d_all[train_idx[f_val]])
            elif cfg_type == "ref_ratio_oem":
                f_tr_n = normalize_by_oem_catalog(X_31d_all[train_idx[f_tr]], tr_pts)
                f_val_n = normalize_by_oem_catalog(X_31d_all[train_idx[f_val]], val_pts)
                s_cv = StandardScaler()
                f_tr_s = s_cv.fit_transform(f_tr_n)
                f_val_s = s_cv.transform(f_val_n)
            elif cfg_type == "pattern_zscore":
                f_tr_s, f_val_s = fit_transform_pattern_zscore(
                    X_31d_all[train_idx[f_tr]], tr_pts,
                    X_31d_all[train_idx[f_val]], val_pts
                )
            else:
                s_cv = StandardScaler()
                f_tr_s = s_cv.fit_transform(X_31d_all[train_idx[f_tr]])
                f_val_s = s_cv.transform(X_31d_all[train_idx[f_val]])
            
            svm_m = get_model("svm")
            svm_m.fit(f_tr_s, y_train[f_tr])
            svm_accs.append(float(accuracy_score(y_train[f_val], svm_m.predict(f_val_s))))

            knn_m = get_model("knn")
            knn_m.fit(f_tr_s, y_train[f_tr])
            knn_accs.append(float(accuracy_score(y_train[f_val], knn_m.predict(f_val_s))))

        # Pattern dependence metrics
        pat_metrics = compute_pattern_dependence_metrics(X_tr_s, y_train.tolist(), pat_train)

        # Held out sample predictions
        heldout_preds = []
        for i, idx in enumerate(test_idx):
            heldout_preds.append({
                "filename": dataset_items[idx]["filename"],
                "tyre_id": dataset_items[idx]["tyre_id"],
                "pattern": dataset_items[idx]["pattern"],
                "true_label": WEAR_CLASSES[y_test[i]],
                "pred_label": WEAR_CLASSES[y_pred[i]],
                "prob": [float(p) for p in probs[i]],
                "confidence": float(np.max(probs[i]))
            })

        results[cfg_name] = {
            "type": cfg_type,
            "holdout_acc": acc,
            "holdout_f1": f1_m,
            "cv_acc_mean": float(np.mean(cv_accs)),
            "cv_acc_std": float(np.std(cv_accs)),
            "cv_f1_mean": float(np.mean(cv_f1s)),
            "cv_f1_std": float(np.std(cv_f1s)),
            "cv_svm_mean": float(np.mean(svm_accs)),
            "cv_knn_mean": float(np.mean(knn_accs)),
            "confusion_matrix": cm.tolist(),
            "report": rep,
            "pattern_metrics": pat_metrics,
            "heldout_preds": heldout_preds
        }

        print(f"\n{cfg_name}:")
        print(f"  Holdout Acc: {acc*100:.1f}% | Macro F1: {f1_m:.4f}")
        print(f"  RF CV Acc:   {np.mean(cv_accs)*100:.1f}% +/- {np.std(cv_accs)*100:.1f}% | CV F1: {np.mean(cv_f1s):.4f}")
        print(f"  SVM CV Acc:  {np.mean(svm_accs)*100:.1f}% | KNN CV Acc: {np.mean(knn_accs)*100:.1f}%")
        print(f"  Between-Pattern Var (Mean across wear classes): {pat_metrics['mean_between_pat_var']:.4f}")
        print(f"  Global Wear Separation: Good-Mod Dist={pat_metrics['global_gm_dist']:.2f} | Mod-Worn Dist={pat_metrics['global_mw_dist']:.2f}")

    # -------------------------------------------------------------------------
    # DETAILED COMPARISON OF FEATURE DISTRIBUTIONS BEFORE VS AFTER NORMALIZATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FEATURE DISTRIBUTIONS BEFORE VS AFTER NORMALIZATION (KEY FEATURES)")
    print("=" * 80)
    
    # Let's inspect 4 key features across patterns:
    # 1. Edge density (index 0)
    # 2. GLCM Contrast (index 4)
    # 3. Groove-to-land contrast (index 25)
    # 4. Sipe visibility frequency (index 27)
    key_feat_indices = [0, 4, 25, 27]
    key_feat_names = ["edge_density", "glcm_contrast", "groove_to_land_contrast", "sipe_visibility_freq"]

    X_base = X_31d_all.copy()
    X_oem_norm = normalize_by_oem_catalog(X_31d_all, patterns_all)

    for f_idx, f_name in zip(key_feat_indices, key_feat_names):
        print(f"\nFeature: {f_name}")
        print(f"  {'Pattern':<18} | {'Good (Raw)':<12} | {'Good (Norm)':<12} | {'Mod (Raw)':<12} | {'Mod (Norm)':<12} | {'Worn (Raw)':<12} | {'Worn (Norm)':<12}")
        print("  " + "-" * 95)
        for p in PATTERNS:
            row_raw_g = [X_base[i, f_idx] for i, itm in enumerate(dataset_items) if itm["pattern"] == p and itm["y"] == 0]
            row_nrm_g = [X_oem_norm[i, f_idx] for i, itm in enumerate(dataset_items) if itm["pattern"] == p and itm["y"] == 0]
            row_raw_m = [X_base[i, f_idx] for i, itm in enumerate(dataset_items) if itm["pattern"] == p and itm["y"] == 1]
            row_nrm_m = [X_oem_norm[i, f_idx] for i, itm in enumerate(dataset_items) if itm["pattern"] == p and itm["y"] == 1]
            row_raw_w = [X_base[i, f_idx] for i, itm in enumerate(dataset_items) if itm["pattern"] == p and itm["y"] == 2]
            row_nrm_w = [X_oem_norm[i, f_idx] for i, itm in enumerate(dataset_items) if itm["pattern"] == p and itm["y"] == 2]

            val_rg = f"{np.mean(row_raw_g):.4f}" if row_raw_g else "N/A"
            val_ng = f"{np.mean(row_nrm_g):.4f}" if row_nrm_g else "N/A"
            val_rm = f"{np.mean(row_raw_m):.4f}" if row_raw_m else "N/A"
            val_nm = f"{np.mean(row_nrm_m):.4f}" if row_nrm_m else "N/A"
            val_rw = f"{np.mean(row_raw_w):.4f}" if row_raw_w else "N/A"
            val_nw = f"{np.mean(row_nrm_w):.4f}" if row_nrm_w else "N/A"

            print(f"  {p:<18} | {val_rg:<12} | {val_ng:<12} | {val_rm:<12} | {val_nm:<12} | {val_rw:<12} | {val_nw:<12}")

    # -------------------------------------------------------------------------
    # MONOTONICITY AUDIT ACROSS ALL 4 PATTERNS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MONOTONICITY AUDIT: GOOD -> MODERATE -> WORN TRAJECTORY")
    print("=" * 80)
    for p in PATTERNS:
        print(f"\nPattern Family: {p}")
        for lbl_name, y_val in [("GOOD", 0), ("MODERATE", 1), ("WORN", 2)]:
            idxs = [i for i, itm in enumerate(dataset_items) if itm["pattern"] == p and itm["y"] == y_val]
            mean_raw_edge = np.mean(X_base[idxs, 0])
            mean_nrm_edge = np.mean(X_oem_norm[idxs, 0])
            mean_raw_glc = np.mean(X_base[idxs, 25])
            mean_nrm_glc = np.mean(X_oem_norm[idxs, 25])
            print(f"  {lbl_name:<8}: Edge Density: Raw={mean_raw_edge:.4f} -> Norm={mean_nrm_edge:.4f} | Groove Contrast: Raw={mean_raw_glc:.4f} -> Norm={mean_nrm_glc:.4f}")

    # -------------------------------------------------------------------------
    # HELD-OUT AUDIT SPECIFICALLY FOR TYRE_G04
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("HELD-OUT AUDIT FOR TYRE_G04 (EFFICIENTGRIP GOOD) ACROSS CONFIGURATIONS")
    print("=" * 80)
    for cfg_name, res in results.items():
        g04_preds = [p for p in res["heldout_preds"] if p["tyre_id"] == "tyre_g04"]
        print(f"\n{cfg_name}:")
        for p in g04_preds:
            print(f"  {p['filename']}: Pred={p['pred_label'].upper()} (Conf={p['confidence']:.2f}) [P(Good)={p['prob'][0]:.2f}, P(Mod)={p['prob'][1]:.2f}, P(Worn)={p['prob'][2]:.2f}]")

    # Save artifacts
    p4_dir = Path(r"C:\Users\puru\.gemini\antigravity\brain\2ab4a37c-e2ff-463d-a77e-4f68bf3e8933\scratch\phase4_artifacts")
    p4_dir.mkdir(parents=True, exist_ok=True)
    with open(p4_dir / "phase4_results.json", "w") as fp:
        json.dump(results, fp, indent=2)

    print(f"\n[+] Phase 4 results saved to {p4_dir / 'phase4_results.json'}")


if __name__ == "__main__":
    run_phase4()
