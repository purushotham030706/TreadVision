"""
TreadVision: Phase 3 Image-Processing Robustness Experiment
Evaluates Fixed Canny (50, 150) vs Adaptive Canny across:
1. Standard nominal dataset under StratifiedGroupKFold protocol
2. Edge density distributions by class (Good, Moderate, Worn)
3. Monotonicity across matched tread-pattern families
4. Environmental robustness under perturbations (brightness, contrast, noise, blur)
5. Model metrics (Holdout, CV Mean +/- Std, Macro F1, Confusion Matrix)
"""
import sys
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
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
from app.image_processing.pipeline import ImageProcessingPipeline, PipelineResult
from app.image_processing.edges import detect_canny_edges, detect_sobel_edges, detect_laplacian_edges
from app.image_processing.morphology import apply_morphology
from app.image_processing.segmentation import segment_tread_region
from app.image_processing.enhancement import enhance_contrast
from app.image_processing.preprocessing import validate_image, resize_image, to_grayscale, reduce_noise
from app.image_processing.feature_fusion import FeatureFusion

sys.path.insert(0, r"C:\Users\puru\.gemini\antigravity\brain\2ab4a37c-e2ff-463d-a77e-4f68bf3e8933\scratch")
from wear_features_diagnostic import extract_wear_specific_features
from phase2_experiment import PATTERN_MAP, WEAR_FEATURE_NAMES


def compute_adaptive_canny_thresholds(gray: np.ndarray):
    """
    Computes image-dependent adaptive Canny thresholds based on the
    Otsu partitioning of the gradient magnitude distribution.
    """
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    max_mag = np.max(mag) if np.max(mag) > 0 else 1.0

    mag_u8 = np.clip(mag / max_mag * 255.0, 0, 255).astype(np.uint8)
    otsu_th, _ = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    th_high = (otsu_th / 255.0) * max_mag
    th_low = 0.40 * th_high

    lower = int(np.clip(th_low, 15, 120))
    upper = int(np.clip(th_high, 40, 220))
    return lower, upper


def process_with_canny_mode(image_input, mode="fixed"):
    """
    Executes standard pipeline with either 'fixed' (50, 150)
    or 'adaptive' Canny edge detection.
    """
    is_valid, err_msg, cv_img = validate_image(image_input)
    if not is_valid:
        raise ValueError(err_msg)

    resized = resize_image(cv_img, target_size=(512, 512), keep_aspect_ratio=True)
    gray = to_grayscale(resized)
    denoised = reduce_noise(gray, method="bilateral")
    enhanced = enhance_contrast(denoised, method="clahe", clip_limit=2.0)

    tread_mask, segmented_tread, seg_meta = segment_tread_region(
        enhanced, method="otsu"
    )
    morph_cleaned = apply_morphology(
        tread_mask, operation="closing", kernel_shape="rect", kernel_size=(3, 3), iterations=1
    )

    if mode == "fixed":
        low_th, high_th = 50, 150
    elif mode == "adaptive":
        low_th, high_th = compute_adaptive_canny_thresholds(enhanced)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    canny_edges = cv2.Canny(enhanced, low_th, high_th)
    tread_edges = cv2.bitwise_and(canny_edges, canny_edges, mask=morph_cleaned)

    # 25-D Feature extraction
    features_dict, vec_25d = FeatureFusion.extract_all(
        enhanced_gray=enhanced,
        edge_map=tread_edges,
        binary_mask=morph_cleaned
    )

    # 6 Wear-specific features
    wear_dict = extract_wear_specific_features(cv_img)
    vec_wear_6d = np.array([wear_dict[k] for k in WEAR_FEATURE_NAMES], dtype=np.float32)

    vec_31d = np.concatenate([vec_25d, vec_wear_6d])

    return {
        "enhanced": enhanced,
        "mask": morph_cleaned,
        "canny_raw": canny_edges,
        "tread_edges": tread_edges,
        "thresholds": (low_th, high_th),
        "vec_25d": vec_25d,
        "vec_wear_6d": vec_wear_6d,
        "vec_31d": vec_31d,
        "features": features_dict,
        "wear_dict": wear_dict,
        "edge_density": float(features_dict["edge"]["total_edge_density"]),
        "horiz_edge_density": float(features_dict["edge"]["horizontal_edge_density"]),
        "vert_edge_density": float(features_dict["edge"]["vertical_edge_density"]),
        "glcm_contrast": float(features_dict["texture"]["contrast"]),
        "glcm_homogeneity": float(features_dict["texture"]["homogeneity"]),
    }


def run_phase3():
    print("=" * 80)
    print("PHASE 3: IMAGE-PROCESSING ROBUSTNESS (ADAPTIVE CANNY EXPERIMENT)")
    print("=" * 80)

    ds = WearDataset()
    records = ds.scan_dataset()

    label_map = {"good": 0, "moderate": 1, "worn": 2}

    fixed_data = []
    adapt_data = []

    for r in records:
        fpath = r["file_path"]
        tid = r["tyre_id"]
        lbl = r["label"]
        y_val = label_map[lbl]
        pat = PATTERN_MAP[tid]

        res_fixed = process_with_canny_mode(fpath, mode="fixed")
        res_adapt = process_with_canny_mode(fpath, mode="adaptive")

        base_item = {
            "file_path": fpath,
            "filename": Path(fpath).name,
            "tyre_id": tid,
            "label": lbl,
            "y": y_val,
            "pattern": pat,
        }

        f_item = dict(base_item)
        f_item.update(res_fixed)
        fixed_data.append(f_item)

        a_item = dict(base_item)
        a_item.update(res_adapt)
        adapt_data.append(a_item)

    df_fixed = pd.DataFrame(fixed_data)
    df_adapt = pd.DataFrame(adapt_data)

    y_all = np.array([r["y"] for r in fixed_data])
    groups_all = np.array([r["tyre_id"] for r in fixed_data])

    sgkf_split = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf_split.split(fixed_data, y_all, groups=groups_all))

    y_train, y_test = y_all[train_idx], y_all[test_idx]
    g_train, g_test = groups_all[train_idx], groups_all[test_idx]
    test_files = [fixed_data[i]["filename"] for i in test_idx]

    print(f"\nTrain Physical Tyres: {sorted(list(set(g_train)))}")
    print(f"Test Physical Tyres:  {sorted(list(set(g_test)))}")

    # 1. Edge Density & Threshold Distribution
    print("\n" + "=" * 80)
    print("EDGE DENSITY & THRESHOLD DISTRIBUTIONS BY WEAR CLASS")
    print("=" * 80)

    for lbl in ["good", "moderate", "worn"]:
        sub_f = df_fixed[df_fixed["label"] == lbl]
        sub_a = df_adapt[df_adapt["label"] == lbl]
        
        f_raw_edges = [np.count_nonzero(x) for x in sub_f["canny_raw"]]
        a_raw_edges = [np.count_nonzero(x) for x in sub_a["canny_raw"]]
        f_tread_edges = [np.count_nonzero(x) for x in sub_f["tread_edges"]]
        a_tread_edges = [np.count_nonzero(x) for x in sub_a["tread_edges"]]
        
        a_low_th = [th[0] for th in sub_a["thresholds"]]
        a_high_th = [th[1] for th in sub_a["thresholds"]]

        print(f"\nClass: {lbl.upper()}")
        print(f"  Fixed (50, 150)  : Raw Edges={np.mean(f_raw_edges):.1f} +/- {np.std(f_raw_edges):.1f} | Tread Edges={np.mean(f_tread_edges):.1f} +/- {np.std(f_tread_edges):.1f} | Edge Density={sub_f['edge_density'].mean():.4f} +/- {sub_f['edge_density'].std():.4f}")
        print(f"  Adaptive (Auto)  : Raw Edges={np.mean(a_raw_edges):.1f} +/- {np.std(a_raw_edges):.1f} | Tread Edges={np.mean(a_tread_edges):.1f} +/- {np.std(a_tread_edges):.1f} | Edge Density={sub_a['edge_density'].mean():.4f} +/- {sub_a['edge_density'].std():.4f}")
        print(f"  Adaptive Thresh  : Low Thresh={np.mean(a_low_th):.1f} +/- {np.std(a_low_th):.1f} | High Thresh={np.mean(a_high_th):.1f} +/- {np.std(a_high_th):.1f}")

    # 2. Monotonicity Across Matched Pattern Families
    print("\n" + "=" * 80)
    print("MONOTONICITY ACROSS MATCHED PATTERN FAMILIES (TREAD CANNY EDGES)")
    print("=" * 80)
    for p in ["Primacy 4", "PremiumContact 6", "Turanza T005", "EfficientGrip"]:
        print(f"\nPattern Family: {p}")
        for lbl in ["good", "moderate", "worn"]:
            sf = df_fixed[(df_fixed["pattern"] == p) & (df_fixed["label"] == lbl)]
            sa = df_adapt[(df_adapt["pattern"] == p) & (df_adapt["label"] == lbl)]
            f_edges = np.mean([np.count_nonzero(x) for x in sf["tread_edges"]])
            a_edges = np.mean([np.count_nonzero(x) for x in sa["tread_edges"]])
            f_dens = sf["edge_density"].mean()
            a_dens = sa["edge_density"].mean()
            tid = sf["tyre_id"].iloc[0]
            print(f"  {lbl.upper():<8} ({tid}): Fixed Edges={f_edges:5.0f} (dens={f_dens:.4f}) | Adapt Edges={a_edges:5.0f} (dens={a_dens:.4f})")

    # 3. Model Performance Comparison: Fixed vs Adaptive (on 31-D and 25-D)
    print("\n" + "=" * 80)
    print("MODEL EVALUATION: FIXED CANNY VS ADAPTIVE CANNY")
    print("=" * 80)

    eval_results = {}
    for feature_set in ["31-D Augmented", "25-D Baseline"]:
        print(f"\n--- Feature Architecture: {feature_set} ---")
        eval_results[feature_set] = {}

        for mode_name, df_mode in [("Fixed Canny", df_fixed), ("Adaptive Canny", df_adapt)]:
            vec_col = "vec_31d" if "31-D" in feature_set else "vec_25d"
            X_tr = np.stack(df_mode.iloc[train_idx][vec_col].values)
            X_te = np.stack(df_mode.iloc[test_idx][vec_col].values)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            rf = get_model("rf")
            rf.fit(X_tr_s, y_train)

            y_pred = rf.predict(X_te_s)
            acc = float(accuracy_score(y_test, y_pred))
            f1_m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
            rep = classification_report(y_test, y_pred, target_names=WEAR_CLASSES, output_dict=True, zero_division=0)

            # 3-fold CV
            sgkf_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
            cv_accs, cv_f1s = [], []
            for f_tr, f_val in sgkf_cv.split(X_tr_s, y_train, groups=g_train):
                f_m = get_model("rf")
                f_m.fit(X_tr_s[f_tr], y_train[f_tr])
                pred_v = f_m.predict(X_tr_s[f_val])
                cv_accs.append(float(accuracy_score(y_train[f_val], pred_v)))
                cv_f1s.append(float(f1_score(y_train[f_val], pred_v, average="macro", zero_division=0)))

            eval_results[feature_set][mode_name] = {
                "holdout_acc": acc,
                "macro_f1": f1_m,
                "cv_acc_mean": float(np.mean(cv_accs)),
                "cv_acc_std": float(np.std(cv_accs)),
                "cv_f1_mean": float(np.mean(cv_f1s)),
                "cv_f1_std": float(np.std(cv_f1s)),
                "confusion_matrix": cm.tolist(),
                "report": rep
            }

            print(f"Mode: {mode_name:<16} | Holdout Acc: {acc:.4f} | Holdout F1: {f1_m:.4f} | CV Acc: {np.mean(cv_accs):.4f} +/- {np.std(cv_accs):.4f} | CV F1: {np.mean(cv_f1s):.4f}")

    # 4. Perturbation & Preprocessing Stability Test
    print("\n" + "=" * 80)
    print("PREPROCESSING ROBUSTNESS UNDER PERTURBATION (NOISE, BLUR, CONTRAST)")
    print("=" * 80)

    # Pick 3 test images across classes
    perturb_results = []
    for cls in ["good", "moderate", "worn"]:
        fpath = str(Path(f"data/raw/tyre_wear/{cls}/tyre_{cls[0]}01_view1.jpg"))
        raw_bgr = cv2.imread(fpath)
        
        # Define 5 stress conditions
        conditions = {
            "Nominal": raw_bgr,
            "Darkened (-25)": np.clip(raw_bgr.astype(int) - 25, 0, 255).astype(np.uint8),
            "Brightened (+25)": np.clip(raw_bgr.astype(int) + 25, 0, 255).astype(np.uint8),
            "Low Contrast (0.75x)": np.clip(raw_bgr.astype(float) * 0.75 + 30, 0, 255).astype(np.uint8),
            "Sensor Noise (sigma=5)": np.clip(raw_bgr.astype(float) + np.random.normal(0, 5, raw_bgr.shape), 0, 255).astype(np.uint8),
            "Mild Blur (k=5, s=1.1)": cv2.GaussianBlur(raw_bgr, (5, 5), 1.1)
        }

        print(f"\nStress Test on Physical Tyre {cls[0]}01 ({cls.upper()}):")
        print(f"  {'Condition':<24} | {'Fixed Canny Edges':<18} | {'Adaptive Canny Edges':<20} | {'Adaptive Thresh':<16}")
        print("  " + "-" * 85)

        for cond_name, c_img in conditions.items():
            rf = process_with_canny_mode(c_img, mode="fixed")
            ra = process_with_canny_mode(c_img, mode="adaptive")
            f_cnt = np.count_nonzero(rf["tread_edges"])
            a_cnt = np.count_nonzero(ra["tread_edges"])
            th_str = f"({ra['thresholds'][0]}, {ra['thresholds'][1]})"
            print(f"  {cond_name:<24} | {f_cnt:<18} | {a_cnt:<20} | {th_str:<16}")

    # 5. Held-Out tyre_g04 Predictions under Fixed vs Adaptive
    print("\n" + "=" * 80)
    print("HELD-OUT PREDICTIONS AUDIT: TYRE_G04 (FIXED VS ADAPTIVE 31-D)")
    print("=" * 80)
    for mode_name, df_mode in [("Fixed Canny", df_fixed), ("Adaptive Canny", df_adapt)]:
        X_tr = np.stack(df_mode.iloc[train_idx]["vec_31d"].values)
        X_te = np.stack(df_mode.iloc[test_idx]["vec_31d"].values)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        rf = get_model("rf")
        rf.fit(X_tr_s, y_train)
        preds = rf.predict(X_te_s)
        probs = rf.predict_proba(X_te_s)

        print(f"\nMode: {mode_name}")
        for i, idx in enumerate(test_idx):
            fn = df_mode.iloc[idx]["filename"]
            tid = df_mode.iloc[idx]["tyre_id"]
            true_lbl = WEAR_CLASSES[y_test[i]]
            pred_lbl = WEAR_CLASSES[preds[i]]
            conf = float(np.max(probs[i]))
            print(f"  {fn:<22} ({tid}, True: {true_lbl.upper():<8}) -> Pred: {pred_lbl.upper():<8} (conf={conf:.2f})")

    # Save Phase 3 artifacts
    p3_dir = Path(r"C:\Users\puru\.gemini\antigravity\brain\2ab4a37c-e2ff-463d-a77e-4f68bf3e8933\scratch\phase3_artifacts")
    p3_dir.mkdir(parents=True, exist_ok=True)
    with open(p3_dir / "phase3_results.json", "w") as fp:
        json.dump(eval_results, fp, indent=2)
    print(f"\n[+] Phase 3 experimental results saved to {p3_dir / 'phase3_results.json'}")

if __name__ == "__main__":
    run_phase3()
