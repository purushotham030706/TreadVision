"""
TreadVision Diagnostic: Wear-Specific Local Feature Extraction and Evaluation
Investigates 6 features designed to capture wear relief independently of pattern geometry.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, r"c:\Users\puru\Desktop\IP-Event")

from app.config import WEAR_DATA_DIR, WEAR_CLASSES
from app.ml.dataset import WearDataset
from app.image_processing.pipeline import ImageProcessingPipeline


def extract_wear_specific_features(gray_img: np.ndarray) -> dict:
    """
    Extract 6 wear-specific local features designed to be less dependent on
    macro pattern geometry:
      1. groove_to_land_contrast: Michelson contrast between detected longitudinal grooves and land crests
      2. groove_visibility_ratio: Groove depth prominence normalized by dynamic range
      3. sipe_visibility_frequency: Density of fine high-frequency incisions in land regions
      4. local_groove_continuity: Normalized vertical continuity of detected groove channels
      5. local_texture_contrast: Mean local standard deviation within tread-block regions
      6. groove_wall_gradient: Mean boundary gradient slope at groove-land transitions
    """
    if len(gray_img.shape) == 3:
        gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_img.copy()

    gray = cv2.resize(gray, (512, 512))

    # 1. Column-wise intensity projection (longitudinal profile across ribs/grooves)
    col_proj = np.mean(gray, axis=0)  # shape: (512,)

    # Land intensity (upper 85th percentile of profile)
    land_intensity = float(np.percentile(col_proj, 85))
    # Groove intensity (lower 15th percentile of profile)
    groove_intensity = float(np.percentile(col_proj, 15))

    # 1. Groove-to-land intensity contrast (Michelson metric: (I_land - I_groove)/(I_land + I_groove))
    denom = land_intensity + groove_intensity
    contrast_gl = float((land_intensity - groove_intensity) / (denom if denom > 0 else 1.0))

    # 2. Groove visibility ratio: relative prominence of valley depression
    p95 = float(np.percentile(col_proj, 95))
    p05 = float(np.percentile(col_proj, 5))
    dyn_range = p95 - p05
    groove_vis_ratio = float((land_intensity - groove_intensity) / (dyn_range if dyn_range > 1.0 else 1.0))

    # Identify groove columns vs land columns
    groove_thresh = groove_intensity + 0.35 * (land_intensity - groove_intensity)
    groove_cols = col_proj < groove_thresh

    land_mask = np.ones((512, 512), dtype=np.uint8)
    land_mask[:, groove_cols] = 0

    # 3. Sipe visibility frequency: High-frequency lateral/angled transitions within land regions
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    # Adaptive threshold: values above noise floor (typically > 12)
    sipe_edge_mask = (sobel_y > 14.0).astype(np.uint8)
    sipe_in_land = cv2.bitwise_and(sipe_edge_mask, sipe_edge_mask, mask=land_mask)
    land_pixel_count = int(np.sum(land_mask))
    sipe_freq = float(np.sum(sipe_in_land) / (land_pixel_count if land_pixel_count > 0 else 1))

    # 4. Local groove continuity: measure unbroken vertical path of groove corridors
    groove_pixel_mask = (gray < groove_thresh).astype(np.uint8)
    col_sums = np.sum(groove_pixel_mask, axis=0)  # count of groove pixels per column
    continuity = float(np.max(col_sums) / 512.0) if len(col_sums) > 0 else 0.0

    # 5. Local texture contrast: average standard deviation within small local land blocks (32x32)
    block_size = 32
    stds = []
    for y in range(0, 512, block_size):
        for x in range(0, 512, block_size):
            if np.mean(land_mask[y:y+block_size, x:x+block_size]) > 0.65:
                blk = gray[y:y+block_size, x:x+block_size]
                stds.append(float(np.std(blk)))
    local_tex_contrast = float(np.mean(stds)) if stds else 0.0

    # 6. Groove wall gradient: lateral gradient slope at the transition boundary between groove and land
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    diff_cols = np.abs(np.diff(groove_cols.astype(int)))
    boundary_cols = np.where(diff_cols > 0)[0]
    wall_grads = []
    for c in boundary_cols:
        wall_grads.append(float(np.mean(sobel_x[:, c])))
    wall_slope = float(np.mean(wall_grads)) if wall_grads else 0.0

    return {
        "groove_to_land_contrast": round(contrast_gl, 4),
        "groove_visibility_ratio": round(groove_vis_ratio, 4),
        "sipe_visibility_freq": round(sipe_freq, 4),
        "local_groove_continuity": round(continuity, 4),
        "local_texture_contrast": round(local_tex_contrast, 4),
        "groove_wall_gradient": round(wall_slope, 4),
    }


def run_evaluation():
    print("=" * 80)
    print("TREADVISION: WEAR-SPECIFIC LOCAL FEATURE EVALUATION")
    print("=" * 80)

    ds = WearDataset()
    records = ds.scan_dataset()

    data_rows = []
    for r in records:
        fpath = r["file_path"]
        t_id = r["tyre_id"]
        label = r["label"]

        bgr = cv2.imread(fpath)
        feats = extract_wear_specific_features(bgr)
        feats["file_path"] = fpath
        feats["tyre_id"] = t_id
        feats["label"] = label
        feats["view"] = Path(fpath).stem.split("_")[-1]
        data_rows.append(feats)

    df = pd.DataFrame(data_rows)
    feature_cols = [
        "groove_to_land_contrast",
        "groove_visibility_ratio",
        "sipe_visibility_freq",
        "local_groove_continuity",
        "local_texture_contrast",
        "groove_wall_gradient"
    ]

    print("\n[1] PER-CLASS MEANS AND STANDARD DEVIATIONS")
    print("-" * 80)
    for f in feature_cols:
        g = df[df["label"] == "good"][f]
        m = df[df["label"] == "moderate"][f]
        w = df[df["label"] == "worn"][f]
        print(f"{f:<26} | Good: {g.mean():.4f} +/- {g.std():.4f} | Mod: {m.mean():.4f} +/- {m.std():.4f} | Worn: {w.mean():.4f} +/- {w.std():.4f}")

    print("\n[2] SEPARATION METRICS (COHEN'S D)")
    print("-" * 80)
    def cohen_d(x1, x2):
        s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
        pooled = np.sqrt((s1**2 + s2**2) / 2.0)
        return abs(np.mean(x1) - np.mean(x2)) / (pooled if pooled > 0 else 1.0)

    for f in feature_cols:
        g = df[df["label"] == "good"][f].values
        m = df[df["label"] == "moderate"][f].values
        w = df[df["label"] == "worn"][f].values
        d_gm = cohen_d(g, m)
        d_mw = cohen_d(m, w)
        d_gw = cohen_d(g, w)
        print(f"{f:<26} | Good vs Mod d: {d_gm:.3f} | Mod vs Worn d: {d_mw:.3f} | Good vs Worn d: {d_gw:.3f}")

    print("\n[3] PER-PHYSICAL-TYRE AVERAGES (GROUPED BY PHYSICAL TYRE ID)")
    print("-" * 80)
    tyre_grp = df.groupby(["label", "tyre_id"])[feature_cols].mean()
    print(tyre_grp.to_string())

    print("\n[4] COMPARISON OF TYRE_G04 (HELD-OUT GOOD) VS GOOD TRAIN VS MODERATE")
    print("-" * 80)
    g_train = df[df["tyre_id"].isin(["tyre_g01", "tyre_g02", "tyre_g03"])][feature_cols].mean()
    g_test  = df[df["tyre_id"] == "tyre_g04"][feature_cols].mean()
    m_all   = df[df["label"] == "moderate"][feature_cols].mean()
    w_all   = df[df["label"] == "worn"][feature_cols].mean()

    comp_df = pd.DataFrame({
        "Good (Train g01-g03)": g_train,
        "Good (Held-out g04)": g_test,
        "Moderate (All)": m_all,
        "Worn (All)": w_all
    })
    print(comp_df.to_string())

    print("\n[5] INTER-PATTERN VS INTRA-PATTERN VARIANCE RATIO")
    print("-" * 80)
    print("Variance within class (due to pattern geometry) vs. Delta between classes (due to wear):")
    for f in feature_cols:
        good_vals = df[df["label"] == "good"]
        mod_vals = df[df["label"] == "moderate"]
        inter_pattern_std_good = good_vals.groupby("tyre_id")[f].mean().std()
        inter_pattern_std_mod = mod_vals.groupby("tyre_id")[f].mean().std()
        delta_good_to_mod = abs(good_vals[f].mean() - mod_vals[f].mean())
        ratio = inter_pattern_std_good / (delta_good_to_mod if delta_good_to_mod > 0 else 1e-4)
        print(f"{f:<26} | Inter-Pattern Std: {inter_pattern_std_good:.4f} | Delta(Good->Mod): {delta_good_to_mod:.4f} | Ratio: {ratio:.2f}")

    print("\n[6] MATCHED PATTERN TRAJECTORIES (WITHIN SAME TREAD-PATTERN FAMILY)")
    print("-" * 80)
    pattern_map = {
        "tyre_g01": "Primacy 4", "tyre_g02": "Pilot Sport 4", "tyre_g03": "Turanza T005", "tyre_g04": "EfficientGrip",
        "tyre_m01": "Turanza T005", "tyre_m02": "EfficientGrip", "tyre_m03": "Primacy 4", "tyre_m04": "Pilot Sport 4",
        "tyre_w01": "EfficientGrip", "tyre_w02": "Primacy 4", "tyre_w03": "Pilot Sport 4", "tyre_w04": "Turanza T005"
    }
    df["pattern"] = df["tyre_id"].map(pattern_map)

    for p in ["EfficientGrip", "Turanza T005", "Primacy 4", "Pilot Sport 4"]:
        print(f"\nPattern: {p}")
        for lbl in ["good", "moderate", "worn"]:
            sub = df[(df["pattern"] == p) & (df["label"] == lbl)]
            tid = sub["tyre_id"].iloc[0]
            m = sub[feature_cols].mean()
            print(f"  {lbl.upper():<8} ({tid}): contrast={m['groove_to_land_contrast']:.4f}, cont={m['local_groove_continuity']:.4f}, tex={m['local_texture_contrast']:.2f}, grad={m['groove_wall_gradient']:.2f}, sipe={m['sipe_visibility_freq']:.4f}, vis_ratio={m['groove_visibility_ratio']:.4f}")

    print("\n[7] GRADIENT DISTRIBUTIONS AND CANNY EXTINCTION")
    print("-" * 80)
    pipe = ImageProcessingPipeline()
    grad_data = []
    for r in records:
        fpath = r["file_path"]
        t_id = r["tyre_id"]
        label = r["label"]
        bgr = cv2.imread(fpath)
        pipe_res = pipe.process(bgr)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobel_x**2 + sobel_y**2)
        canny_edges = np.count_nonzero(pipe_res.stages["Canny Edges"])
        grad_data.append({
            "label": label,
            "tyre_id": t_id,
            "max_grad": np.max(mag),
            "p99_grad": np.percentile(mag, 99.0),
            "p999_grad": np.percentile(mag, 99.9),
            "canny_edges": canny_edges,
        })
    gdf = pd.DataFrame(grad_data)
    for lbl in ["good", "moderate", "worn"]:
        sub = gdf[gdf["label"] == lbl]
        print(f"Class: {lbl.upper():<8} | MaxGrad: {sub['max_grad'].mean():.1f} +/- {sub['max_grad'].std():.1f} | P99.9Grad: {sub['p999_grad'].mean():.1f} +/- {sub['p999_grad'].std():.1f} | CannyEdges: {sub['canny_edges'].mean():.1f} +/- {sub['canny_edges'].std():.1f}")


if __name__ == "__main__":
    run_evaluation()
