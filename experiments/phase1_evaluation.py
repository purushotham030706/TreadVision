"""
TreadVision: Phase 1 Evaluation Script
Computes both 25-D pipeline features and 6 wear-specific local features
on the newly regenerated dataset across all 36 images.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, r"c:\Users\puru\Desktop\IP-Event")
from app.config import WEAR_DATA_DIR, WEAR_CLASSES
from app.ml.dataset import WearDataset
from app.image_processing.pipeline import ImageProcessingPipeline

sys.path.insert(0, r"C:\Users\puru\.gemini\antigravity\brain\2ab4a37c-e2ff-463d-a77e-4f68bf3e8933\scratch")
from wear_features_diagnostic import extract_wear_specific_features

def run_phase1_audit():
    pipe = ImageProcessingPipeline()
    ds = WearDataset()
    records = ds.scan_dataset()

    pattern_map = {
        "tyre_g01": "Primacy 4", "tyre_g02": "PremiumContact 6", "tyre_g03": "Turanza T005", "tyre_g04": "EfficientGrip",
        "tyre_m01": "Primacy 4", "tyre_m02": "PremiumContact 6", "tyre_m03": "Turanza T005", "tyre_m04": "EfficientGrip",
        "tyre_w01": "Primacy 4", "tyre_w02": "PremiumContact 6", "tyre_w03": "Turanza T005", "tyre_w04": "EfficientGrip"
    }

    rows = []
    for r in records:
        fpath = r["file_path"]
        t_id = r["tyre_id"]
        label = r["label"]
        bgr = cv2.imread(fpath)

        # 1. Full pipeline processing (25-D features + stages)
        pipe_res = pipe.process(bgr)
        canny_edges_count = np.count_nonzero(pipe_res.stages["Canny Edges"])
        enhanced = pipe_res.stages["Enhanced (CLAHE)"]
        canny_raw_count = np.count_nonzero(cv2.Canny(enhanced, 50, 150))

        # 2. Six wear-specific diagnostic features
        wear_feats = extract_wear_specific_features(bgr)

        row = {
            "tyre_id": t_id,
            "label": label,
            "pattern": pattern_map[t_id],
            "canny_edges": canny_edges_count,
            "canny_raw": canny_raw_count,
            "total_edge_density": pipe_res.features["edge"]["total_edge_density"],
            "horizontal_edge_density": pipe_res.features["edge"]["horizontal_edge_density"],
            "vertical_edge_density": pipe_res.features["edge"]["vertical_edge_density"],
            "glcm_contrast": pipe_res.features["texture"]["contrast"],
            "glcm_homogeneity": pipe_res.features["texture"]["homogeneity"],
            "glcm_energy": pipe_res.features["texture"]["energy"],
            "shape_solidity": pipe_res.features["shape"]["solidity"],
            "coverage_pct": pipe_res.metrics["segmentation_coverage_pct"],
        }
        row.update(wear_feats)
        rows.append(row)

    df = pd.DataFrame(rows)

    print("=" * 80)
    print("PHASE 1: REGENERATED DATASET EVALUATION")
    print("=" * 80)

    print("\n[1] CANNY EDGE DETECTION & EXTINCTION CHECK")
    print("-" * 80)
    for lbl in ["good", "moderate", "worn"]:
        sub = df[df["label"] == lbl]
        print(f"Class: {lbl.upper():<8} | Raw Canny: {sub['canny_raw'].mean():.1f} +/- {sub['canny_raw'].std():.1f} (min={sub['canny_raw'].min()}, max={sub['canny_raw'].max()}) | Tread Canny: {sub['canny_edges'].mean():.1f} +/- {sub['canny_edges'].std():.1f} | Zero Edge Count: {(sub['canny_edges'] == 0).sum()}/12")

    print("\n[2] EXISTING 25-D PIPELINE KEY FEATURES (MEAN +/- STD)")
    print("-" * 80)
    pipe_cols = ["total_edge_density", "glcm_contrast", "glcm_homogeneity", "coverage_pct"]
    for c in pipe_cols:
        g = df[df["label"] == "good"][c]
        m = df[df["label"] == "moderate"][c]
        w = df[df["label"] == "worn"][c]
        print(f"{c:<24} | Good: {g.mean():.4f} +/- {g.std():.4f} | Mod: {m.mean():.4f} +/- {m.std():.4f} | Worn: {w.mean():.4f} +/- {w.std():.4f}")

    print("\n[3] SIX WEAR-SPECIFIC LOCAL FEATURES (MEAN +/- STD)")
    print("-" * 80)
    wear_cols = [
        "groove_to_land_contrast",
        "groove_visibility_ratio",
        "sipe_visibility_freq",
        "local_groove_continuity",
        "local_texture_contrast",
        "groove_wall_gradient"
    ]
    for c in wear_cols:
        g = df[df["label"] == "good"][c]
        m = df[df["label"] == "moderate"][c]
        w = df[df["label"] == "worn"][c]
        print(f"{c:<26} | Good: {g.mean():.4f} +/- {g.std():.4f} | Mod: {m.mean():.4f} +/- {m.std():.4f} | Worn: {w.mean():.4f} +/- {w.std():.4f}")

    print("\n[4] SEPARATION METRICS (COHEN'S D)")
    print("-" * 80)
    def cohen_d(x1, x2):
        s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
        pooled = np.sqrt((s1**2 + s2**2) / 2.0)
        return abs(np.mean(x1) - np.mean(x2)) / (pooled if pooled > 0 else 1.0)

    for c in wear_cols + pipe_cols:
        g = df[df["label"] == "good"][c].values
        m = df[df["label"] == "moderate"][c].values
        w = df[df["label"] == "worn"][c].values
        d_gm = cohen_d(g, m)
        d_mw = cohen_d(m, w)
        d_gw = cohen_d(g, w)
        print(f"{c:<26} | Good vs Mod d: {d_gm:.3f} | Mod vs Worn d: {d_mw:.3f} | Good vs Worn d: {d_gw:.3f}")

    print("\n[5] MATCHED TREAD-PATTERN PROGRESSIONS ACROSS WEAR LEVELS")
    print("-" * 80)
    for p in ["Primacy 4", "PremiumContact 6", "Turanza T005", "EfficientGrip"]:
        print(f"\nPattern Family: {p}")
        sub_p = df[df["pattern"] == p]
        for lbl in ["good", "moderate", "worn"]:
            sub_lbl = sub_p[sub_p["label"] == lbl]
            tid = sub_lbl["tyre_id"].iloc[0]
            print(f"  {lbl.upper():<8} ({tid}): RawCanny={sub_lbl['canny_raw'].mean():.0f}, TreadCanny={sub_lbl['canny_edges'].mean():.0f}, GLCM_Cont={sub_lbl['glcm_contrast'].mean():.2f}, WearContrast={sub_lbl['groove_to_land_contrast'].mean():.4f}, WearTexture={sub_lbl['local_texture_contrast'].mean():.2f}")

    print("\n[6] PER-PHYSICAL-TYRE SUMMARY")
    print("-" * 80)
    tyre_summary = df.groupby(["label", "tyre_id"])[["canny_raw", "canny_edges", "total_edge_density", "glcm_contrast", "local_texture_contrast", "groove_to_land_contrast"]].mean()
    print(tyre_summary.to_string())

if __name__ == "__main__":
    run_phase1_audit()
