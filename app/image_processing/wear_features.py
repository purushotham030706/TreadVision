"""
TreadVision: Wear-Specific Local Feature Extraction
Extracts 6 wear-specific local descriptors designed to capture physical tread relief
independently of macro pattern geometry:
  1. groove_to_land_contrast: Michelson contrast between longitudinal groove valleys and land crests.
  2. groove_visibility_ratio: Valley prominence normalized by the profile dynamic range.
  3. sipe_visibility_freq: Density of high-frequency incisions located strictly within land regions.
  4. local_groove_continuity: Maximum unbroken vertical continuity of detected groove channels.
  5. local_texture_contrast: Mean local standard deviation within tread-block regions (32x32 blocks).
  6. groove_wall_gradient: Mean boundary gradient slope at the transition between grooves and lands.
"""
from typing import Dict, Any, Union
import numpy as np
import cv2


WEAR_FEATURE_NAMES = [
    "groove_to_land_contrast",
    "groove_visibility_ratio",
    "sipe_visibility_freq",
    "local_groove_continuity",
    "local_texture_contrast",
    "groove_wall_gradient"
]


def extract_wear_specific_features(image_input: np.ndarray) -> Dict[str, float]:
    """
    Extract 6 wear-specific local features from a grayscale or BGR image.
    Ensures robust handling of flat/uniform images, division-by-zero guards,
    and returns bounded floating-point descriptors.
    """
    if len(image_input.shape) == 3:
        gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_input.copy()

    # Standardize processing resolution to 512x512
    if gray.shape[:2] != (512, 512):
        gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)

    # 1. Column-wise intensity projection (longitudinal profile across ribs/grooves)
    col_proj = np.mean(gray, axis=0)  # shape: (512,)

    # Land intensity (upper 85th percentile of profile)
    land_intensity = float(np.percentile(col_proj, 85))
    # Groove intensity (lower 15th percentile of profile)
    groove_intensity = float(np.percentile(col_proj, 15))

    # 1. Groove-to-land intensity contrast (Michelson metric: (I_land - I_groove)/(I_land + I_groove))
    denom = land_intensity + groove_intensity
    contrast_gl = float((land_intensity - groove_intensity) / (denom if denom > 1e-4 else 1.0))
    contrast_gl = max(0.0, min(1.0, contrast_gl))

    # 2. Groove visibility ratio: relative prominence of valley depression
    p95 = float(np.percentile(col_proj, 95))
    p05 = float(np.percentile(col_proj, 5))
    dyn_range = p95 - p05
    groove_vis_ratio = float((land_intensity - groove_intensity) / (dyn_range if dyn_range > 1.0 else 1.0))
    groove_vis_ratio = max(0.0, min(2.0, groove_vis_ratio))

    # Identify groove columns vs land columns
    groove_thresh = groove_intensity + 0.35 * (land_intensity - groove_intensity)
    groove_cols = col_proj < groove_thresh

    land_mask = np.ones((512, 512), dtype=np.uint8)
    land_mask[:, groove_cols] = 0

    # 3. Sipe visibility frequency: High-frequency lateral/angled transitions within land regions
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    sipe_edge_mask = (sobel_y > 14.0).astype(np.uint8)
    sipe_in_land = cv2.bitwise_and(sipe_edge_mask, sipe_edge_mask, mask=land_mask)
    land_pixel_count = int(np.sum(land_mask))
    sipe_freq = float(np.sum(sipe_in_land) / (land_pixel_count if land_pixel_count > 0 else 1))
    sipe_freq = max(0.0, min(1.0, sipe_freq))

    # 4. Local groove continuity: measure unbroken vertical path of groove corridors
    groove_pixel_mask = (gray < groove_thresh).astype(np.uint8)
    col_sums = np.sum(groove_pixel_mask, axis=0)  # count of groove pixels per column
    continuity = float(np.max(col_sums) / 512.0) if len(col_sums) > 0 else 0.0
    continuity = max(0.0, min(1.0, continuity))

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
        "groove_wall_gradient": round(wall_slope, 4)
    }
