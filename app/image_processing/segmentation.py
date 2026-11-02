"""
TreadVision: Tread Region Segmentation and Thresholding
"""
from typing import Tuple, Optional
import cv2
import numpy as np


def otsu_threshold(enhanced_gray: np.ndarray, inverse: bool = True) -> Tuple[float, np.ndarray]:
    """
    Otsu's thresholding automatically computes the optimal global threshold
    by minimizing weighted within-class variance of two intensity classes.
    """
    flag = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    threshold_val, binary = cv2.threshold(enhanced_gray, 0, 255, flag + cv2.THRESH_OTSU)
    return float(threshold_val), binary


def adaptive_threshold(
    enhanced_gray: np.ndarray,
    method: str = "gaussian",
    block_size: int = 21,
    c: int = 5,
    inverse: bool = True
) -> np.ndarray:
    """
    Adaptive thresholding calculates thresholds for small regions,
    handling harsh shadows, mud gradations, or illumination gradients.
    """
    if block_size % 2 == 0:
        block_size += 1
    adaptive_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    thresh_type = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(enhanced_gray, 255, adaptive_type, thresh_type, block_size, c)


def segment_tread_region(
    enhanced_gray: np.ndarray,
    method: str = "otsu",
    fallback_roi: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Segment the primary tread pattern region from background/sidewall/rim.
    Returns:
        (tread_mask, segmented_tread, metadata_dict)
    """
    h, w = enhanced_gray.shape[:2]

    # Generate initial binary segmentation mask
    if method == "adaptive":
        binary = adaptive_threshold(enhanced_gray, method="gaussian", block_size=25, c=4)
        thresh_info = "adaptive_gaussian"
    else:
        thresh_val, binary = otsu_threshold(enhanced_gray, inverse=True)
        thresh_info = f"otsu_{thresh_val:.1f}"

    # Connected component analysis to filter noise and isolate tread blocks
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Filter components by area (ignore tiny specks < 0.05% of image area)
    min_area = int(0.0005 * (h * w))
    clean_mask = np.zeros_like(binary)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == i] = 255

    # Check if segmentation successfully captured tread features or failed
    coverage = np.sum(clean_mask > 0) / (h * w)

    # If segmentation is too sparse (< 5%) or too dense (> 95%), apply fallback ROI
    if coverage < 0.05 or coverage > 0.95:
        if fallback_roi is not None:
            rx, ry, rw, rh = fallback_roi
            clean_mask = np.zeros_like(binary)
            clean_mask[ry:ry+rh, rx:rx+rw] = 255
            thresh_info += "_fallback_custom_roi"
        else:
            # Default central 70% rectangular ROI typical of tyre tread photograph
            pad_h, pad_w = int(h * 0.15), int(w * 0.15)
            clean_mask = np.zeros_like(binary)
            clean_mask[pad_h:h-pad_h, pad_w:w-pad_w] = binary[pad_h:h-pad_h, pad_w:w-pad_w]
            thresh_info += "_fallback_central_roi"

    # Segmented output: enhanced tread isolated by mask
    segmented_tread = cv2.bitwise_and(enhanced_gray, enhanced_gray, mask=clean_mask)

    meta = {
        "method": thresh_info,
        "coverage_ratio": float(np.sum(clean_mask > 0) / (h * w)),
        "connected_components_count": int(num_labels - 1)
    }

    return clean_mask, segmented_tread, meta
