"""
TreadVision: Shape Descriptors, Contour Geometry, and Hu Moments
"""
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np


def extract_contours(binary_mask: np.ndarray, min_area: int = 50) -> List[np.ndarray]:
    """
    Find external and internal contours from binary tread mask.
    Filters out noise contours smaller than min_area.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    return filtered


def compute_shape_features(binary_mask: np.ndarray) -> Dict[str, Any]:
    """
    Compute rigorous geometric and shape features from segmented tread regions:
    - area: total pixel area of tread elements
    - perimeter: cumulative boundary length
    - aspect_ratio: width / height of dominant tread bounding box
    - extent: contour area / bounding rect area
    - solidity: contour area / convex hull area
    - circularity: 4 * pi * area / (perimeter^2)
    - equivalent_diameter: sqrt(4 * area / pi)
    - hu_moments: 7 rotation, translation, and scale invariant moments
    - block_count: number of discrete tread blocks
    """
    h, w = binary_mask.shape[:2]
    total_image_area = float(h * w)

    contours = extract_contours(binary_mask, min_area=40)

    if not contours:
        return {
            "area_ratio": 0.0,
            "perimeter_ratio": 0.0,
            "aspect_ratio": 1.0,
            "extent": 0.0,
            "solidity": 0.0,
            "circularity": 0.0,
            "equivalent_diameter_ratio": 0.0,
            "block_count": 0,
            "hu_moments": [0.0] * 7
        }

    # Combine all contours or take dominant tread structures
    total_area = sum(cv2.contourArea(c) for c in contours)
    total_perimeter = sum(cv2.arcLength(c, True) for c in contours)

    # Largest contour properties
    largest_cnt = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(largest_cnt)
    aspect_ratio = float(bw) / float(bh) if bh > 0 else 1.0

    rect_area = bw * bh
    largest_area = cv2.contourArea(largest_cnt)
    extent = float(largest_area / rect_area) if rect_area > 0 else 0.0

    hull = cv2.convexHull(largest_cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(largest_area / hull_area) if hull_area > 0 else 0.0

    # Circularity of dominant tread elements
    if total_perimeter > 0:
        circularity = float(4.0 * np.pi * total_area / (total_perimeter ** 2))
    else:
        circularity = 0.0

    # Equivalent diameter
    equiv_diameter = np.sqrt(4.0 * total_area / np.pi) if total_area > 0 else 0.0

    # Hu Moments (Log scale transformation for stability)
    moments = cv2.moments(binary_mask)
    raw_hu = cv2.HuMoments(moments).flatten()
    log_hu = []
    for h_val in raw_hu:
        if abs(h_val) > 1e-15:
            # -sign(h) * log10(abs(h))
            log_hu.append(round(float(-1.0 * np.copysign(1.0, h_val) * np.log10(abs(h_val))), 4))
        else:
            log_hu.append(0.0)

    return {
        "area_ratio": round(float(total_area / total_image_area), 4),
        "perimeter_ratio": round(float(total_perimeter / (2 * (h + w))), 4),
        "aspect_ratio": round(aspect_ratio, 3),
        "extent": round(extent, 4),
        "solidity": round(solidity, 4),
        "circularity": round(circularity, 4),
        "equivalent_diameter_ratio": round(float(equiv_diameter / np.sqrt(total_image_area)), 4),
        "block_count": int(len(contours)),
        "hu_moments": log_hu
    }
