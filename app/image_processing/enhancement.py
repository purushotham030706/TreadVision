"""
TreadVision: Contrast Enhancement (Histogram Equalization & CLAHE)
"""
from typing import Tuple
import cv2
import numpy as np


def histogram_equalization(gray_image: np.ndarray) -> np.ndarray:
    """
    Standard global histogram equalization for contrast enhancement.
    Spreads the most frequent intensity values across the entire 0-255 range.
    """
    return cv2.equalizeHist(gray_image)


def clahe(
    gray_image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Divides the image into small contextual regions (tiles), computes individual histograms,
    clips amplification to avoid noise amplification, and applies bilinear interpolation.
    Essential for uneven illumination on curved tyre surfaces and forensic impressions.
    """
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe_obj.apply(gray_image)


def enhance_contrast(
    gray_image: np.ndarray,
    method: str = "clahe",
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Unified contrast enhancement dispatcher.
    """
    if method.lower() == "clahe":
        return clahe(gray_image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    elif method.lower() in ("histeq", "global"):
        return histogram_equalization(gray_image)
    return clahe(gray_image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
