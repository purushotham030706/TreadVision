"""
TreadVision: Edge Detection and Edge Orientation Feature Extraction
"""
from typing import Tuple, Dict, Any
import cv2
import numpy as np


def detect_canny_edges(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    aperture_size: int = 3
) -> np.ndarray:
    """
    Canny Edge Detector with non-maximum suppression and hysteresis thresholding.
    Produces thin, well-localized edge contours defining tread grooves.
    """
    return cv2.Canny(image, low_threshold, high_threshold, apertureSize=aperture_size)


def detect_sobel_edges(image: np.ndarray, ksize: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sobel gradient operator computes first-order directional derivatives.
    Returns:
        (gradient_magnitude_8u, grad_x, grad_y)
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    mag_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return mag_normalized, grad_x, grad_y


def detect_laplacian_edges(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Laplacian second-order derivative operator.
    Captures zero-crossings for rapid intensity transitions.
    """
    lap = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
    return cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def extract_edge_features(
    edge_map: np.ndarray,
    grad_x: Optional[np.ndarray] = None,
    grad_y: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Extract quantifiable edge descriptors:
    - total_edge_density: total edge pixels / total pixel count
    - horizontal_edge_density: circumferential grooves oriented horizontally (or along tyre circumference)
    - vertical_edge_density: lateral blocks/sipes oriented vertically
    - edge_orientation_mean: average orientation angle in degrees
    - edge_orientation_std: dispersion of edge directions (reflects multi-directional siping vs plain tread)
    """
    h, w = edge_map.shape[:2]
    total_pixels = float(h * w)

    edge_pixels = np.sum(edge_map > 0)
    total_edge_density = float(edge_pixels / total_pixels)

    if grad_x is None or grad_y is None:
        grad_x = cv2.Sobel(edge_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(edge_map, cv2.CV_64F, 0, 1, ksize=3)

    # Compute orientation for edge pixels only
    edge_mask = edge_map > 0
    if np.any(edge_mask):
        gx = grad_x[edge_mask]
        gy = grad_y[edge_mask]
        # Angles in degrees in range [-180, 180] -> map to [0, 180]
        angles = np.mod(np.degrees(np.arctan2(gy, gx)), 180.0)

        # Horizontal edges: gradient points primarily vertical (angles near 90°)
        # Vertical edges: gradient points primarily horizontal (angles near 0° or 180°)
        horizontal_edges = np.sum((angles >= 60) & (angles <= 120))
        vertical_edges = np.sum((angles <= 30) | (angles >= 150))

        horiz_density = float(horizontal_edges / total_pixels)
        vert_density = float(vertical_edges / total_pixels)
        orient_mean = float(np.mean(angles))
        orient_std = float(np.std(angles))
    else:
        horiz_density = 0.0
        vert_density = 0.0
        orient_mean = 0.0
        orient_std = 0.0

    return {
        "total_edge_density": round(total_edge_density, 5),
        "horizontal_edge_density": round(horiz_density, 5),
        "vertical_edge_density": round(vert_density, 5),
        "edge_orientation_mean": round(orient_mean, 2),
        "edge_orientation_std": round(orient_std, 2)
    }
