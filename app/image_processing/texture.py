"""
TreadVision: GLCM (Gray-Level Co-occurrence Matrix) Texture Feature Extraction
"""
from typing import Dict, List, Tuple
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from app.config import GLCM_DISTANCES, GLCM_ANGLES, GLCM_PROPERTIES


def compute_glcm(
    gray_image: np.ndarray,
    distances: List[int] = None,
    angles: List[float] = None,
    levels: int = 256
) -> np.ndarray:
    """
    Compute Gray-Level Co-occurrence Matrix (GLCM) over specified spatial offsets and directions.
    To improve efficiency and noise resilience, images are quantized if levels < 256.
    """
    if distances is None:
        distances = GLCM_DISTANCES
    if angles is None:
        angles = GLCM_ANGLES

    img = gray_image.copy()
    if levels < 256:
        # Quantize to fewer gray levels (e.g., 64)
        factor = 256 / levels
        img = (img / factor).astype(np.uint8)

    glcm = graycomatrix(
        img,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )
    return glcm


def extract_glcm_features(
    gray_image: np.ndarray,
    distances: List[int] = None,
    angles: List[float] = None,
    levels: int = 64
) -> Dict[str, float]:
    """
    Extract key GLCM statistical descriptors:
    - contrast: measures local intensity variations (high in sharp, fresh grooves)
    - dissimilarity: linear measure of local variation
    - homogeneity: closeness of pixel distributions (high in worn, smooth tread)
    - energy: sum of squared elements in GLCM (measures uniformity)
    - correlation: joint probability occurrences of specified pixel pairs
    - ASM: Angular Second Moment (energy squared)
    Averages across angles and distances for rotational stability.
    """
    glcm = compute_glcm(gray_image, distances=distances, angles=angles, levels=levels)

    features = {}
    for prop in GLCM_PROPERTIES:
        val_matrix = graycoprops(glcm, prop)
        # Average across all distances and angles
        mean_val = float(np.mean(val_matrix))
        features[prop] = round(mean_val, 4)

    return features


def extract_directional_glcm(
    gray_image: np.ndarray,
    levels: int = 64
) -> Dict[str, Dict[str, float]]:
    """
    Extract GLCM features per angle (0°, 45°, 90°, 135°) to analyze directional tread siping.
    """
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    angle_names = ["0_deg_horizontal", "45_deg_diagonal", "90_deg_vertical", "135_deg_diagonal"]

    glcm = compute_glcm(gray_image, distances=[1], angles=angles, levels=levels)

    directional_results = {}
    for i, name in enumerate(angle_names):
        directional_results[name] = {}
        for prop in GLCM_PROPERTIES:
            val = float(graycoprops(glcm, prop)[0, i])
            directional_results[name][prop] = round(val, 4)

    return directional_results
