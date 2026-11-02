"""
TreadVision: End-to-End Image Processing Pipeline
Orchestrates preprocessing, enhancement, segmentation, edge detection, morphology, and feature extraction.
"""
from typing import Dict, Any, Optional, Tuple
import time
import numpy as np
import cv2

from app.image_processing.preprocessing import (
    validate_image, resize_image, to_grayscale, reduce_noise
)
from app.image_processing.enhancement import enhance_contrast
from app.image_processing.segmentation import segment_tread_region
from app.image_processing.edges import detect_canny_edges, detect_sobel_edges, detect_laplacian_edges
from app.image_processing.morphology import apply_morphology
from app.image_processing.feature_fusion import FeatureFusion


class PipelineResult:
    """Encapsulates all intermediate visual stages, features, and diagnostic metrics."""
    def __init__(
        self,
        success: bool,
        error_message: str = "",
        stages: Dict[str, np.ndarray] = None,
        features: Dict[str, Any] = None,
        feature_vector: np.ndarray = None,
        metrics: Dict[str, Any] = None,
        execution_time_ms: float = 0.0
    ):
        self.success = success
        self.error_message = error_message
        self.stages = stages or {}
        self.features = features or {}
        self.feature_vector = feature_vector
        self.metrics = metrics or {}
        self.execution_time_ms = execution_time_ms


class ImageProcessingPipeline:
    """
    Modular, transparent classical computer vision pipeline for tyre tread analysis.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        noise_method: str = "bilateral",
        contrast_method: str = "clahe",
        clahe_clip_limit: float = 2.0,
        segmentation_method: str = "otsu",
        canny_low: int = 50,
        canny_high: int = 150,
        morph_op: str = "closing",
        morph_kernel_size: Tuple[int, int] = (3, 3)
    ):
        self.target_size = target_size
        self.noise_method = noise_method
        self.contrast_method = contrast_method
        self.clahe_clip_limit = clahe_clip_limit
        self.segmentation_method = segmentation_method
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.morph_op = morph_op
        self.morph_kernel_size = morph_kernel_size

    def process(
        self,
        image_input: Any,
        fallback_roi: Optional[Tuple[int, int, int, int]] = None
    ) -> PipelineResult:
        """
        Execute full end-to-end pipeline and capture all visual stages.
        """
        start_time = time.perf_counter()

        # Stage 1: Image Validation
        is_valid, err_msg, cv_img = validate_image(image_input)
        if not is_valid:
            return PipelineResult(success=False, error_message=err_msg)

        # Stage 2: Resize / Dimension Normalization
        original_resized = resize_image(cv_img, target_size=self.target_size, keep_aspect_ratio=True)

        # Stage 3: Grayscale Conversion
        gray = to_grayscale(original_resized)

        # Stage 4: Noise Reduction
        denoised = reduce_noise(gray, method=self.noise_method)

        # Stage 5: Contrast Enhancement
        enhanced = enhance_contrast(denoised, method=self.contrast_method, clip_limit=self.clahe_clip_limit)

        # Stage 6: Tread Region Segmentation
        tread_mask, segmented_tread, seg_meta = segment_tread_region(
            enhanced, method=self.segmentation_method, fallback_roi=fallback_roi
        )

        # Stage 7: Morphological Processing
        morph_cleaned = apply_morphology(
            tread_mask,
            operation=self.morph_op,
            kernel_shape="rect",
            kernel_size=self.morph_kernel_size,
            iterations=1
        )

        # Stage 8: Edge Detection
        canny_edges = detect_canny_edges(enhanced, low_threshold=self.canny_low, high_threshold=self.canny_high)
        # Mask edges with segmented tread to focus strictly on tread contact patch
        tread_edges = cv2.bitwise_and(canny_edges, canny_edges, mask=morph_cleaned)

        # Optional Sobel & Laplacian for rich diagnostic inspection
        sobel_mag, grad_x, grad_y = detect_sobel_edges(enhanced)
        laplacian_mag = detect_laplacian_edges(enhanced)

        # Stage 9: Feature Extraction & Fusion
        features_dict, feature_vector = FeatureFusion.extract_all(
            enhanced_gray=enhanced,
            edge_map=tread_edges,
            binary_mask=morph_cleaned
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Calculate high-level pattern integrity and groove visibility indices
        edge_density = features_dict["edge"]["total_edge_density"]
        glcm_contrast = features_dict["texture"]["contrast"]
        glcm_homogeneity = features_dict["texture"]["homogeneity"]
        solidity = features_dict["shape"]["solidity"]

        # Tread pattern integrity heuristic (0-100%)
        # Sharp grooves + high contrast + defined edges indicate high pattern integrity
        integrity_score = min(100.0, max(10.0, (
            (edge_density / 0.12) * 35.0 +
            (min(glcm_contrast, 8.0) / 8.0) * 35.0 +
            (1.0 - glcm_homogeneity) * 30.0
        )))

        # Groove visibility index
        groove_visibility = min(100.0, max(10.0, (
            (features_dict["edge"]["horizontal_edge_density"] / 0.06) * 50.0 +
            (features_dict["edge"]["vertical_edge_density"] / 0.06) * 50.0
        )))

        # Edge definition score
        edge_definition = min(100.0, max(15.0, (edge_density / 0.10) * 100.0))

        # Texture condition score (higher contrast & lower homogeneity = better condition)
        texture_condition = min(100.0, max(15.0, (1.0 - glcm_homogeneity) * 100.0))

        metrics = {
            "pattern_integrity_pct": round(integrity_score, 1),
            "groove_visibility_pct": round(groove_visibility, 1),
            "edge_definition_pct": round(edge_definition, 1),
            "texture_condition_pct": round(texture_condition, 1),
            "segmentation_coverage_pct": round(seg_meta["coverage_ratio"] * 100.0, 1),
            "connected_components": seg_meta["connected_components_count"]
        }

        stages = {
            "Original": original_resized,
            "Grayscale": gray,
            "Denoised": denoised,
            "Enhanced (CLAHE)": enhanced,
            "Segmentation Mask": morph_cleaned,
            "Segmented Tread": segmented_tread,
            "Canny Edges": tread_edges,
            "Sobel Magnitude": sobel_mag,
            "Laplacian": laplacian_mag
        }

        return PipelineResult(
            success=True,
            stages=stages,
            features=features_dict,
            feature_vector=feature_vector,
            metrics=metrics,
            execution_time_ms=round(elapsed_ms, 2)
        )
