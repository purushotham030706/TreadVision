"""
TreadVision: Feature Fusion and Unified Vector Serialization
Combines Edge (5), Texture (6), Shape (14: 7 morphological + 7 Hu moments),
and Physical Wear-Specific (6) Descriptors into a structured 31-dimensional numerical vector.
"""
from typing import Dict, Any, List, Tuple, Optional
import json
import numpy as np
from app.image_processing.edges import extract_edge_features
from app.image_processing.texture import extract_glcm_features
from app.image_processing.shape import compute_shape_features
from app.image_processing.wear_features import extract_wear_specific_features, WEAR_FEATURE_NAMES

# Standard ordered feature names for consistent vectorization (31-D)
FEATURE_NAMES = [
    # 5 Edge features (0-4)
    "edge_total_density",
    "edge_horiz_density",
    "edge_vert_density",
    "edge_orient_mean",
    "edge_orient_std",
    # 6 Texture features (5-10)
    "texture_contrast",
    "texture_dissimilarity",
    "texture_homogeneity",
    "texture_energy",
    "texture_correlation",
    "texture_asm",
    # 7 Shape morphological features (11-17)
    "shape_area_ratio",
    "shape_perimeter_ratio",
    "shape_aspect_ratio",
    "shape_extent",
    "shape_solidity",
    "shape_circularity",
    "shape_equiv_diameter",
    # 7 Shape Hu Moments (18-24)
    "hu_moment_1",
    "hu_moment_2",
    "hu_moment_3",
    "hu_moment_4",
    "hu_moment_5",
    "hu_moment_6",
    "hu_moment_7",
    # 6 Physical Wear-Specific features (25-30)
    "groove_to_land_contrast",
    "groove_visibility_ratio",
    "sipe_visibility_freq",
    "local_groove_continuity",
    "local_texture_contrast",
    "groove_wall_gradient"
]


class FeatureFusion:
    """
    Orchestrates the fusion of diverse computer vision feature extractors
    into a standardized, normalized 31-dimensional feature vector suitable
    for machine learning and forensic similarity metrics.
    """

    @staticmethod
    def extract_all(
        enhanced_gray: np.ndarray,
        edge_map: np.ndarray,
        binary_mask: np.ndarray,
        raw_image: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Extract structured dictionary and flat 1D numpy vector (len = 31).
        """
        edge_dict = extract_edge_features(edge_map)
        texture_dict = extract_glcm_features(enhanced_gray)
        shape_dict = compute_shape_features(binary_mask)

        # Extract 6 physical wear-specific features
        wear_input = raw_image if raw_image is not None else enhanced_gray
        wear_dict = extract_wear_specific_features(wear_input)

        # Build ordered vector
        vector_values = [
            edge_dict.get("total_edge_density", 0.0),
            edge_dict.get("horizontal_edge_density", 0.0),
            edge_dict.get("vertical_edge_density", 0.0),
            edge_dict.get("edge_orientation_mean", 0.0),
            edge_dict.get("edge_orientation_std", 0.0),

            texture_dict.get("contrast", 0.0),
            texture_dict.get("dissimilarity", 0.0),
            texture_dict.get("homogeneity", 0.0),
            texture_dict.get("energy", 0.0),
            texture_dict.get("correlation", 0.0),
            texture_dict.get("ASM", 0.0),

            shape_dict.get("area_ratio", 0.0),
            shape_dict.get("perimeter_ratio", 0.0),
            shape_dict.get("aspect_ratio", 1.0),
            shape_dict.get("extent", 0.0),
            shape_dict.get("solidity", 0.0),
            shape_dict.get("circularity", 0.0),
            shape_dict.get("equivalent_diameter_ratio", 0.0),
        ]

        # Append 7 Hu moments
        hu_moments = shape_dict.get("hu_moments", [0.0] * 7)
        if len(hu_moments) < 7:
            hu_moments = hu_moments + [0.0] * (7 - len(hu_moments))
        vector_values.extend(hu_moments[:7])

        # Append 6 physical wear-specific features
        wear_values = [wear_dict.get(k, 0.0) for k in WEAR_FEATURE_NAMES]
        vector_values.extend(wear_values)

        vector = np.array(vector_values, dtype=np.float32)

        composite_dict = {
            "edge": edge_dict,
            "texture": texture_dict,
            "shape": shape_dict,
            "wear": wear_dict,
            "vector": vector.tolist(),
            "feature_names": FEATURE_NAMES
        }

        return composite_dict, vector

    @staticmethod
    def to_json(composite_dict: Dict[str, Any]) -> str:
        """Serialize features dictionary to JSON."""
        return json.dumps(composite_dict)

    @staticmethod
    def from_json(json_str: str) -> Dict[str, Any]:
        """Deserialize features dictionary from JSON."""
        return json.loads(json_str)
