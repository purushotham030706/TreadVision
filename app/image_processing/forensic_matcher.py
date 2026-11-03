"""
TreadVision: Forensic Tread Impression Analysis and Multi-Criteria Similarity Engine
Matches photographed crime-scene tyre impressions (in mud, soil, sand, dirt)
against verified reference tyre tread patterns using documented mathematical metrics.
"""
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np
import cv2

from app.config import FORENSIC_WEIGHTS
from app.image_processing.pipeline import ImageProcessingPipeline, PipelineResult


class ForensicMatchCandidate:
    """Represents a ranked candidate match from the tyre database."""
    def __init__(
        self,
        rank: int,
        tyre_id: int,
        manufacturer: str,
        model: str,
        pattern_name: str,
        pattern_type: str,
        symmetry: str,
        similarity_score: float,
        confidence_level: str,
        component_scores: Dict[str, Any],
        reference_image_path: str,
        available_sizes: List[str],
        compatible_vehicles: List[str],
        description: str
    ):
        self.rank = rank
        self.tyre_id = tyre_id
        self.manufacturer = manufacturer
        self.model = model
        self.pattern_name = pattern_name
        self.pattern_type = pattern_type
        self.symmetry = symmetry
        self.similarity_score = round(similarity_score, 1)
        self.confidence_level = confidence_level
        self.component_scores = component_scores
        self.reference_image_path = reference_image_path
        self.available_sizes = available_sizes
        self.compatible_vehicles = compatible_vehicles
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "tyre_id": self.tyre_id,
            "full_name": f"{self.manufacturer} {self.model}",
            "pattern_name": self.pattern_name,
            "pattern_type": self.pattern_type,
            "symmetry": self.symmetry,
            "similarity_score": self.similarity_score,
            "confidence_level": self.confidence_level,
            "component_scores": self.component_scores,
            "reference_image_path": self.reference_image_path,
            "available_sizes": self.available_sizes,
            "compatible_vehicles": self.compatible_vehicles,
            "description": self.description
        }


class ForensicMatcher:
    """
    Forensic impression matching engine with multi-criteria feature fusion.

    Mathematical Formulation:
      - Edge similarity:    S_edge    = 1.0 - (L1_norm_edge / max_L1_edge)
      - Texture similarity: S_texture = 1.0 - (L1_norm_texture / max_L1_texture)
      - Shape similarity:   S_shape   = 1.0 - (L1_norm_shape / max_L1_shape)
      - Local similarity:   S_local   = good_matches / min(des1, des2)   [if matches > 0]
                                      None                                [if 0 matches or keypoints < 2]

      Overall Similarity:
        When local features available:
          S_overall = (w_e * S_e + w_t * S_t + w_s * S_s + w_l * S_l) / (w_e + w_t + w_s + w_l)
        When local features insufficient:
          S_overall = (w_e * S_e + w_t * S_t + w_s * S_s) / (w_e + w_t + w_s)
    """

    FORENSIC_DISCLAIMER = (
        "Forensic Analysis Notice: TreadVision provides image-based similarity analysis "
        "for research, investigative lead generation, and educational purposes. A similarity result "
        "does not establish that a particular vehicle or tyre was present at a crime scene and "
        "must not be treated as definitive forensic evidence without professional physical "
        "examination, calibrated cast/photogrammetric evidence, and verified chain-of-custody procedures."
    )

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or FORENSIC_WEIGHTS.copy()
        self.pipeline = ImageProcessingPipeline(
            target_size=(512, 512),
            noise_method="bilateral",
            contrast_method="clahe",
            clahe_clip_limit=3.0,
            segmentation_method="adaptive"
        )
        self.orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def process_impression(
        self,
        impression_input: Any,
        perspective_pts: Optional[np.ndarray] = None
    ) -> PipelineResult:
        """Preprocess crime-scene impression image with optional perspective correction."""
        if perspective_pts is not None and len(perspective_pts) == 4:
            pts1 = np.float32(perspective_pts)
            pts2 = np.float32([[0, 0], [512, 0], [512, 512], [0, 512]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(impression_input, matrix, (512, 512))
            return self.pipeline.process(warped)

        return self.pipeline.process(impression_input)

    def compute_local_similarity(
        self,
        query_gray: np.ndarray,
        ref_image_path: str
    ) -> Tuple[Optional[float], int, str]:
        """
        Compute local keypoint similarity using ORB descriptors and Lowe's ratio test.
        Returns:
            (local_score, match_count, status_str)
            Where local_score is None if insufficient keypoints or 0 matches.
        """
        if not ref_image_path or not Path(ref_image_path).exists():
            return None, 0, "INSUFFICIENT_LOCAL_FEATURES"

        try:
            ref_bgr = cv2.imread(ref_image_path)
            if ref_bgr is None:
                return None, 0, "INSUFFICIENT_LOCAL_FEATURES"

            ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
            ref_gray = cv2.resize(ref_gray, (512, 512))

            kp1, des1 = self.orb.detectAndCompute(query_gray, None)
            kp2, des2 = self.orb.detectAndCompute(ref_gray, None)

            if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                return None, 0, "INSUFFICIENT_LOCAL_FEATURES"

            matches = self.matcher.knnMatch(des1, des2, k=2)
            good_matches = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            match_count = len(good_matches)
            if match_count == 0:
                return None, 0, "INSUFFICIENT_LOCAL_FEATURES"

            max_possible = min(len(des1), len(des2))
            ratio = float(match_count / max_possible) if max_possible > 0 else 0.0
            local_score = float(np.clip(ratio, 0.0, 1.0))
            return local_score, match_count, "AVAILABLE"
        except Exception:
            return None, 0, "INSUFFICIENT_LOCAL_FEATURES"

    def calculate_similarity(
        self,
        query_features: Dict[str, Any],
        query_gray: np.ndarray,
        db_tyre_features: Dict[str, Any],
        ref_image_path: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate documented multi-criteria similarity score with bounded mathematical metrics:
        Edge, Texture, and Shape use bounded L1 distances: S = 1.0 - (L1_norm / max_L1).
        When ORB matching fails (insufficient keypoints), weights are renormalized across available metrics.
        """
        # 1. Edge Similarity: Bounded L1 distance on density and directional distribution
        q_edge = query_features.get("edge", {})
        r_edge = db_tyre_features.get("edge", {})

        q_density = float(q_edge.get("total_edge_density", 0.0))
        r_density = float(r_edge.get("total_edge_density", 0.0))
        diff_density = abs(q_density - r_density)  # max 1.0

        q_horiz = float(q_edge.get("horizontal_edge_density", 0.0))
        r_horiz = float(r_edge.get("horizontal_edge_density", 0.0))
        diff_horiz = abs(q_horiz - r_horiz)  # max 1.0

        q_vert = float(q_edge.get("vertical_edge_density", 0.0))
        r_vert = float(r_edge.get("vertical_edge_density", 0.0))
        diff_vert = abs(q_vert - r_vert)  # max 1.0

        # L1 edge norm with sub-weights summing to 1.0 (max possible = 1.0)
        l1_edge = 0.5 * diff_density + 0.3 * diff_horiz + 0.2 * diff_vert
        edge_score = float(np.clip(1.0 - l1_edge, 0.0, 1.0))

        # 2. Texture Similarity: Bounded L1 distance on normalized GLCM properties
        q_tex = query_features.get("texture", {})
        r_tex = db_tyre_features.get("texture", {})

        diff_contrast = min(1.0, abs(float(q_tex.get("contrast", 0.0)) - float(r_tex.get("contrast", 0.0))) / 50.0)
        diff_dissim = min(1.0, abs(float(q_tex.get("dissimilarity", 0.0)) - float(r_tex.get("dissimilarity", 0.0))) / 20.0)
        diff_homo = min(1.0, abs(float(q_tex.get("homogeneity", 0.0)) - float(r_tex.get("homogeneity", 0.0))))
        diff_energy = min(1.0, abs(float(q_tex.get("energy", 0.0)) - float(r_tex.get("energy", 0.0))))
        diff_corr = min(1.0, abs(float(q_tex.get("correlation", 0.0)) - float(r_tex.get("correlation", 0.0))) / 2.0)
        diff_asm = min(1.0, abs(float(q_tex.get("ASM", 0.0)) - float(r_tex.get("ASM", 0.0))))

        l1_texture = (diff_contrast + diff_dissim + diff_homo + diff_energy + diff_corr + diff_asm) / 6.0
        texture_score = float(np.clip(1.0 - l1_texture, 0.0, 1.0))

        # 3. Shape Similarity: Bounded L1 distance on shape and invariant moments
        q_shape = query_features.get("shape", {})
        r_shape = db_tyre_features.get("shape", {})

        diff_area = min(1.0, abs(float(q_shape.get("area_ratio", 0.0)) - float(r_shape.get("area_ratio", 0.0))))
        diff_solid = min(1.0, abs(float(q_shape.get("solidity", 0.0)) - float(r_shape.get("solidity", 0.0))))
        diff_circ = min(1.0, abs(float(q_shape.get("circularity", 0.0)) - float(r_shape.get("circularity", 0.0))))

        q_hu = np.array(q_shape.get("hu_moments", [0.0]*7), dtype=float)
        r_hu = np.array(r_shape.get("hu_moments", [0.0]*7), dtype=float)
        if len(q_hu) == 7 and len(r_hu) == 7:
            diff_hu = min(1.0, float(np.mean(np.abs(q_hu - r_hu)) / 15.0))
        else:
            diff_hu = 0.5

        l1_shape = 0.25 * diff_hu + 0.25 * diff_solid + 0.25 * diff_circ + 0.25 * diff_area
        shape_score = float(np.clip(1.0 - l1_shape, 0.0, 1.0))

        # 4. Local Pattern Keypoint Similarity (ORB)
        local_score, match_count, local_status = self.compute_local_similarity(query_gray, ref_image_path)

        # 5. Weighted Combination with Renormalization on Missing Local Features
        w_e = self.weights.get("edge", 0.30)
        w_t = self.weights.get("texture", 0.25)
        w_s = self.weights.get("shape", 0.25)
        w_l = self.weights.get("local", 0.20)

        if local_score is not None:
            w_total = w_e + w_t + w_s + w_l
            overall = (w_e * edge_score + w_t * texture_score + w_s * shape_score + w_l * local_score) / w_total
            local_pct = round(float(local_score * 100.0), 1)
        else:
            # Renormalize component weights across remaining available metrics
            w_total = w_e + w_t + w_s
            overall = (w_e * edge_score + w_t * texture_score + w_s * shape_score) / w_total
            local_pct = None

        overall_score = round(float(np.clip(overall * 100.0, 0.0, 100.0)), 1)

        component_scores = {
            "edge_similarity": round(float(edge_score * 100.0), 1),
            "texture_similarity": round(float(texture_score * 100.0), 1),
            "shape_similarity": round(float(shape_score * 100.0), 1),
            "local_pattern_similarity": local_pct,
            "local_feature_status": local_status,
            "local_match_count": match_count
        }

        return overall_score, component_scores

    def match(
        self,
        query_result: PipelineResult,
        db_records: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[ForensicMatchCandidate]:
        """Rank reference database tyres against the processed impression."""
        if not query_result.success:
            return []

        query_gray = query_result.stages.get("Enhanced (CLAHE)", query_result.stages.get("Grayscale"))
        candidates = []

        for record in db_records:
            ref_features = record.get("features", {})
            ref_image_path = record.get("image_path", "")

            sim_score, comp_scores = self.calculate_similarity(
                query_features=query_result.features,
                query_gray=query_gray,
                db_tyre_features=ref_features,
                ref_image_path=ref_image_path
            )

            # Assign similarity tier
            if sim_score >= 80.0:
                similarity_tier = "High Similarity Match"
            elif sim_score >= 60.0:
                similarity_tier = "Moderate Similarity Candidate"
            else:
                similarity_tier = "Low Similarity Lead"

            candidate = ForensicMatchCandidate(
                rank=0,
                tyre_id=record["id"],
                manufacturer=record["manufacturer"],
                model=record["model"],
                pattern_name=record.get("pattern_name", ""),
                pattern_type=record.get("pattern_type", "Asymmetric"),
                symmetry=record.get("symmetry", "Asymmetric"),
                similarity_score=sim_score,
                confidence_level=similarity_tier,
                component_scores=comp_scores,
                reference_image_path=ref_image_path,
                available_sizes=record.get("sizes", []),
                compatible_vehicles=record.get("vehicles", []),
                description=record.get("description", "")
            )
            candidates.append(candidate)

        # Sort descending by similarity score
        candidates.sort(key=lambda c: c.similarity_score, reverse=True)

        # Assign ranks
        for idx, cand in enumerate(candidates[:top_k], 1):
            cand.rank = idx

        return candidates[:top_k]
