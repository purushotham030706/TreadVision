"""
TreadVision: Forensic Tread Impression Analysis and Multi-Criteria Similarity Engine
Matches photographed crime-scene tyre impressions (in mud, soil, sand, dirt)
against verified reference tyre tread patterns.
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from scipy.spatial.distance import cosine, euclidean

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
        component_scores: Dict[str, float],
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
    """

    FORENSIC_DISCLAIMER = (
        "Forensic Analysis Notice: TreadVision provides image-based similarity analysis "
        "for research, investigative lead generation, and educational purposes. A similarity result "
        "does not establish that a particular vehicle or tyre was present at a crime scene and "
        "must not be treated as definitive forensic evidence without professional physical "
        "examination, calibrated cast/photogrammetric evidence, and verified chain-of-custody procedures."
    )

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or FORENSIC_WEIGHTS
        self.pipeline = ImageProcessingPipeline(
            target_size=(512, 512),
            noise_method="bilateral",
            contrast_method="clahe",
            clahe_clip_limit=3.0,  # Slightly higher contrast for degraded impressions
            segmentation_method="adaptive"  # Adaptive thresholding handles uneven mud/soil lighting
        )
        # ORB local keypoint detector for invariant structural feature matching
        self.orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def process_impression(
        self,
        impression_input: Any,
        perspective_pts: Optional[np.ndarray] = None
    ) -> PipelineResult:
        """
        Preprocess the crime-scene impression image, applying perspective correction
        if 4 corner points are provided, followed by contrast enhancement and edge extraction.
        """
        # If perspective correction coordinates provided:
        if perspective_pts is not None and len(perspective_pts) == 4:
            # Warp perspective to top-down view
            is_valid, _, cv_img = self.pipeline.process(impression_input).stages.get("Original"), "", None
            # Standard wrap
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
    ) -> Tuple[float, int]:
        """
        Compute local keypoint similarity using ORB descriptors and Lowe's ratio test.
        """
        try:
            ref_bgr = cv2.imread(ref_image_path)
            if ref_bgr is None:
                return 0.5, 0

            ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
            ref_gray = cv2.resize(ref_gray, (512, 512))

            kp1, des1 = self.orb.detectAndCompute(query_gray, None)
            kp2, des2 = self.orb.detectAndCompute(ref_gray, None)

            if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                return 0.4, 0

            matches = self.matcher.knnMatch(des1, des2, k=2)
            good_matches = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            match_count = len(good_matches)
            max_possible = min(len(des1), len(des2))
            ratio = float(match_count / max_possible) if max_possible > 0 else 0.0

            # Scale ratio into [0.3, 0.95] range
            local_score = min(0.98, max(0.20, 0.30 + ratio * 1.8))
            return local_score, match_count
        except Exception:
            return 0.5, 0

    def calculate_similarity(
        self,
        query_features: Dict[str, Any],
        query_gray: np.ndarray,
        db_tyre_features: Dict[str, Any],
        ref_image_path: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate documented multi-criteria similarity score:
        S_overall = w_edge * S_edge + w_texture * S_texture + w_shape * S_shape + w_local * S_local
        """
        # 1. Edge Similarity (Density & directional distribution)
        q_edge = query_features.get("edge", {})
        r_edge = db_tyre_features.get("edge", {})

        q_density = q_edge.get("total_edge_density", 0.05)
        r_density = r_edge.get("total_edge_density", 0.05)
        density_diff = abs(q_density - r_density) / (max(q_density, r_density) + 1e-4)
        edge_sim = max(0.1, 1.0 - density_diff)

        # Directional factor
        q_horiz = q_edge.get("horizontal_edge_density", 0.0)
        r_horiz = r_edge.get("horizontal_edge_density", 0.0)
        horiz_diff = abs(q_horiz - r_horiz) / (max(q_horiz, r_horiz) + 1e-4)
        edge_score = 0.6 * edge_sim + 0.4 * max(0.1, 1.0 - horiz_diff)

        # 2. Texture Similarity (GLCM contrast, homogeneity, correlation cosine similarity)
        q_tex = query_features.get("texture", {})
        r_tex = db_tyre_features.get("texture", {})

        q_vec = np.array([
            q_tex.get("contrast", 1.0) / 10.0,
            q_tex.get("dissimilarity", 1.0) / 5.0,
            q_tex.get("homogeneity", 0.5),
            q_tex.get("energy", 0.2),
            (q_tex.get("correlation", 0.5) + 1.0) / 2.0,
            q_tex.get("ASM", 0.1)
        ])
        r_vec = np.array([
            r_tex.get("contrast", 1.0) / 10.0,
            r_tex.get("dissimilarity", 1.0) / 5.0,
            r_tex.get("homogeneity", 0.5),
            r_tex.get("energy", 0.2),
            (r_tex.get("correlation", 0.5) + 1.0) / 2.0,
            r_tex.get("ASM", 0.1)
        ])

        norm_q = np.linalg.norm(q_vec)
        norm_r = np.linalg.norm(r_vec)
        if norm_q > 0 and norm_r > 0:
            texture_score = float(np.dot(q_vec, r_vec) / (norm_q * norm_r))
        else:
            texture_score = 0.5
        texture_score = max(0.1, min(0.99, texture_score))

        # 3. Shape Similarity (Hu moments exponential distance & solidity)
        q_hu = np.array(query_features.get("shape", {}).get("hu_moments", [0.0]*7))
        r_hu = np.array(db_tyre_features.get("shape", {}).get("hu_moments", [0.0]*7))
        hu_dist = np.linalg.norm(q_hu - r_hu)
        shape_hu_sim = float(np.exp(-0.25 * hu_dist))

        q_solid = query_features.get("shape", {}).get("solidity", 0.5)
        r_solid = db_tyre_features.get("shape", {}).get("solidity", 0.5)
        solid_sim = max(0.1, 1.0 - abs(q_solid - r_solid))
        shape_score = 0.6 * shape_hu_sim + 0.4 * solid_sim

        # 4. Local Pattern Keypoint Similarity (ORB feature matching)
        local_score, _ = self.compute_local_similarity(query_gray, ref_image_path)

        # Weighted aggregation
        w = self.weights
        overall_score = (
            w["edge"] * edge_score +
            w["texture"] * texture_score +
            w["shape"] * shape_score +
            w["local"] * local_score
        ) * 100.0

        overall_score = max(15.0, min(96.5, overall_score))

        component_scores = {
            "edge_similarity": round(float(edge_score * 100.0), 1),
            "texture_similarity": round(float(texture_score * 100.0), 1),
            "shape_similarity": round(float(shape_score * 100.0), 1),
            "local_pattern_similarity": round(float(local_score * 100.0), 1)
        }

        return overall_score, component_scores

    def match(
        self,
        query_result: PipelineResult,
        db_records: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[ForensicMatchCandidate]:
        """
        Rank reference database tyres against the processed impression.
        """
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

            # Assign academic confidence tier
            if sim_score >= 85.0:
                confidence = "High Similarity Match"
            elif sim_score >= 70.0:
                confidence = "Moderate Similarity Candidate"
            else:
                confidence = "Low Similarity Lead"

            candidate = ForensicMatchCandidate(
                rank=0,
                tyre_id=record["id"],
                manufacturer=record["manufacturer"],
                model=record["model"],
                pattern_name=record.get("pattern_name", ""),
                pattern_type=record.get("pattern_type", "Asymmetric"),
                symmetry=record.get("symmetry", "Asymmetric"),
                similarity_score=sim_score,
                confidence_level=confidence,
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
