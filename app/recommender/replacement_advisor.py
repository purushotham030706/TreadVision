"""
TreadVision: Tyre Replacement Advisor and Fitment Intelligence
Combines vision-based pattern analysis, user-entered tread depths,
and database size compatibility to formulate safety-oriented replacement guidance.
"""
from typing import Dict, Any, List, Optional
from app.config import (
    LEGAL_MINIMUM_DEPTH_MM,
    WET_WEATHER_RECOMMENDED_DEPTH_MM,
    WINTER_RECOMMENDED_DEPTH_MM
)


class ReplacementAdvisor:
    """
    Evaluates tyre wear and delivers structured replacement recommendations.
    """

    @staticmethod
    def evaluate(
        wear_class: str,
        metrics: Dict[str, Any],
        measured_depth_mm: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Formulate replacement recommendation using clear, transparent criteria.
        """
        integrity = metrics.get("pattern_integrity_pct", 50.0)
        groove_vis = metrics.get("groove_visibility_pct", 50.0)

        # 1. Depth-based assessment (if physical measurement provided)
        depth_status = None
        depth_note = None

        if measured_depth_mm is not None:
            if measured_depth_mm <= LEGAL_MINIMUM_DEPTH_MM:
                depth_status = "CRITICAL_BELOW_LEGAL_LIMIT"
                depth_note = (
                    f"Measured tread depth ({measured_depth_mm:.1f} mm) is at or below the legal limit "
                    f"({LEGAL_MINIMUM_DEPTH_MM} mm). Immediate replacement is legally required in most jurisdictions."
                )
            elif measured_depth_mm <= WET_WEATHER_RECOMMENDED_DEPTH_MM:
                depth_status = "WARNING_APPROACHING_LIMIT"
                depth_note = (
                    f"Measured tread depth ({measured_depth_mm:.1f} mm) is below the recommended wet-weather safety "
                    f"threshold ({WET_WEATHER_RECOMMENDED_DEPTH_MM} mm). Significantly increased risk of aquaplaning."
                )
            else:
                depth_status = "SATISFACTORY_DEPTH"
                depth_note = (
                    f"Measured tread depth ({measured_depth_mm:.1f} mm) remains above standard safety warning levels."
                )

        # 2. Vision + Wear Class Assessment
        if wear_class == "worn" or (depth_status == "CRITICAL_BELOW_LEGAL_LIMIT"):
            action = "IMMEDIATE REPLACEMENT ADVISED"
            status_level = "CRITICAL"
            advice = (
                "The tread pattern exhibits substantial groove depletion and edge loss. "
                "Water evacuation channels are compromised. Prompt replacement is strongly advised "
                "to maintain vehicle braking distance and wet handling stability."
            )
        elif wear_class == "moderate" or (depth_status == "WARNING_APPROACHING_LIMIT"):
            action = "MONITOR & PLAN REPLACEMENT"
            status_level = "MONITOR"
            advice = (
                "The tyre displays intermediate wear with discernible groove structure but softened edges. "
                "Monitor tyre wear monthly and inspect wear indicator bars. "
                "Plan replacement before entering severe wet or winter driving conditions."
            )
        else:
            action = "SATISFACTORY — ROUTINE INSPECTION"
            status_level = "GOOD"
            advice = (
                "The tyre surface demonstrates healthy pattern integrity, sharp groove boundaries, "
                "and robust siping. Maintain recommended tyre inflation pressures and rotate periodically."
            )

        disclaimer = (
            "Safety Notice: TreadVision optical assessments are assistive and research-oriented. "
            "They do not constitute a certified mechanical inspection or legal roadworthiness certificate. "
            "Always verify tread depth across multiple circumferential grooves with a calibrated gauge."
        )

        return {
            "action": action,
            "status_level": status_level,
            "advice": advice,
            "disclaimer": disclaimer,
            "depth_status": depth_status,
            "depth_note": depth_note,
            "measured_depth_mm": measured_depth_mm,
            "integrity_pct": integrity,
            "groove_visibility_pct": groove_vis
        }

    @staticmethod
    def find_replacements_for_tyre(
        current_tyre_id: int,
        db_session
    ) -> List[Dict[str, Any]]:
        """
        Query database for replacement tyres sharing the same available sizes or category.
        """
        from app.database.models import Tyre, TyreSize

        current = db_session.query(Tyre).filter(Tyre.id == current_tyre_id).first()
        if not current:
            return []

        # Find all sizes supported by current tyre
        current_sizes = {(s.width, s.aspect_ratio, s.rim_diameter) for s in current.sizes}

        # Query other tyres that share at least one identical dimension
        all_tyres = db_session.query(Tyre).filter(Tyre.id != current_tyre_id).all()
        replacements = []

        for candidate in all_tyres:
            cand_sizes = {(s.width, s.aspect_ratio, s.rim_diameter) for s in candidate.sizes}
            shared = current_sizes.intersection(cand_sizes)
            if shared:
                replacements.append({
                    "id": candidate.id,
                    "manufacturer": candidate.manufacturer,
                    "model": candidate.model,
                    "category": candidate.category,
                    "shared_sizes": [f"{w}/{a} R{r}" for (w, a, r) in shared],
                    "pattern_type": candidate.tread_pattern.pattern_type if candidate.tread_pattern else "Asymmetric",
                    "description": candidate.description
                })

        return replacements
