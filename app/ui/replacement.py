"""
TreadVision: Module 2 — Tyre Replacement Advisor
Provides replacement guidance, legal threshold checks, manual depth gauge evaluation,
and compatible database tyre recommendations.
"""
import streamlit as st
from app.config import (
    LEGAL_MINIMUM_DEPTH_MM,
    WET_WEATHER_RECOMMENDED_DEPTH_MM,
    WINTER_RECOMMENDED_DEPTH_MM
)
from app.database.connection import get_db_session
from app.database.models import Tyre, TyreSize
from app.recommender.replacement_advisor import ReplacementAdvisor
from app.ui.components import render_header


def render_replacement():
    """Render the Tyre Replacement Advisor page."""
    render_header(
        "Module 2: Tyre Replacement Advisor",
        "Translate tread wear assessments into safety guidance, legal threshold checks, and compatible replacement options."
    )

    # Check if we have analysis data from Module 1
    has_analysis = "latest_analysis" in st.session_state
    latest = st.session_state.get("latest_analysis", {})

    if not has_analysis:
        st.info("ℹ️ No recent analysis loaded from the Tread Analyzer. You can select a simulated tyre condition below or analyze an image in Module 1 first.")

    col_assess, col_depth = st.columns([1.5, 1], gap="medium")

    with col_assess:
        st.subheader("1. Tyre Wear Status")
        if has_analysis:
            pred = latest.get("prediction", {})
            metrics = latest.get("metrics", {})
            wear_class = pred.get("predicted_class", "moderate")
            label = pred.get("label", "Moderately Worn")
            st.success(f"Loaded Analysis for: {latest.get('image_label', 'Tyre Image')}")
        else:
            wear_options = {
                "Good / Low Wear": "good",
                "Moderately Worn": "moderate",
                "Heavily Worn": "worn"
            }
            selected_label = st.selectbox("Select Current Tyre Condition:", list(wear_options.keys()), index=1)
            wear_class = wear_options[selected_label]
            label = selected_label
            metrics = {
                "pattern_integrity_pct": 78.0 if wear_class == "moderate" else (92.0 if wear_class == "good" else 35.0),
                "groove_visibility_pct": 72.0 if wear_class == "moderate" else (88.0 if wear_class == "good" else 30.0),
                "edge_definition_pct": 65.0 if wear_class == "moderate" else (90.0 if wear_class == "good" else 25.0),
                "texture_condition_pct": 74.0 if wear_class == "moderate" else (91.0 if wear_class == "good" else 28.0)
            }

        # Metrics display
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pattern Integrity", f"{metrics.get('pattern_integrity_pct', 0)}%")
        m2.metric("Groove Visibility", f"{metrics.get('groove_visibility_pct', 0)}%")
        m3.metric("Edge Definition", f"{metrics.get('edge_definition_pct', 0)}%")
        m4.metric("Texture Condition", f"{metrics.get('texture_condition_pct', 0)}%")

    with col_depth:
        st.subheader("2. Physical Tread Depth (Gauge)")
        has_manual_depth = st.checkbox("Enter measured physical depth", value=False)
        measured_depth = None
        if has_manual_depth:
            measured_depth = st.slider(
                "Measured Depth (mm):",
                min_value=0.5,
                max_value=9.0,
                value=3.2,
                step=0.1,
                help="Calibrated physical depth measurement across main circumferential grooves."
            )
            st.caption(f"Legal minimum: **{LEGAL_MINIMUM_DEPTH_MM} mm** | Wet advisory: **{WET_WEATHER_RECOMMENDED_DEPTH_MM} mm**")

    # Recommendation Calculation
    eval_res = ReplacementAdvisor.evaluate(
        wear_class=wear_class,
        metrics=metrics,
        measured_depth_mm=measured_depth
    )

    # Structured Recommendation Card
    st.markdown("<br>", unsafe_allow_html=True)
    status_bg = {
        "GOOD": "rgba(16, 185, 129, 0.15)",
        "MONITOR": "rgba(245, 158, 11, 0.15)",
        "CRITICAL": "rgba(239, 68, 68, 0.15)"
    }.get(eval_res["status_level"], "rgba(6, 182, 212, 0.15)")

    status_border = {
        "GOOD": "#10b981",
        "MONITOR": "#f59e0b",
        "CRITICAL": "#ef4444"
    }.get(eval_res["status_level"], "#06b6d4")

    st.markdown(f"""
    <div style="background: {status_bg}; border: 1px solid {status_border}; border-radius: 12px; padding: 22px 24px; margin-bottom: 24px;">
        <span class="tv-badge" style="background: {status_border}; color: #000; font-weight: 700;">RECOMMENDATION</span>
        <h2 style="color: #ffffff; margin: 8px 0 6px 0;">{eval_res['action']}</h2>
        <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6; margin-bottom: 8px;">
            {eval_res['advice']}
        </p>
        {"<div style='color: #f87171; font-weight: 600; margin-top: 8px;'>⚠️ " + eval_res['depth_note'] + "</div>" if eval_res.get('depth_note') else ""}
    </div>
    """, unsafe_allow_html=True)

    # Replacement Matching Against Database
    st.markdown("---")
    st.subheader("3. Select Current Tyre for Direct Replacements")

    session = get_db_session()
    try:
        all_tyres = session.query(Tyre).all()
        tyre_options = {f"{t.manufacturer} {t.model} ({t.category})": t.id for t in all_tyres}
        selected_tyre_label = st.selectbox("Current Installed Tyre Model:", list(tyre_options.keys()))
        selected_tyre_id = tyre_options[selected_tyre_label]

        current_tyre = session.query(Tyre).filter(Tyre.id == selected_tyre_id).first()

        # Display current tyre sizes
        sizes_str = ", ".join([s.formatted_size for s in current_tyre.sizes])
        st.markdown(f"**Available OEM Sizes for Selected Tyre:** `{sizes_str}`")

        # Find compatible replacements
        replacements = ReplacementAdvisor.find_replacements_for_tyre(selected_tyre_id, session)

        st.markdown(f"#### Compatible Replacement Options ({len(replacements)} Available)")

        rep_cols = st.columns(min(3, max(1, len(replacements))))
        for idx, rep in enumerate(replacements):
            col_target = rep_cols[idx % len(rep_cols)]
            with col_target:
                st.markdown(f"""
                <div class="tv-card" style="margin-bottom: 12px;">
                    <span class="tv-badge tv-badge-cyan">{rep['category']}</span>
                    <div style="font-weight: 700; font-size: 1.1rem; color: #ffffff; margin-top: 6px;">
                        {rep['manufacturer']} {rep['model']}
                    </div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin: 6px 0;">
                        Pattern: {rep['pattern_type']}
                    </div>
                    <div style="font-size: 0.82rem; color: #38bdf8;">
                        Matching Sizes: {', '.join(rep['shared_sizes'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    finally:
        session.close()

    # Safety Disclaimer
    st.markdown(f"""
    <div style="margin-top: 30px; padding: 12px 16px; background: #0f172a; border-radius: 8px; border: 1px solid #1e293b; color: #94a3b8; font-size: 0.82rem;">
        {eval_res['disclaimer']}
    </div>
    """, unsafe_allow_html=True)
