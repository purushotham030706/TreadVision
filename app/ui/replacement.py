"""
TreadVision: Module 2 — Tyre Replacement Advisor
Evaluates tyre wear metrics against reference safety thresholds and identifies compatible database replacements.
"""
from typing import Optional
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
    """Render the Tyre Replacement Advisor interface."""
    render_header(
        "TYRE REPLACEMENT",
        "Evaluate tyre condition and explore potential replacement options."
    )

    has_analysis = "latest_analysis" in st.session_state
    latest = st.session_state.get("latest_analysis", {})

    col_current, col_depth = st.columns([1.4, 1], gap="large")

    session = get_db_session()
    try:
        all_tyres = session.query(Tyre).all()
        tyre_options = {f"{t.manufacturer} {t.model} ({t.category})": t.id for t in all_tyres}

        with col_current:
            st.markdown("""
            <div style="font-size: 0.76rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F4; font-weight: 700; margin-bottom: 8px;">
                CURRENT TYRE PROFILE
            </div>
            """, unsafe_allow_html=True)

            selected_tyre_label = st.selectbox(
                "Select Current Tyre Model:",
                list(tyre_options.keys()),
                label_visibility="collapsed"
            )
            selected_tyre_id = tyre_options[selected_tyre_label]
            current_tyre = session.query(Tyre).filter(Tyre.id == selected_tyre_id).first()

            # Display sizes
            sizes_str = ", ".join([s.formatted_size for s in current_tyre.sizes])

            st.markdown(f"""
            <div style="background: #23252E; border: 1px solid #41434E; border-radius: 4px; padding: 18px; margin-top: 10px;">
                <div style="font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #D8125B; font-weight: 600;">
                    CURRENT TYRE
                </div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #F2F2F4; margin: 4px 0;">
                    {current_tyre.manufacturer.upper()} {current_tyre.model.upper()}
                </div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: #F2F2F4; margin-bottom: 8px;">
                    {sizes_str if sizes_str else 'Standard OEM Specifications'}
                </div>
                <div style="font-size: 0.8rem; color: #B4B5BC;">
                    Category: {current_tyre.category} &nbsp;|&nbsp; Construction: {current_tyre.construction}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_depth:
            st.markdown("""
            <div style="font-size: 0.76rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F4; font-weight: 700; margin-bottom: 8px;">
                REFERENCE TREAD DEPTH
            </div>
            """, unsafe_allow_html=True)

            has_depth_input = st.checkbox("Include physical gauge measurement", value=False)
            measured_depth = None

            if has_depth_input:
                measured_depth = st.slider(
                    "Calibrated Tread Depth (mm):",
                    min_value=0.5,
                    max_value=9.0,
                    value=3.2,
                    step=0.1
                )

            st.markdown(f"""
            <div style="background: #23252E; border: 1px solid #41434E; border-radius: 4px; padding: 16px; margin-top: 10px;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.2rem; font-weight: 600; color: #D8125B;">
                    {LEGAL_MINIMUM_DEPTH_MM:.1f} mm
                </div>
                <div style="font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; color: #F2F2F4; font-weight: 600; margin-bottom: 6px;">
                    Reference Legal Minimum
                </div>
                <div style="font-size: 0.78rem; color: #B4B5BC; line-height: 1.45;">
                    Applicable legal tread-depth requirements vary by jurisdiction.
                    Consult the relevant local regulations and tyre manufacturer guidance.
                </div>
                <div style="border-top: 1px solid #41434E; margin-top: 10px; padding-top: 8px; font-size: 0.76rem; color: #B4B5BC;">
                    Wet Weather Advisory: <span class="tv-mono">{WET_WEATHER_RECOMMENDED_DEPTH_MM:.1f} mm</span><br>
                    Winter Advisory: <span class="tv-mono">{WINTER_RECOMMENDED_DEPTH_MM:.1f} mm</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Condition Assessment Data
        st.markdown("<br>", unsafe_allow_html=True)
        if has_analysis:
            pred = latest.get("prediction", {})
            metrics = latest.get("metrics", {})
            wear_class = pred.get("predicted_class", "moderate")
            condition_label = pred.get("label", "Moderately Worn").upper()
            source_note = f"Loaded from Tread Analyzer: {latest.get('image_label', 'Active Session')}"
        else:
            condition_map = {
                "Good / Low Wear": "good",
                "Moderately Worn": "moderate",
                "Heavily Worn": "worn"
            }
            sel_condition = st.selectbox(
                "Assessed Condition:",
                list(condition_map.keys()),
                index=1
            )
            wear_class = condition_map[sel_condition]
            condition_label = sel_condition.upper()
            metrics = {
                "pattern_integrity_pct": 78.0 if wear_class == "moderate" else (92.0 if wear_class == "good" else 35.0),
                "groove_visibility_pct": 72.0 if wear_class == "moderate" else (88.0 if wear_class == "good" else 30.0),
                "edge_definition_pct": 65.0 if wear_class == "moderate" else (90.0 if wear_class == "good" else 25.0),
                "texture_condition_pct": 74.0 if wear_class == "moderate" else (91.0 if wear_class == "good" else 28.0)
            }
            source_note = "Manual Assessment Selection"

        # Calculate Recommendation
        eval_res = ReplacementAdvisor.evaluate(
            wear_class=wear_class,
            metrics=metrics,
            measured_depth_mm=measured_depth
        )

        # Recommendation Display Card
        st.markdown(f"""
        <div style="background: #23252E; border: 1px solid #41434E; border-left: 4px solid #D8125B; border-radius: 4px; padding: 22px 24px; margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                <div>
                    <div style="font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: #D8125B; font-weight: 700;">
                        CURRENT CONDITION
                    </div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #F2F2F4; margin: 2px 0 10px 0;">
                        {condition_label}
                    </div>
                </div>
                <div style="font-size: 0.76rem; color: #B4B5BC;">
                    {source_note}
                </div>
            </div>

            <div style="border-top: 1px solid #30323D; padding-top: 14px; margin-top: 6px;">
                <div style="font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #B4B5BC; font-weight: 600;">
                    RECOMMENDATION
                </div>
                <div style="font-size: 1.25rem; font-weight: 600; color: #F2F2F4; margin: 4px 0 8px 0;">
                    {eval_res['action']}
                </div>
                <div style="font-size: 0.9rem; color: #B4B5BC; line-height: 1.6;">
                    {eval_res['advice']}
                </div>
                {f'<div style="margin-top: 10px; font-size: 0.82rem; color: #EF246D; font-weight: 600;">NOTICE: {eval_res["depth_note"]}</div>' if eval_res.get('depth_note') else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Database Replacement Options Section
        replacements = ReplacementAdvisor.find_replacements_for_tyre(selected_tyre_id, session)

        st.markdown("""
        <div style="font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; color: #F2F2F4; font-weight: 700; margin-bottom: 12px;">
            REPLACEMENT OPTIONS
        </div>
        """, unsafe_allow_html=True)

        if replacements:
            st.markdown(f"<div style='font-size: 0.85rem; color: #B4B5BC; margin-bottom: 14px;'>Identified {len(replacements)} verified database replacements sharing compatible dimensional specifications:</div>", unsafe_allow_html=True)

            rep_cols = st.columns(min(3, max(1, len(replacements))), gap="medium")
            for idx, rep in enumerate(replacements):
                target_col = rep_cols[idx % len(rep_cols)]
                with target_col:
                    st.markdown(f"""
                    <div style="background: #23252E; border: 1px solid #41434E; border-radius: 4px; padding: 18px; margin-bottom: 16px; height: 100%;">
                        <div style="font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; color: #D8125B; font-weight: 600;">
                            {rep['category']}
                        </div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #F2F2F4; margin: 4px 0;">
                            {rep['manufacturer']} {rep['model']}
                        </div>
                        <div style="font-size: 0.82rem; color: #B4B5BC; margin-bottom: 8px;">
                            Pattern Geometry: {rep['pattern_type']}
                        </div>
                        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; color: #F2F2F4; background: #282A34; padding: 6px 10px; border-radius: 3px; border: 1px solid #41434E; margin-bottom: 8px;">
                            Sizes: {', '.join(rep['shared_sizes'])}
                        </div>
                        <div style="font-size: 0.8rem; color: #B4B5BC; line-height: 1.45;">
                            {rep['description']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="tv-notice">
                <div class="tv-notice-title">NO DIRECT REPLACEMENTS INDEXED</div>
                No secondary replacement models sharing exact dimensional sizes were found in the local catalog.
            </div>
            """, unsafe_allow_html=True)

        # Standard Safety Notice
        st.markdown(f"""
        <div style="margin-top: 24px; padding: 14px 18px; background: #282A34; border-radius: 4px; border: 1px solid #41434E; color: #B4B5BC; font-size: 0.8rem; line-height: 1.5;">
            {eval_res['disclaimer']}
        </div>
        """, unsafe_allow_html=True)

    finally:
        session.close()
