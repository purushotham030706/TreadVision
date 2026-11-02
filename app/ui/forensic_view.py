"""
TreadVision: Forensic Tread Matcher UI
Workstation for digital impression processing, multi-criteria similarity matching,
and vehicle fitment retrieval.
"""
from pathlib import Path
import json
import streamlit as st
import numpy as np
import cv2
from PIL import Image

from app.config import FORENSIC_IMPRESSIONS_DIR, FORENSIC_WEIGHTS
from app.database.connection import get_db_session
from app.database.models import Tyre, TreadFeatures
from app.image_processing.forensic_matcher import ForensicMatcher
from app.ui.components import render_header, render_forensic_notice


def render_forensic():
    """Render the Forensic Tread Matcher workstation."""
    render_header(
        "FORENSIC TREAD MATCHER",
        "Compare a photographed tyre impression against reference tread patterns."
    )

    # Workflow Pipeline Sequence Indicator
    st.markdown("""
    <div style="background: #17191C; border: 1px solid #34373B; border-radius: 4px; padding: 12px 16px; margin-bottom: 20px;">
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; font-weight: 600; letter-spacing: 0.08em; color: #F2F2F0; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
            <span><span style="color: #A51F32;">01</span> INPUT</span>
            <span style="color: #34373B;">&mdash;</span>
            <span><span style="color: #A51F32;">02</span> PROCESS</span>
            <span style="color: #34373B;">&mdash;</span>
            <span><span style="color: #A51F32;">03</span> EXTRACT</span>
            <span style="color: #34373B;">&mdash;</span>
            <span><span style="color: #A51F32;">04</span> MATCH</span>
            <span style="color: #34373B;">&mdash;</span>
            <span><span style="color: #A51F32;">05</span> REVIEW</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Conceptual Pipeline Relationship Diagram
    st.markdown("""
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #A7A8A5; background: #1E2024; padding: 8px 14px; border-radius: 3px; border: 1px solid #34373B; margin-bottom: 20px;">
        TRACEABILITY: CRIME-SCENE IMPRESSION &rarr; TREAD PATTERN MATCH &rarr; TYRE DATABASE &rarr; VEHICLE FITMENT
    </div>
    """, unsafe_allow_html=True)

    # Input Section: Crime-Scene Impression Upload or Provided Samples
    col_upload, col_settings = st.columns([1.6, 1], gap="large")

    with col_upload:
        st.markdown("""
        <div style="font-size: 0.76rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F0; font-weight: 700; margin-bottom: 8px;">
            CRIME-SCENE IMPRESSION
        </div>
        """, unsafe_allow_html=True)

        source_mode = st.radio(
            "Evidence Selection:",
            ["Select Verified Evidence Sample", "Upload Custom Impression Photograph"],
            horizontal=True,
            label_visibility="collapsed"
        )

        impression_to_process = None
        evidence_name = ""

        if source_mode == "Select Verified Evidence Sample":
            evidence_samples = {
                "Evidence Sample 01 — Mud Track (Michelin candidate)": FORENSIC_IMPRESSIONS_DIR / "impression_crime_scene_01_mud_michelin.jpg",
                "Evidence Sample 02 — Dry Soil Impression (Continental candidate)": FORENSIC_IMPRESSIONS_DIR / "impression_crime_scene_02_soil_continental.jpg",
                "Evidence Sample 03 — Dirt Surface Mark (Bridgestone candidate)": FORENSIC_IMPRESSIONS_DIR / "impression_crime_scene_03_dirt_bridgestone.jpg",
                "Evidence Sample 04 — Sand Impression (Goodyear candidate)": FORENSIC_IMPRESSIONS_DIR / "impression_crime_scene_04_sand_goodyear.jpg",
                "Evidence Sample 05 — Soil Track Mark (Pirelli candidate)": FORENSIC_IMPRESSIONS_DIR / "impression_crime_scene_05_mud_pirelli.jpg",
            }
            selected_key = st.selectbox(
                "Evidence Sample File:",
                list(evidence_samples.keys()),
                label_visibility="collapsed"
            )
            sample_file = evidence_samples[selected_key]
            if sample_file.exists():
                impression_to_process = str(sample_file)
                evidence_name = selected_key
        else:
            uploaded = st.file_uploader(
                "Upload photographed tyre mark from substrate (mud, sand, soil, dirt):",
                type=["jpg", "jpeg", "png", "bmp", "webp"]
            )
            if uploaded is not None:
                impression_to_process = Image.open(uploaded)
                evidence_name = uploaded.name

            st.markdown("""
            <div style="font-size: 0.76rem; color: #A7A8A5; margin-top: 6px;">
                Supported formats: JPG, PNG<br>
                Recommended: Clear tread geometry, visible impression area, minimal obstruction.
            </div>
            """, unsafe_allow_html=True)

    with col_settings:
        st.markdown("""
        <div style="font-size: 0.76rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F0; font-weight: 700; margin-bottom: 8px;">
            SIMILARITY WEIGHT MATRIX
        </div>
        """, unsafe_allow_html=True)

        w_edge = st.slider("Edge Structure Weight:", 0.10, 0.50, FORENSIC_WEIGHTS["edge"], 0.05)
        w_tex = st.slider("Texture (GLCM) Weight:", 0.10, 0.50, FORENSIC_WEIGHTS["texture"], 0.05)
        w_shape = st.slider("Shape / Moments Weight:", 0.10, 0.50, FORENSIC_WEIGHTS["shape"], 0.05)
        w_local = st.slider("ORB Keypoint Weight:", 0.05, 0.40, FORENSIC_WEIGHTS["local"], 0.05)

        total_w = w_edge + w_tex + w_shape + w_local
        normalized_weights = {
            "edge": w_edge / total_w,
            "texture": w_tex / total_w,
            "shape": w_shape / total_w,
            "local": w_local / total_w
        }

    if impression_to_process is None:
        st.markdown("""
        <div class="tv-notice" style="margin-top: 24px;">
            <div class="tv-notice-title">AWAITING IMPRESSION INPUT</div>
            Select an evidence sample or upload a crime-scene impression to run forensic matching.
        </div>
        """, unsafe_allow_html=True)
        return

    # Execute Forensic Matcher
    matcher = ForensicMatcher(weights=normalized_weights)

    with st.spinner("Processing impression and calculating multi-criteria pattern similarity..."):
        query_res = matcher.process_impression(impression_to_process)

        if not query_res.success:
            st.markdown(f"""
            <div class="tv-notice" style="border-left-color: #A51F32;">
                <div class="tv-notice-title">PROCESSING UNAVAILABLE</div>
                {query_res.error_message}.
            </div>
            """, unsafe_allow_html=True)
            return

        session = get_db_session()
        db_records = []
        try:
            tyres = session.query(Tyre).all()
            for t in tyres:
                if t.features and t.tread_pattern:
                    db_records.append({
                        "id": t.id,
                        "manufacturer": t.manufacturer,
                        "model": t.model,
                        "pattern_name": t.pattern_name,
                        "pattern_type": t.tread_pattern.pattern_type,
                        "symmetry": t.tread_pattern.symmetry,
                        "image_path": t.tread_pattern.image_path,
                        "features": {
                            "edge": json.loads(t.features.edge_features),
                            "texture": json.loads(t.features.texture_features),
                            "shape": json.loads(t.features.shape_features),
                        },
                        "sizes": [s.formatted_size for s in t.sizes],
                        "vehicles": [f"{f.vehicle.manufacturer} {f.vehicle.model}" for f in t.fitments if f.vehicle],
                        "description": t.description
                    })
        finally:
            session.close()

        candidates = matcher.match(query_res, db_records, top_k=5)

    # 1. Forensic Processing Stages View
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; color: #F2F2F0; font-weight: 700; margin-bottom: 12px;">
        IMPRESSION PROCESSING VIEW
    </div>
    """, unsafe_allow_html=True)

    proc_col1, proc_col2, proc_col3, proc_col4, proc_col5 = st.columns(5, gap="small")
    with proc_col1:
        st.markdown("<div style='font-size:0.72rem; font-weight:600; text-transform:uppercase; color:#A7A8A5; margin-bottom:4px;'>ORIGINAL</div>", unsafe_allow_html=True)
        st.image(query_res.stages["Original"], use_container_width=True)
    with proc_col2:
        st.markdown("<div style='font-size:0.72rem; font-weight:600; text-transform:uppercase; color:#A7A8A5; margin-bottom:4px;'>ENHANCED</div>", unsafe_allow_html=True)
        st.image(query_res.stages["Enhanced (CLAHE)"], use_container_width=True)
    with proc_col3:
        st.markdown("<div style='font-size:0.72rem; font-weight:600; text-transform:uppercase; color:#A7A8A5; margin-bottom:4px;'>SEGMENTED</div>", unsafe_allow_html=True)
        st.image(query_res.stages["Segmentation Mask"], use_container_width=True)
    with proc_col4:
        st.markdown("<div style='font-size:0.72rem; font-weight:600; text-transform:uppercase; color:#A7A8A5; margin-bottom:4px;'>EDGE MAP</div>", unsafe_allow_html=True)
        st.image(query_res.stages["Canny Edges"], use_container_width=True)
    with proc_col5:
        st.markdown("<div style='font-size:0.72rem; font-weight:600; text-transform:uppercase; color:#A7A8A5; margin-bottom:4px;'>EXTRACTED PATTERN</div>", unsafe_allow_html=True)
        st.image(query_res.stages["Segmented Tread"], use_container_width=True)

    # 2. Ranked Match Results
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; color: #F2F2F0; font-weight: 700; margin-bottom: 12px;">
        MATCH RESULTS ({len(candidates)} CANDIDATES RANKED)
    </div>
    """, unsafe_allow_html=True)

    if not candidates:
        st.markdown("""
        <div class="tv-notice">
            <div class="tv-notice-title">NO CANDIDATE MATCHES</div>
            No reference patterns are indexed in the active database slice.
        </div>
        """, unsafe_allow_html=True)
        return

    # Top Candidate Primary Highlight
    top = candidates[0]
    st.markdown(f"""
    <div style="background: #17191C; border: 1px solid #34373B; border-left: 4px solid #A51F32; border-radius: 4px; padding: 22px 24px; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; font-weight: 600; color: #A51F32; letter-spacing: 0.08em;">
                    RANK #01 &mdash; HIGHEST SIMILARITY CANDIDATE
                </div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #F2F2F0; margin: 4px 0 2px 0;">
                    {top.manufacturer.upper()} {top.model.upper()}
                </div>
                <div style="font-size: 0.85rem; color: #A7A8A5;">
                    Pattern: {top.pattern_name} &nbsp;|&nbsp; Geometry: {top.pattern_type} &nbsp;|&nbsp; Symmetry: {top.symmetry}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 2.2rem; font-weight: 700; color: #A51F32;">
                    {top.similarity_score:.1f}%
                </div>
                <div style="font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #A7A8A5; font-weight: 600;">
                    OVERALL SIMILARITY
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Side-by-Side Visual Comparison & Feature Breakdown
    side_col1, side_col2, side_col3 = st.columns([1.2, 1.2, 1.6], gap="large")

    with side_col1:
        st.markdown("""
        <div style="font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F0; font-weight: 700; margin-bottom: 6px;">
            EVIDENCE IMPRESSION
        </div>
        """, unsafe_allow_html=True)
        st.image(query_res.stages["Enhanced (CLAHE)"], caption="Contrast-Enhanced Forensic Impression", use_container_width=True)

    with side_col2:
        st.markdown("""
        <div style="font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F0; font-weight: 700; margin-bottom: 6px;">
            REFERENCE MATCH (01)
        </div>
        """, unsafe_allow_html=True)
        if Path(top.reference_image_path).exists():
            st.image(top.reference_image_path, caption=f"{top.manufacturer} {top.model}", use_container_width=True)
        else:
            st.markdown("<div style='background:#1E2024; height:200px; display:flex; align-items:center; justify-content:center; color:#A7A8A5; font-size:0.8rem;'>Reference image unavailable</div>", unsafe_allow_html=True)

    with side_col3:
        st.markdown("""
        <div style="font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F0; font-weight: 700; margin-bottom: 6px;">
            MATCH COMPONENTS
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background: #17191C; border: 1px solid #34373B; border-radius: 4px; padding: 14px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.82rem; border-bottom: 1px solid #1E2024;">
                <span style="color: #A7A8A5;">EDGE SIMILARITY</span>
                <span class="tv-mono" style="font-weight: 600; color: #F2F2F0;">{top.component_scores['edge_similarity']:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.82rem; border-bottom: 1px solid #1E2024;">
                <span style="color: #A7A8A5;">TEXTURE SIMILARITY</span>
                <span class="tv-mono" style="font-weight: 600; color: #F2F2F0;">{top.component_scores['texture_similarity']:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.82rem; border-bottom: 1px solid #1E2024;">
                <span style="color: #A7A8A5;">SHAPE SIMILARITY</span>
                <span class="tv-mono" style="font-weight: 600; color: #F2F2F0;">{top.component_scores['shape_similarity']:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.82rem; border-bottom: 1px solid #1E2024;">
                <span style="color: #A7A8A5;">LOCAL FEATURE MATCH</span>
                <span class="tv-mono" style="font-weight: 600; color: #F2F2F0;">{top.component_scores['local_pattern_similarity']:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 6px 0 2px 0; font-size: 0.85rem; font-weight: 700;">
                <span style="color: #A51F32;">OVERALL SIMILARITY</span>
                <span class="tv-mono" style="color: #A51F32;">{top.similarity_score:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F0; font-weight: 700; margin-bottom: 6px;">
            POSSIBLE VEHICLE APPLICATIONS
        </div>
        """, unsafe_allow_html=True)

        if top.compatible_vehicles:
            veh_list = sorted(list(set(top.compatible_vehicles)))
            st.markdown(f"""
            <div style="background: #17191C; border: 1px solid #34373B; border-radius: 4px; padding: 12px; font-size: 0.84rem; line-height: 1.6; color: #F2F2F0;">
                {', '.join(veh_list)}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:0.8rem; color:#A7A8A5;'>No direct vehicle fitments mapped in current database slice.</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size: 0.72rem; color: #A7A8A5; margin-top: 6px; line-height: 1.4;">
            Possible/common vehicle applications based on indexed factory specifications.
            Does not establish confirmed vehicle presence.
        </div>
        """, unsafe_allow_html=True)

    # Remaining Ranked Candidates
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F0; font-weight: 700; margin-bottom: 10px;">
        SECONDARY RANKED CANDIDATES
    </div>
    """, unsafe_allow_html=True)

    for cand in candidates[1:]:
        with st.expander(f"0{cand.rank}  |  {cand.manufacturer.upper()} {cand.model.upper()}  —  SIMILARITY: {cand.similarity_score:.1f}%", expanded=False):
            c_col_img, c_col_info = st.columns([1, 2.5], gap="medium")
            with c_col_img:
                if Path(cand.reference_image_path).exists():
                    st.image(cand.reference_image_path, caption=cand.pattern_name, use_container_width=True)
            with c_col_info:
                st.markdown(f"**Tyre Model:** {cand.manufacturer} {cand.model}")
                st.markdown(f"**Pattern Geometry:** {cand.pattern_type} | **Symmetry:** {cand.symmetry}")
                st.markdown(f"**Calculated Similarity:** `{cand.similarity_score:.1f}%`")
                st.markdown(f"**Component Scores:** Edge `{cand.component_scores['edge_similarity']:.1f}%` | Texture `{cand.component_scores['texture_similarity']:.1f}%` | Shape `{cand.component_scores['shape_similarity']:.1f}%` | Local ORB `{cand.component_scores['local_pattern_similarity']:.1f}%`")
                if cand.available_sizes:
                    st.markdown(f"**Available Sizes:** `{', '.join(cand.available_sizes)}`")
                if cand.compatible_vehicles:
                    st.markdown(f"**Possible/Common Vehicle Applications:** {', '.join(sorted(list(set(cand.compatible_vehicles))))}")
                st.markdown(f"*{cand.description}*")

    # Mandatory Professional Forensic Notice
    render_forensic_notice()
