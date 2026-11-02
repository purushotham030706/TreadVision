"""
TreadVision: Forensic Tread Matcher UI
Compares photographed crime-scene impressions against known reference patterns
and retrieves possible vehicle applications.
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
    """Render the Forensic Tread Matcher interface."""
    render_header(
        "Forensic Tread Matcher",
        "Compare a photographed crime-scene tyre impression against reference patterns to identify probable tyre models and vehicle applications."
    )

    # Mandatory Forensic Disclaimer Callout
    render_forensic_notice()

    # Input Section: Crime-Scene Impression Upload or Provided Samples
    col_upload, col_settings = st.columns([2, 1], gap="medium")

    with col_upload:
        source_mode = st.radio(
            "Select Crime-Scene Impression:",
            ["Choose Evidence Sample", "Upload Custom Impression Photograph"],
            horizontal=True
        )

        impression_to_process = None
        evidence_name = ""

        if source_mode == "Choose Evidence Sample":
            evidence_samples = {
                "Evidence Sample 01 — Wet Mud Track (Michelin Primacy candidate)": FORENSIC_IMPRESSIONS_DIR / "impression_crime_scene_01_mud_michelin.jpg",
                "Evidence Sample 02 — Dry Soil Impression (Continental candidate)": FORENSIC_IMPRESSIONS_DIR / "impression_crime_scene_02_soil_continental.jpg",
                "Evidence Sample 03 — Dirt Surface Impression (Bridgestone candidate)": FORENSIC_IMPRESSIONS_DIR / "impression_crime_scene_03_dirt_bridgestone.jpg",
                "Evidence Sample 04 — Sand Dune Mark (Goodyear candidate)": FORENSIC_IMPRESSIONS_DIR / "impression_crime_scene_04_sand_goodyear.jpg",
                "Evidence Sample 05 — Soil Track Mark (Pirelli candidate)": FORENSIC_IMPRESSIONS_DIR / "impression_crime_scene_05_mud_pirelli.jpg",
            }
            selected_sample_key = st.selectbox("Select evidence file:", list(evidence_samples.keys()))
            sample_file = evidence_samples[selected_sample_key]
            if sample_file.exists():
                impression_to_process = str(sample_file)
                evidence_name = selected_sample_key
        else:
            uploaded = st.file_uploader(
                "Upload photographed tyre mark from crime-scene surface (mud, sand, soil, dirt):",
                type=["jpg", "jpeg", "png", "bmp", "webp"]
            )
            if uploaded is not None:
                impression_to_process = Image.open(uploaded)
                evidence_name = uploaded.name

    with col_settings:
        st.markdown("<div style='font-weight:600; color:#38bdf8; margin-bottom:8px;'>⚖️ Similarity Weight Matrix</div>", unsafe_allow_html=True)
        w_edge = st.slider("Edge Structure Weight:", 0.10, 0.50, FORENSIC_WEIGHTS["edge"], 0.05)
        w_tex = st.slider("Texture (GLCM) Weight:", 0.10, 0.50, FORENSIC_WEIGHTS["texture"], 0.05)
        w_shape = st.slider("Shape / Hu Weight:", 0.10, 0.50, FORENSIC_WEIGHTS["shape"], 0.05)
        w_local = st.slider("ORB Keypoint Weight:", 0.05, 0.40, FORENSIC_WEIGHTS["local"], 0.05)

        # Normalize weights so they sum to 1.0
        total_w = w_edge + w_tex + w_shape + w_local
        normalized_weights = {
            "edge": w_edge / total_w,
            "texture": w_tex / total_w,
            "shape": w_shape / total_w,
            "local": w_local / total_w
        }

    if impression_to_process is None:
        st.info("👆 Please select an evidence sample or upload a crime-scene impression to run matching.")
        return

    # Initialize Forensic Matcher
    matcher = ForensicMatcher(weights=normalized_weights)

    with st.spinner("Analyzing crime-scene impression and querying reference pattern database..."):
        # Process impression
        query_res = matcher.process_impression(impression_to_process)

        if not query_res.success:
            st.error(f"Impression Processing Error: {query_res.error_message}")
            return

        # Fetch reference database tyres and precomputed features
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

        # Perform matching
        candidates = matcher.match(query_res, db_records, top_k=5)

    # 1. Processing Preview Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Impression Digital Image Processing Stages")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    with p_col1:
        st.image(query_res.stages["Original"], caption="1. Photographed Impression", use_container_width=True)
    with p_col2:
        st.image(query_res.stages["Enhanced (CLAHE)"], caption="2. CLAHE Contrast Enhanced", use_container_width=True)
    with p_col3:
        st.image(query_res.stages["Segmentation Mask"], caption="3. Adaptive Threshold Mask", use_container_width=True)
    with p_col4:
        st.image(query_res.stages["Canny Edges"], caption="4. Extracted Tread Edges", use_container_width=True)

    # 2. Forensic Match Results
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(f"Probable Tread Pattern Candidates ({len(candidates)} Ranked Matches)")

    if not candidates:
        st.warning("No reference patterns available in database for matching.")
        return

    # Top Candidate Highlight
    top = candidates[0]
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #111827 0%, #172554 100%); border: 2px solid #38bdf8; border-radius: 14px; padding: 24px; margin-bottom: 24px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
            <div>
                <span class="tv-badge tv-badge-cyan">#1 HIGHEST SIMILARITY CANDIDATE</span>
                <h2 style="color: #ffffff; margin: 6px 0 0 0;">{top.manufacturer} {top.model}</h2>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">Pattern: {top.pattern_name} ({top.pattern_type})</div>
            </div>
            <div style="text-align: right;">
                <div style="font-family:'Chakra Petch',sans-serif; font-size: 2.8rem; font-weight:700; color: #38bdf8;">
                    {top.similarity_score}%
                </div>
                <span class="tv-badge tv-badge-emerald">{top.confidence_level}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Side-by-side Visual Comparison of Top Candidate
    st.markdown("#### Side-by-Side Visual Comparison")
    side_col1, side_col2, side_col3 = st.columns([1.2, 1.2, 1.6], gap="medium")

    with side_col1:
        st.markdown("**Uploaded Evidence Impression**")
        st.image(query_res.stages["Enhanced (CLAHE)"], use_container_width=True, caption=evidence_name)

    with side_col2:
        st.markdown(f"**Database Reference Match (#{top.rank})**")
        if Path(top.reference_image_path).exists():
            st.image(top.reference_image_path, use_container_width=True, caption=f"{top.manufacturer} {top.model}")
        else:
            st.write("Pattern image preview unavailable")

    with side_col3:
        st.markdown("**Multi-Criteria Component Similarity**")
        st.write(f"- Edge Structure: **{top.component_scores['edge_similarity']}%** (weight {normalized_weights['edge']*100:.0f}%)")
        st.write(f"- GLCM Texture: **{top.component_scores['texture_similarity']}%** (weight {normalized_weights['texture']*100:.0f}%)")
        st.write(f"- Shape / Moments: **{top.component_scores['shape_similarity']}%** (weight {normalized_weights['shape']*100:.0f}%)")
        st.write(f"- ORB Keypoints: **{top.component_scores['local_pattern_similarity']}%** (weight {normalized_weights['local']*100:.0f}%)")

        st.markdown("---")
        st.markdown("**Possible / Common Vehicle Applications:**")
        if top.compatible_vehicles:
            for veh in sorted(list(set(top.compatible_vehicles))):
                st.markdown(f"- 🚗 *{veh}*")
        else:
            st.write("No vehicles indexed for this tyre.")
        st.caption("Notice: Listed models represent known OEM/aftermarket fitments for this tyre pattern, not definitive proof of suspect vehicle presence.")

    # All Ranked Candidates
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### All Top Candidate Matches")

    for cand in candidates:
        with st.expander(f"Rank #{cand.rank}: {cand.manufacturer} {cand.model} — Similarity: {cand.similarity_score}% ({cand.confidence_level})", expanded=(cand.rank == 1)):
            c_c1, c_c2 = st.columns([1, 2])
            with c_c1:
                if Path(cand.reference_image_path).exists():
                    st.image(cand.reference_image_path, caption=cand.pattern_name, use_container_width=True)
            with c_c2:
                st.markdown(f"**Tyre Model:** {cand.manufacturer} {cand.model}")
                st.markdown(f"**Pattern Type:** {cand.pattern_type} | **Symmetry:** {cand.symmetry}")
                st.markdown(f"**Similarity Score:** `{cand.similarity_score}%` | **Confidence:** `{cand.confidence_level}`")
                st.markdown(f"**Extracted Scores:** Edge `{cand.component_scores['edge_similarity']}%`, Texture `{cand.component_scores['texture_similarity']}%`, Shape `{cand.component_scores['shape_similarity']}%`, Local `{cand.component_scores['local_pattern_similarity']}%`")
                if cand.available_sizes:
                    st.markdown(f"**Available Sizes:** `{', '.join(cand.available_sizes)}`")
                if cand.compatible_vehicles:
                    st.markdown(f"**Possible/Common Vehicle Applications:** {', '.join(set(cand.compatible_vehicles))}")
                st.markdown(f"*{cand.description}*")
