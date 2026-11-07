"""
TreadVision: Forensic Tread Matcher UI
Workstation for digital impression processing, multi-criteria similarity matching,
database-backed evidence archiving, and vehicle fitment retrieval.
"""
from pathlib import Path
import json
from datetime import date
import streamlit as st
import numpy as np
import cv2
from PIL import Image

from app.config import FORENSIC_IMPRESSIONS_DIR, FORENSIC_WEIGHTS, BASE_DIR
from app.database.connection import get_db_session, verify_database_integrity
from app.database.models import Tyre, TreadFeatures, Investigation, EvidenceImage
from app.database.forensic_repository import (
    list_investigations,
    get_or_create_investigation,
    archive_investigation,
    save_forensic_match_transaction,
    query_investigation_summaries_view,
    get_evidence_details_with_matches,
    get_manufacturer_match_aggregates,
    verify_evidence_storage_integrity,
    cleanup_orphan_evidence_files
)
from app.image_processing.forensic_matcher import ForensicMatcher
from app.ui.components import render_header, render_forensic_notice


def render_forensic():
    """Render the Forensic Tread Matcher workstation with database persistence."""
    render_header(
        "FORENSIC TREAD MATCHER",
        "Compare a photographed tyre impression against reference tread patterns with full DBMS traceability."
    )

    # Workflow Pipeline Sequence Indicator
    st.markdown("""
    <div style="background: #23252E; border: 1px solid #41434E; border-radius: 4px; padding: 12px 16px; margin-bottom: 20px;">
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; font-weight: 600; letter-spacing: 0.08em; color: #F2F2F4; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
            <span><span style="color: #D8125B;">01</span> CASE & INPUT</span>
            <span style="color: #41434E;">&mdash;</span>
            <span><span style="color: #D8125B;">02</span> PROCESS</span>
            <span style="color: #41434E;">&mdash;</span>
            <span><span style="color: #D8125B;">03</span> EXTRACT 31-D</span>
            <span style="color: #41434E;">&mdash;</span>
            <span><span style="color: #D8125B;">04</span> MATCH & PERSIST</span>
            <span style="color: #41434E;">&mdash;</span>
            <span><span style="color: #D8125B;">05</span> REVIEW & FITMENTS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Conceptual Pipeline Relationship Diagram
    st.markdown("""
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #B4B5BC; background: #282A34; padding: 8px 14px; border-radius: 3px; border: 1px solid #41434E; margin-bottom: 20px;">
        TRACEABILITY: INVESTIGATION &rarr; CRIME-SCENE EVIDENCE &rarr; 31-D FEATURES &rarr; MATCH RESULTS &rarr; REFERENCE PATTERNS &rarr; VEHICLE FITMENTS
    </div>
    """, unsafe_allow_html=True)

    # Workstation Navigation Tabs
    tab_match, tab_archive, tab_dbms = st.tabs([
        "FORENSIC MATCHING WORKSTATION",
        "INVESTIGATION CASE ARCHIVE",
        "DBMS ANALYTICAL VIEWS"
    ])

    with tab_match:
        _render_matching_workstation()

    with tab_archive:
        _render_case_archive()

    with tab_dbms:
        _render_dbms_analytics()

    # Mandatory Professional Forensic Notice
    render_forensic_notice()


def _render_matching_workstation():
    """Render the active impression processing and matching interface with DBMS persistence."""
    session = get_db_session()
    investigations = []
    try:
        investigations = list_investigations(session)
    finally:
        session.close()

    # 1. Investigation Case Selection / Registration
    st.markdown("""
    <div style="font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F4; font-weight: 700; margin-bottom: 8px;">
        ACTIVE INVESTIGATION CASE SELECTION
    </div>
    """, unsafe_allow_html=True)

    case_col1, case_col2 = st.columns([1.6, 1], gap="large")

    with case_col1:
        case_mode = st.radio(
            "Case Assignment:",
            ["Associate with Existing Case", "Register New Investigation Case"],
            horizontal=True,
            label_visibility="collapsed"
        )

        selected_case_number = "CASE-2026-001"
        case_title = "Active Scene Investigation"
        case_location = "Field Evidence Collection"
        case_description = "Photographed crime-scene tyre mark."
        matching_inv = None

        if case_mode == "Associate with Existing Case" and investigations:
            case_options = {
                f"{inv.case_number} — {inv.title} ({inv.status})": inv.case_number
                for inv in investigations
            }
            selected_label = st.selectbox(
                "Select Active Case:",
                list(case_options.keys()),
                label_visibility="collapsed"
            )
            selected_case_number = case_options[selected_label]
            matching_inv = next((i for i in investigations if i.case_number == selected_case_number), None)
            if matching_inv:
                case_title = matching_inv.title
                case_location = matching_inv.location or ""
                case_description = matching_inv.description or ""
        else:
            c_sub1, c_sub2 = st.columns(2)
            with c_sub1:
                selected_case_number = st.text_input("Case Number (Unique):", value=f"CASE-2026-00{len(investigations) + 1}")
                case_title = st.text_input("Case Title:", value="Highway Incident Tyre Recovery")
            with c_sub2:
                case_location = st.text_input("Scene Location:", value="Sector 29 Collision Site")
                case_description = st.text_input("Brief Description:", value="Impression lifted from wet roadside shoulder.")

    with case_col2:
        st.markdown(f"""
        <div style="background: #23252E; border: 1px solid #41434E; border-left: 3px solid #D8125B; border-radius: 4px; padding: 12px 14px; font-size: 0.78rem;">
            <div style="font-family: 'IBM Plex Mono', monospace; font-weight: 700; color: #D8125B;">DBMS CASE CONTEXT</div>
            <div style="margin-top: 4px; color: #F2F2F4; font-weight: 600;">{selected_case_number}</div>
            <div style="color: #B4B5BC; margin-top: 2px;">{case_title}</div>
            <div style="color: #B4B5BC; font-size: 0.72rem; margin-top: 4px;">Location: {case_location}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border: none; border-top: 1px solid #282A34; margin: 16px 0 20px 0;'>", unsafe_allow_html=True)

    # 2. Input Section: Crime-Scene Impression Upload or Provided Samples
    col_upload, col_settings = st.columns([1.6, 1], gap="large")

    with col_upload:
        st.markdown("""
        <div style="font-size: 0.76rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F4; font-weight: 700; margin-bottom: 8px;">
            CRIME-SCENE IMPRESSION
        </div>
        """, unsafe_allow_html=True)

        source_options = ["Select Verified Evidence Sample", "Upload Custom Impression Photograph"]
        if matching_inv and matching_inv.evidence_images:
            source_options.append("Re-analyze Existing Case Evidence (New Matching Run)")

        source_mode = st.radio(
            "Evidence Selection:",
            source_options,
            horizontal=True,
            label_visibility="collapsed"
        )

        impression_to_process = None
        evidence_name = ""
        target_evidence_id = None

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
        elif source_mode == "Re-analyze Existing Case Evidence (New Matching Run)":
            ev_items = {
                f"Evidence #{ev.id} — {ev.original_filename} ({ev.image_width}x{ev.image_height})": ev
                for ev in matching_inv.evidence_images
            }
            chosen_label = st.selectbox("Select Evidence Item to Re-match:", list(ev_items.keys()))
            chosen_ev = ev_items[chosen_label]
            target_evidence_id = chosen_ev.id
            evidence_name = chosen_ev.original_filename
            disk_p = BASE_DIR / chosen_ev.image_path
            if disk_p.exists():
                impression_to_process = str(disk_p)
        else:
            uploaded = st.file_uploader(
                "Upload photographed tyre mark from substrate (mud, sand, soil, dirt):",
                type=["jpg", "jpeg", "png", "bmp", "webp"]
            )
            if uploaded is not None:
                impression_to_process = Image.open(uploaded)
                evidence_name = uploaded.name

            st.markdown("""
            <div style="font-size: 0.76rem; color: #B4B5BC; margin-top: 6px;">
                Supported formats: JPG, PNG<br>
                Recommended: Clear tread geometry, visible impression area, minimal obstruction.
            </div>
            """, unsafe_allow_html=True)

    with col_settings:
        st.markdown("""
        <div style="font-size: 0.76rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F4; font-weight: 700; margin-bottom: 8px;">
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
            <div class="tv-notice" style="border-left-color: #D8125B;">
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

        # 3. DBMS Atomic Transaction Persistence
        tx_result = None
        tx_error = None
        try:
            tx_result = save_forensic_match_transaction(
                case_number=selected_case_number,
                image_input=impression_to_process,
                original_filename=evidence_name or "uploaded_impression.jpg",
                pipeline_result=query_res,
                candidates=candidates,
                title=case_title,
                description=case_description,
                location=case_location,
                weights=normalized_weights,
                evidence_id=target_evidence_id,
                notes=f"Forensic run (E={normalized_weights['edge']:.2f}, T={normalized_weights['texture']:.2f}, S={normalized_weights['shape']:.2f}, L={normalized_weights['local']:.2f})"
            )
        except Exception as e:
            tx_error = str(e)

    # Transaction Confirmation Banner
    if tx_result:
        st.markdown(f"""
        <div style="background: #1B2B23; border: 1px solid #2E5C44; border-left: 4px solid #38B000; border-radius: 4px; padding: 12px 16px; margin: 20px 0;">
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; font-weight: 700; color: #38B000; letter-spacing: 0.08em;">
                DBMS TRANSACTION COMMITTED &mdash; ATOMIC PERSISTENCE VERIFIED
            </div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; color: #F2F2F4; margin-top: 4px;">
                Investigation: <span style="color:#FFF; font-weight:600;">{tx_result['case_number']}</span> &nbsp;|&nbsp;
                Evidence ID: <span style="color:#FFF; font-weight:600;">#{tx_result['evidence_id']}</span> &nbsp;|&nbsp;
                Matching Run: <span style="color:#38B000; font-weight:700;">#{tx_result.get('run_number', 1)}</span> (Run ID #{tx_result.get('run_id')}) &nbsp;|&nbsp;
                Ranked Candidates Stored: <span style="color:#FFF; font-weight:600;">{tx_result['matches_stored']}</span><br>
                SHA-256 Checksum: <span style="color:#B4B5BC;">{tx_result['checksum']}</span><br>
                Storage Path: <span style="color:#B4B5BC;">{tx_result['image_path']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif tx_error:
        st.markdown(f"""
        <div class="tv-notice" style="border-left-color: #D8125B;">
            <div class="tv-notice-title">DBMS PERSISTENCE NOTICE</div>
            Database transaction encountered an issue: {tx_error}. Match preview displayed from memory.
        </div>
        """, unsafe_allow_html=True)
    elif tx_error:
        st.markdown(f"""
        <div class="tv-notice" style="border-left-color: #D8125B;">
            <div class="tv-notice-title">DBMS PERSISTENCE NOTICE</div>
            Database transaction encountered an issue: {tx_error}. Match preview displayed from memory.
        </div>
        """, unsafe_allow_html=True)

    # 4. Forensic Processing Stages View
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; color: #F2F2F4; font-weight: 700; margin-bottom: 12px;">
        IMPRESSION PROCESSING VIEW
    </div>
    """, unsafe_allow_html=True)

    proc_col1, proc_col2, proc_col3, proc_col4, proc_col5 = st.columns(5, gap="small")
    with proc_col1:
        st.markdown("<div style='font-size:0.72rem; font-weight:600; text-transform:uppercase; color:#B4B5BC; margin-bottom:4px;'>ORIGINAL</div>", unsafe_allow_html=True)
        st.image(query_res.stages["Original"], use_container_width=True)
    with proc_col2:
        st.markdown("<div style='font-size:0.72rem; font-weight:600; text-transform:uppercase; color:#B4B5BC; margin-bottom:4px;'>ENHANCED</div>", unsafe_allow_html=True)
        st.image(query_res.stages["Enhanced (CLAHE)"], use_container_width=True)
    with proc_col3:
        st.markdown("<div style='font-size:0.72rem; font-weight:600; text-transform:uppercase; color:#B4B5BC; margin-bottom:4px;'>SEGMENTED</div>", unsafe_allow_html=True)
        st.image(query_res.stages["Segmentation Mask"], use_container_width=True)
    with proc_col4:
        st.markdown("<div style='font-size:0.72rem; font-weight:600; text-transform:uppercase; color:#B4B5BC; margin-bottom:4px;'>EDGE MAP</div>", unsafe_allow_html=True)
        st.image(query_res.stages["Canny Edges"], use_container_width=True)
    with proc_col5:
        st.markdown("<div style='font-size:0.72rem; font-weight:600; text-transform:uppercase; color:#B4B5BC; margin-bottom:4px;'>EXTRACTED PATTERN</div>", unsafe_allow_html=True)
        st.image(query_res.stages["Segmented Tread"], use_container_width=True)

    # 5. Ranked Match Results
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; color: #F2F2F4; font-weight: 700; margin-bottom: 12px;">
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
    <div style="background: #23252E; border: 1px solid #41434E; border-left: 4px solid #D8125B; border-radius: 4px; padding: 22px 24px; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; font-weight: 600; color: #D8125B; letter-spacing: 0.08em;">
                    RANK #01 &mdash; HIGHEST SIMILARITY CANDIDATE
                </div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #F2F2F4; margin: 4px 0 2px 0;">
                    {top.manufacturer.upper()} {top.model.upper()}
                </div>
                <div style="font-size: 0.85rem; color: #B4B5BC;">
                    Pattern: {top.pattern_name} &nbsp;|&nbsp; Geometry: {top.pattern_type} &nbsp;|&nbsp; Symmetry: {top.symmetry}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 2.2rem; font-weight: 700; color: #D8125B;">
                    {top.similarity_score:.1f}%
                </div>
                <div style="font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #B4B5BC; font-weight: 600;">
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
        <div style="font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F4; font-weight: 700; margin-bottom: 6px;">
            EVIDENCE IMPRESSION
        </div>
        """, unsafe_allow_html=True)
        st.image(query_res.stages["Enhanced (CLAHE)"], caption="Contrast-Enhanced Forensic Impression", use_container_width=True)

    with side_col2:
        st.markdown("""
        <div style="font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F4; font-weight: 700; margin-bottom: 6px;">
            REFERENCE MATCH (01)
        </div>
        """, unsafe_allow_html=True)
        if Path(top.reference_image_path).exists():
            st.image(top.reference_image_path, caption=f"{top.manufacturer} {top.model}", use_container_width=True)
        else:
            st.markdown("<div style='background:#282A34; height:200px; display:flex; align-items:center; justify-content:center; color:#B4B5BC; font-size:0.8rem;'>Reference image unavailable</div>", unsafe_allow_html=True)

    with side_col3:
        st.markdown("""
        <div style="font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F4; font-weight: 700; margin-bottom: 6px;">
            MATCH COMPONENTS
        </div>
        """, unsafe_allow_html=True)

        local_score_val = top.component_scores.get("local_pattern_similarity")
        local_status = top.component_scores.get("local_feature_status", "AVAILABLE")
        local_score_str = f"{local_score_val:.1f}%" if local_score_val is not None else "N/A"

        st.markdown(f"""
        <div style="background: #23252E; border: 1px solid #41434E; border-radius: 4px; padding: 14px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.82rem; border-bottom: 1px solid #282A34;">
                <span style="color: #B4B5BC;">EDGE SIMILARITY</span>
                <span class="tv-mono" style="font-weight: 600; color: #F2F2F4;">{top.component_scores['edge_similarity']:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.82rem; border-bottom: 1px solid #282A34;">
                <span style="color: #B4B5BC;">TEXTURE SIMILARITY</span>
                <span class="tv-mono" style="font-weight: 600; color: #F2F2F4;">{top.component_scores['texture_similarity']:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.82rem; border-bottom: 1px solid #282A34;">
                <span style="color: #B4B5BC;">SHAPE SIMILARITY</span>
                <span class="tv-mono" style="font-weight: 600; color: #F2F2F4;">{top.component_scores['shape_similarity']:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.82rem; border-bottom: 1px solid #282A34;">
                <span style="color: #B4B5BC;">LOCAL FEATURE MATCH</span>
                <span class="tv-mono" style="font-weight: 600; color: #F2F2F4;">{local_score_str}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 6px 0 2px 0; font-size: 0.85rem; font-weight: 700;">
                <span style="color: #D8125B;">OVERALL SIMILARITY</span>
                <span class="tv-mono" style="color: #D8125B;">{top.similarity_score:.1f}%</span>
            </div>
            {f'<div style="margin-top: 10px; font-size: 0.76rem; color: #D8125B; line-height: 1.4; border-top: 1px solid #41434E; padding-top: 8px;">LOCAL FEATURES: Insufficient keypoints detected &mdash; overall similarity reweighted across global descriptors (edge, texture, shape).</div>' if local_status == 'INSUFFICIENT_LOCAL_FEATURES' else ''}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F4; font-weight: 700; margin-bottom: 6px;">
            POSSIBLE VEHICLE APPLICATIONS
        </div>
        """, unsafe_allow_html=True)

        if top.compatible_vehicles:
            veh_list = sorted(list(set(top.compatible_vehicles)))
            st.markdown(f"""
            <div style="background: #23252E; border: 1px solid #41434E; border-radius: 4px; padding: 12px; font-size: 0.84rem; line-height: 1.6; color: #F2F2F4;">
                {', '.join(veh_list)}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:0.8rem; color:#B4B5BC;'>No direct vehicle fitments mapped in current database slice.</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size: 0.72rem; color: #B4B5BC; margin-top: 6px; line-height: 1.4;">
            Vehicle fitments shown are reference specifications from the seed catalog and represent potential vehicle applications, not confirmed scene attribution.
        </div>
        """, unsafe_allow_html=True)

    # Remaining Ranked Candidates
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F4; font-weight: 700; margin-bottom: 10px;">
        SECONDARY RANKED CANDIDATES
    </div>
    """, unsafe_allow_html=True)

    for cand in candidates[1:]:
        c_local_val = cand.component_scores.get("local_pattern_similarity")
        c_local_str = f"{c_local_val:.1f}%" if c_local_val is not None else "N/A"
        with st.expander(f"0{cand.rank}  |  {cand.manufacturer.upper()} {cand.model.upper()}  —  SIMILARITY: {cand.similarity_score:.1f}%", expanded=False):
            c_col_img, c_col_info = st.columns([1, 2.5], gap="medium")
            with c_col_img:
                if Path(cand.reference_image_path).exists():
                    st.image(cand.reference_image_path, caption=cand.pattern_name, use_container_width=True)
            with c_col_info:
                st.markdown(f"**Tyre Model:** {cand.manufacturer} {cand.model}")
                st.markdown(f"**Pattern Geometry:** {cand.pattern_type} | **Symmetry:** {cand.symmetry}")
                st.markdown(f"**Calculated Overall Similarity:** `{cand.similarity_score:.1f}%`")
                st.markdown(f"**Component Scores:** Edge `{cand.component_scores['edge_similarity']:.1f}%` | Texture `{cand.component_scores['texture_similarity']:.1f}%` | Shape `{cand.component_scores['shape_similarity']:.1f}%` | Local ORB `{c_local_str}`")
                if cand.available_sizes:
                    st.markdown(f"**Available Sizes:** `{', '.join(cand.available_sizes)}`")
                if cand.compatible_vehicles:
                    st.markdown(f"**Possible/Common Vehicle Applications:** {', '.join(sorted(list(set(cand.compatible_vehicles))))}")
                st.markdown(f"*{cand.description}*")
                st.markdown("<div style='font-size:0.72rem; color:#B4B5BC;'>Vehicle fitments shown are reference specifications from the seed catalog and represent potential vehicle applications, not confirmed scene attribution.</div>", unsafe_allow_html=True)


def _render_case_archive():
    """Render the historical investigation browser and evidence retrieval workstation."""
    st.markdown("""
    <div style="font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F4; font-weight: 700; margin-bottom: 8px;">
        INVESTIGATION ARCHIVE & HISTORICAL EVIDENCE RETRIEVAL
    </div>
    <div style="font-size: 0.8rem; color: #B4B5BC; margin-bottom: 16px;">
        Revisit previously recorded investigations, inspect cryptographically verified evidence images, and review persisted candidate matches directly from SQLite.
    </div>
    """, unsafe_allow_html=True)

    session = get_db_session()
    try:
        col_flt1, col_flt2 = st.columns([2, 1])
        with col_flt1:
            show_archived = st.checkbox("Include Archived Cases", value=False)

        investigations = list_investigations(session, include_archived=show_archived)
        if not investigations:
            st.markdown("""
            <div class="tv-notice">
                <div class="tv-notice-title">NO INVESTIGATION RECORDS</div>
                No investigations found matching the current archive filter. Execute a matching session to persist the first case.
            </div>
            """, unsafe_allow_html=True)
            return

        inv_map = {
            f"{'[ARCHIVED] ' if inv.is_archived else ''}{inv.case_number} — {inv.title} ({inv.status})": inv
            for inv in investigations
        }
        sel_inv_label = st.selectbox("Select Investigation to Inspect:", list(inv_map.keys()))
        selected_inv = inv_map[sel_inv_label]

        # Case summary metadata header
        st.markdown(f"""
        <div style="background: #23252E; border: 1px solid #41434E; border-left: 4px solid {'#B4B5BC' if selected_inv.is_archived else '#D8125B'}; border-radius: 4px; padding: 18px 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                <div>
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; font-weight: 600; color: {'#B4B5BC' if selected_inv.is_archived else '#D8125B'};">
                        CASE RECORD: {selected_inv.case_number} {'[ARCHIVED]' if selected_inv.is_archived else ''}
                    </div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #F2F2F4; margin: 4px 0;">
                        {selected_inv.title}
                    </div>
                    <div style="font-size: 0.82rem; color: #B4B5BC;">
                        Location: {selected_inv.location or 'N/A'} &nbsp;|&nbsp; Date: {selected_inv.investigation_date} &nbsp;|&nbsp; Status: {selected_inv.status}
                    </div>
                    <div style="font-size: 0.82rem; color: #F2F2F4; margin-top: 6px;">
                        {selected_inv.description or ''}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #F2F2F4;">
                        {len(selected_inv.evidence_images)}
                    </div>
                    <div style="font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #B4B5BC; font-weight: 600;">
                        EVIDENCE ITEMS
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Archival status control
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn2:
            if selected_inv.is_archived:
                if st.button("Unarchive Investigation", key=f"unarch_{selected_inv.id}"):
                    archive_investigation(session, selected_inv.id, archive=False)
                    st.rerun()
            else:
                if st.button("Soft-Archive Investigation", key=f"arch_{selected_inv.id}"):
                    archive_investigation(session, selected_inv.id, archive=True)
                    st.rerun()

        if not selected_inv.evidence_images:
            st.markdown("""
            <div class="tv-notice">
                <div class="tv-notice-title">NO EVIDENCE ITEMS ASSOCIATED</div>
                No evidence impressions have been linked to this investigation yet.
            </div>
            """, unsafe_allow_html=True)
            return

        # Evidence selector
        evidence_options = {
            f"Evidence #{ev.id} — {ev.original_filename} ({ev.image_format}, {ev.image_width}x{ev.image_height})": ev.id
            for ev in selected_inv.evidence_images
        }
        sel_ev_label = st.selectbox("Select Evidence Item to Inspect:", list(evidence_options.keys()))
        selected_ev_id = evidence_options[sel_ev_label]

        # Retrieve full details via multi-table join
        details = get_evidence_details_with_matches(session, selected_ev_id)
        if not details:
            st.warning("Evidence details could not be retrieved from database.")
            return

        # Matching Run selector if multiple runs exist
        runs = details.get("available_runs", [])
        if len(runs) > 1:
            run_options = {
                f"Run #{r['run_number']} — {r['run_timestamp']} ({r['matching_method']}, {r['match_count']} ranked)": r["id"]
                for r in runs
            }
            chosen_run_lbl = st.selectbox("Select Matching Run Analysis:", list(run_options.keys()))
            chosen_run_id = run_options[chosen_run_lbl]
            if details.get("active_run") and details["active_run"]["id"] != chosen_run_id:
                details = get_evidence_details_with_matches(session, selected_ev_id, run_id=chosen_run_id)

        # Evidence display
        col_img, col_meta = st.columns([1.2, 1.8], gap="large")

        with col_img:
            ev_file = BASE_DIR / details["image_path"]
            if ev_file.exists():
                st.image(str(ev_file), caption=f"Evidence Image #{details['evidence_id']}", use_container_width=True)
            else:
                st.markdown("<div style='background:#282A34; height:200px; display:flex; align-items:center; justify-content:center; color:#B4B5BC;'>Image file not found on disk</div>", unsafe_allow_html=True)

        with col_meta:
            # Cryptographic status badge
            chk_status = details.get("checksum_status", "UNKNOWN")
            if chk_status == "VERIFIED":
                st.markdown("""
                <div style="background: #1B2B23; border: 1px solid #2E5C44; border-left: 3px solid #38B000; border-radius: 3px; padding: 6px 10px; margin-bottom: 10px; font-size: 0.74rem; font-family: 'IBM Plex Mono', monospace; color: #38B000; font-weight: 600;">
                    CRYPTOGRAPHIC INTEGRITY: VERIFIED (SHA-256 MATCH ON DISK FILE)
                </div>
                """, unsafe_allow_html=True)
            elif chk_status == "TAMPER_DETECTED":
                st.markdown("""
                <div style="background: #2D1820; border: 1px solid #5C2230; border-left: 3px solid #D8125B; border-radius: 3px; padding: 6px 10px; margin-bottom: 10px; font-size: 0.74rem; font-family: 'IBM Plex Mono', monospace; color: #D8125B; font-weight: 600;">
                    CRITICAL INTEGRITY ALERT: TAMPER DETECTED (SHA-256 MISMATCH)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #2D1820; border: 1px solid #5C2230; border-left: 3px solid #D8125B; border-radius: 3px; padding: 6px 10px; margin-bottom: 10px; font-size: 0.74rem; font-family: 'IBM Plex Mono', monospace; color: #D8125B; font-weight: 600;">
                    STORAGE ERROR: PHYSICAL EVIDENCE FILE MISSING ON DISK
                </div>
                """, unsafe_allow_html=True)

            active_run_info = ""
            if details.get("active_run"):
                arun = details["active_run"]
                w_str = ""
                if arun.get("weights"):
                    w = arun["weights"]
                    w_str = f" | Weights: E={w.get('edge', 0)*100:.0f}%, T={w.get('texture', 0)*100:.0f}%, S={w.get('shape', 0)*100:.0f}%, L={w.get('local', 0)*100:.0f}%"
                active_run_info = f"<div style='color: #F2F2F4; font-weight: 600; margin-top: 6px;'>Run #{arun['run_number']} ({arun['matching_method']}){w_str}</div>"

            st.markdown(f"""
            <div style="background: #23252E; border: 1px solid #41434E; border-radius: 4px; padding: 14px; font-size: 0.8rem;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-weight: 700; color: #D8125B; margin-bottom: 8px;">
                    CRYPTOGRAPHIC & STORAGE METADATA
                </div>
                <div style="color: #B4B5BC; margin-bottom: 4px;">Original File: <span style="color:#F2F2F4;">{details['original_filename']}</span></div>
                <div style="color: #B4B5BC; margin-bottom: 4px;">Dimensions: <span style="color:#F2F2F4;">{details['image_width']} x {details['image_height']} px</span></div>
                <div style="color: #B4B5BC; margin-bottom: 4px;">Format: <span style="color:#F2F2F4;">{details['image_format']}</span></div>
                <div style="color: #B4B5BC; margin-bottom: 4px;">Relative Path: <span class="tv-mono" style="color:#F2F2F4;">{details['image_path']}</span></div>
                <div style="color: #B4B5BC; margin-bottom: 4px;">Database Checksum: <span class="tv-mono" style="color:#F2F2F4; font-size: 0.72rem;">{details['image_checksum']}</span></div>
                <div style="color: #B4B5BC; margin-bottom: 4px;">Live Disk Checksum: <span class="tv-mono" style="color:#F2F2F4; font-size: 0.72rem;">{details['live_checksum']}</span></div>
                <div style="color: #B4B5BC;">Logged Timestamp: <span style="color:#F2F2F4;">{details['upload_timestamp']}</span></div>
                {active_run_info}
            </div>
            """, unsafe_allow_html=True)

            if details["features"]:
                feats = details["features"]
                vec = feats.get("feature_vector", [])
                st.markdown(f"""
                <div style="background: #23252E; border: 1px solid #41434E; border-radius: 4px; padding: 14px; margin-top: 12px; font-size: 0.8rem;">
                    <div style="font-family: 'IBM Plex Mono', monospace; font-weight: 700; color: #D8125B; margin-bottom: 6px;">
                        PERSISTED FEATURE VECTOR (31-D)
                    </div>
                    <div style="color: #B4B5BC; margin-bottom: 4px;">Method: <span style="color:#F2F2F4;">{feats.get('extraction_method')}</span></div>
                    <div style="color: #B4B5BC; margin-bottom: 6px;">Vector Dimensions: <span style="color:#F2F2F4; font-weight:600;">{len(vec)} floats</span></div>
                    <div class="tv-mono" style="background:#1B1C22; padding:8px; border-radius:3px; font-size:0.68rem; color:#F2F2F4; overflow-x:auto; white-space:nowrap;">
                        [{', '.join(f'{v:.4f}' for v in vec[:10])}... +{len(vec)-10} more]
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Ranked matches from database
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; color: #F2F2F4; font-weight: 700; margin-bottom: 12px;">
            PERSISTED CANDIDATE MATCHES ({len(details['matches'])} RANKED IN DATABASE)
        </div>
        """, unsafe_allow_html=True)

        for match in details["matches"]:
            tyre = match["tyre"]
            scores = match["component_scores"]
            with st.expander(f"Rank #{match['rank']} — {tyre['manufacturer']} {tyre['model']} | Similarity: {match['similarity_score']:.1f}%", expanded=(match['rank'] == 1)):
                m_col1, m_col2 = st.columns([1, 2.5], gap="medium")
                with m_col1:
                    if Path(tyre["image_path"]).exists():
                        st.image(tyre["image_path"], caption=tyre["pattern_name"], use_container_width=True)
                with m_col2:
                    st.markdown(f"**Pattern:** {tyre['pattern_name']} ({tyre['pattern_type']}, {tyre['symmetry']})")
                    st.markdown(f"**Category:** {tyre['category']}")
                    st.markdown(f"**Similarity Score:** `{match['similarity_score']:.1f}%` &nbsp;|&nbsp; **Method:** {match['matching_method']}")
                    if scores:
                        st.markdown(f"**Component Scores:** Edge `{scores.get('edge_similarity', 0.0):.1f}%` | Texture `{scores.get('texture_similarity', 0.0):.1f}%` | Shape `{scores.get('shape_similarity', 0.0):.1f}%`")
                    if tyre["available_sizes"]:
                        st.markdown(f"**Available Sizes:** `{', '.join(tyre['available_sizes'][:6])}`")
                    if tyre["compatible_vehicles"]:
                        st.markdown(f"**Possible Vehicle Applications:** {', '.join(tyre['compatible_vehicles'])}")
                    st.markdown(f"*{tyre['description']}*")

    finally:
        session.close()


def _render_dbms_analytics():
    """Render analytical views, aggregations, and DBMS architectural integrity verification."""
    st.markdown("""
    <div style="font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F4; font-weight: 700; margin-bottom: 8px;">
        DBMS ARCHITECTURE & ANALYTICAL QUERIES
    </div>
    <div style="font-size: 0.8rem; color: #B4B5BC; margin-bottom: 20px;">
        Demonstration of SQL relational views, multi-table joins, GROUP BY aggregate statistics, and foreign key constraint validation.
    </div>
    """, unsafe_allow_html=True)

    session = get_db_session()
    try:
        # 1. SQL View: v_investigation_summaries
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; font-weight: 700; color: #D8125B; margin-bottom: 6px;">
            SQL VIEW: v_investigation_summaries
        </div>
        <div style="font-size: 0.76rem; color: #B4B5BC; margin-bottom: 10px;">
            Precompiled relational view aggregating investigations, evidence items, matching runs, total matches, highest recorded similarity, and top candidate tyre.
        </div>
        """, unsafe_allow_html=True)

        summaries = query_investigation_summaries_view(session)
        if summaries:
            table_rows = []
            for s in summaries:
                table_rows.append({
                    "Case Number": s["case_number"],
                    "Case Title": s["title"],
                    "Status": s["status"],
                    "Archived": "YES" if s.get("is_archived") else "NO",
                    "Date": s["investigation_date"],
                    "Evidence Items": s["evidence_count"],
                    "Matching Runs": s.get("total_matching_runs", 1),
                    "Matches Logged": s["total_matches_recorded"],
                    "Peak Similarity": f"{s['highest_similarity_score']:.1f}%",
                    "Top Pattern Type": s["top_pattern_type"],
                    "Top Candidate Tyre": s["top_candidate_tyre"]
                })
            st.dataframe(table_rows, use_container_width=True)
        else:
            st.info("No investigation summaries currently indexed in view.")

        # 2. Manufacturer Match Aggregate Query
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; font-weight: 700; color: #D8125B; margin-bottom: 6px;">
            SQL AGGREGATION: MANUFACTURER CANDIDATE FREQUENCY
        </div>
        <div style="font-size: 0.76rem; color: #B4B5BC; margin-bottom: 10px;">
            SELECT manufacturer, COUNT(m.id), AVG(m.similarity_score), MAX(m.similarity_score) FROM match_results GROUP BY manufacturer;
        </div>
        """, unsafe_allow_html=True)

        mfg_stats = get_manufacturer_match_aggregates(session)
        if mfg_stats:
            mfg_rows = []
            for m in mfg_stats:
                mfg_rows.append({
                    "Manufacturer": m["manufacturer"],
                    "Top-5 Appearances": m["appearance_count"],
                    "Mean Similarity": f"{m['avg_similarity']:.2f}%",
                    "Peak Similarity": f"{m['max_similarity']:.2f}%"
                })
            st.dataframe(mfg_rows, use_container_width=True)
        else:
            st.info("No match results available yet for manufacturer aggregation.")

        # 3. Storage & Cryptographic Verification Audit
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; font-weight: 700; color: #D8125B; margin-bottom: 6px;">
            STORAGE AUDIT & CRYPTOGRAPHIC SHA-256 VERIFICATION
        </div>
        <div style="font-size: 0.76rem; color: #B4B5BC; margin-bottom: 10px;">
            Active forensic audit comparing EvidenceImage records in SQLite with physical files on disk, detecting bit rot, tampering, and unindexed orphan files.
        </div>
        """, unsafe_allow_html=True)

        storage_audit = verify_evidence_storage_integrity(session)
        sa1, sa2, sa3, sa4 = st.columns(4)
        with sa1:
            st.metric("Evidence Records", storage_audit["total_evidence_records"])
        with sa2:
            st.metric("Cryptographically Verified", storage_audit["verified_intact"])
        with sa3:
            st.metric("Tampered / Missing Files", len(storage_audit["tampered_files"]) + len(storage_audit["missing_files"]))
        with sa4:
            st.metric("Untracked Orphan Files", len(storage_audit["orphan_files"]))

        if storage_audit["orphan_files"]:
            st.warning(f"Detected {len(storage_audit['orphan_files'])} untracked orphan file(s) in evidence storage directory.")
            if st.button("Purge Untracked Orphan Files"):
                removed_count = cleanup_orphan_evidence_files(session)
                st.success(f"Purged {removed_count} orphan file(s) from storage.")
                st.rerun()

        # 4. Database Schema Integrity Status
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; font-weight: 700; color: #D8125B; margin-bottom: 6px;">
            RELATIONAL INTEGRITY & CONSTRAINT STATUS
        </div>
        """, unsafe_allow_html=True)

        integrity = verify_database_integrity(session_instance=session)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Tyres", integrity["counts"].get("tyres", 0))
        with c2:
            st.metric("Investigations", integrity["counts"].get("investigations", 0))
        with c3:
            st.metric("Evidence Records", integrity["counts"].get("evidence_images", 0))
        with c4:
            st.metric("Match Rankings", integrity["counts"].get("match_results", 0))

        st.markdown(f"""
        <div style="background: #23252E; border: 1px solid #41434E; border-radius: 4px; padding: 14px; margin-top: 14px; font-size: 0.78rem;">
            <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #282A34;">
                <span style="color: #B4B5BC;">SQLite Foreign Key Enforcement (PRAGMA foreign_keys = ON):</span>
                <span style="color: #38B000; font-weight: 700;">ACTIVE</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #282A34;">
                <span style="color: #B4B5BC;">Foreign Key Check Status:</span>
                <span style="color: {'#38B000' if integrity['fk_check_passed'] else '#D8125B'}; font-weight: 700;">
                    {'PASSED (0 VIOLATIONS)' if integrity['fk_check_passed'] else 'VIOLATIONS DETECTED'}
                </span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #282A34;">
                <span style="color: #B4B5BC;">Database Tables Verified:</span>
                <span class="tv-mono" style="color: #F2F2F4;">{len(integrity['tables_checked'])} Tables Present</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                <span style="color: #B4B5BC;">Relational View (v_investigation_summaries):</span>
                <span style="color: #38B000; font-weight: 700;">COMPILED</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    finally:
        session.close()
