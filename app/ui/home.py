"""
TreadVision: Home Dashboard Page
Concise engineering overview and primary workflow navigation.
"""
import streamlit as st


def render_home():
    """Render concise, dark engineering overview page without decorative images or redundant library counts."""
    # Top Header
    st.markdown("""
    <div style="margin-bottom: 28px; padding-bottom: 18px; border-bottom: 1px solid #34373B;">
        <div style="font-size: 0.76rem; letter-spacing: 0.14em; text-transform: uppercase; color: #A51F32; font-weight: 700; margin-bottom: 6px;">
            TREADVISION
        </div>
        <h1 style="color: #F2F2F0; font-size: 2.1rem; font-weight: 700; margin: 0 0 8px 0; letter-spacing: -0.02em;">
            INTELLIGENT TREAD ANALYSIS
        </h1>
        <div style="font-size: 1rem; color: #A7A8A5; line-height: 1.5; max-width: 820px;">
            Computer vision for tyre condition, tread identification and vehicle fitment.
            Grounded in digital image processing, morphological edge statistics, GLCM texture matrices,
            and multi-criteria forensic similarity matching.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Workflow Architecture Introduction
    st.markdown("""
    <div style="background: #17191C; border: 1px solid #34373B; border-radius: 4px; padding: 20px 24px; margin-bottom: 24px;">
        <div style="font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: #A51F32; font-weight: 700; margin-bottom: 6px;">
            SYSTEM ARCHITECTURE
        </div>
        <div style="font-size: 1.1rem; font-weight: 600; color: #F2F2F0; margin-bottom: 8px;">
            Digital Image Processing & Pattern Analysis
        </div>
        <p style="font-size: 0.88rem; color: #A7A8A5; line-height: 1.6; margin-bottom: 10px;">
            TreadVision extracts 25 deterministic spatial, texture, and shape descriptors directly from 
            calibrated pixel data before classification. All intermediate representations—including bilateral smoothing, 
            CLAHE contrast grids, Otsu component masks, and Canny boundary fields—remain fully inspectable for academic 
            and forensic verification.
        </p>
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; color: #F2F2F0; background: #1E2024; padding: 8px 12px; border-radius: 3px; border: 1px solid #34373B;">
            PIPELINE: Input &rarr; Preprocessing &rarr; CLAHE &rarr; Segmentation &rarr; Edge Density &rarr; GLCM &rarr; Hu Moments &rarr; Classification
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Three Primary Workflows
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown("""
        <div class="tv-card">
            <div class="tv-section-num">01</div>
            <div class="tv-card-header">SECTION 01</div>
            <div class="tv-card-title">TREAD ANALYZER</div>
            <div class="tv-card-desc">
                Assess tyre tread condition using image processing, feature analysis and machine learning.
                Measures groove integrity, edge definition, and surface homogeneity.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("ANALYZE TREAD", key="btn_home_analyzer", use_container_width=True, type="primary"):
            st.session_state["nav_selection"] = "analyzer"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="tv-card">
            <div class="tv-section-num">02</div>
            <div class="tv-card-header">SECTION 02</div>
            <div class="tv-card-title">TYRE REPLACEMENT</div>
            <div class="tv-card-desc">
                Evaluate tyre condition and explore replacement options.
                Benchmarked against reference depth safety thresholds and database sizes.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("VIEW REPLACEMENT", key="btn_home_replacement", use_container_width=True):
            st.session_state["nav_selection"] = "replacement"
            st.rerun()

    with col3:
        st.markdown("""
        <div class="tv-card">
            <div class="tv-section-num">03</div>
            <div class="tv-card-header">SECTION 03</div>
            <div class="tv-card-title">TREAD DATABASE</div>
            <div class="tv-card-desc">
                Explore tread patterns, specifications and vehicle applications.
                Search manufacturer catalogs, available rim diameters, and verified fitments.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("EXPLORE DATABASE", key="btn_home_database", use_container_width=True):
            st.session_state["nav_selection"] = "database"
            st.rerun()

    # Dedicated Forensic Workflow Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: #17191C; border: 1px solid #34373B; border-left: 4px solid #A51F32; border-radius: 4px; padding: 22px 24px;">
        <div style="font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: #A51F32; font-weight: 700; margin-bottom: 4px;">
            INVESTIGATIVE WORKFLOW
        </div>
        <div style="font-size: 1.25rem; font-weight: 700; color: #F2F2F0; margin-bottom: 8px;">
            FORENSIC TREAD MATCHING
        </div>
        <div style="font-size: 0.9rem; color: #A7A8A5; line-height: 1.6; margin-bottom: 16px;">
            Compare photographed tread impressions against reference tread patterns and associated vehicle applications.
            Evaluates impressions recovered from mud, soil, sand, and asphalt using weighted multi-criteria similarity
            (edge alignment, GLCM texture, contour moments, and invariant ORB keypoints) to retrieve candidate tyre models
            and mapped vehicle applications.
        </div>
    </div>
    """, unsafe_allow_html=True)

    f_col1, f_col2 = st.columns([2.5, 1])
    with f_col2:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("ANALYZE IMPRESSION", key="btn_home_forensic", use_container_width=True, type="primary"):
            st.session_state["nav_selection"] = "forensic"
            st.rerun()
