"""
TreadVision: Home Dashboard Page
Sophisticated automotive engineering and computer vision platform landing page.
"""
from pathlib import Path
import streamlit as st
from app.config import REFERENCE_PATTERNS_DIR
from app.database.connection import get_db_session
from app.database.models import Tyre, Vehicle, Fitment, TreadPattern


def render_home():
    """Render editorial landing page with three numbered workflows and forensic section."""
    # Header Section
    st.markdown("""
    <div style="margin-bottom: 28px; padding-bottom: 20px; border-bottom: 1px solid #D8D5CE;">
        <div style="font-size: 0.8rem; letter-spacing: 0.14em; text-transform: uppercase; color: #8F1D2C; font-weight: 700; margin-bottom: 6px;">
            TREADVISION
        </div>
        <h1 style="color: #171717; font-size: 2.2rem; font-weight: 700; margin: 0 0 8px 0; letter-spacing: -0.02em;">
            INTELLIGENT TREAD ANALYSIS
        </h1>
        <div style="font-size: 1.05rem; color: #696762; line-height: 1.5; max-width: 820px;">
            Computer vision for tyre condition, tread identification and vehicle fitment.
            Grounded in digital image processing, morphological edge statistics, GLCM texture matrices,
            and multi-criteria forensic similarity matching.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Visual Inspection Spotlight (high-res reference pattern visual)
    ref_img_path = REFERENCE_PATTERNS_DIR / "michelin_primacy4.jpg"
    if ref_img_path.exists():
        v_col1, v_col2 = st.columns([1.4, 2.6], gap="large")
        with v_col1:
            st.image(str(ref_img_path), caption="Reference Pattern: 4-Channel Asymmetric Rib Geometry", use_container_width=True)
        with v_col2:
            st.markdown("""
            <div style="background: #FFFFFF; border: 1px solid #D8D5CE; border-radius: 4px; padding: 22px; height: 100%;">
                <div style="font-size: 0.74rem; letter-spacing: 0.1em; text-transform: uppercase; color: #8F1D2C; font-weight: 600; margin-bottom: 6px;">
                    METHODOLOGICAL ARCHITECTURE
                </div>
                <div style="font-size: 1.15rem; font-weight: 600; color: #171717; margin-bottom: 10px;">
                    Digital Image Processing First
                </div>
                <p style="font-size: 0.88rem; color: #696762; line-height: 1.6; margin-bottom: 12px;">
                    TreadVision extracts 25 deterministic spatial, texture, and shape descriptors directly from 
                    calibrated pixel data before classification. Rather than obscuring degradation inside opaque neural weights,
                    all intermediate representations—including bilateral smoothing, CLAHE contrast grids, Otsu component masks,
                    and Canny boundary fields—remain fully inspectable for academic and forensic verification.
                </p>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; color: #171717; background: #ECEAE5; padding: 8px 12px; border-radius: 3px; border: 1px solid #D8D5CE;">
                    PIPELINE: Preprocessing &rarr; CLAHE &rarr; Segmentation &rarr; Edge Density &rarr; GLCM &rarr; Hu Moments &rarr; Classification
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Three Numbered Editorial Workflow Sections
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown("""
        <div class="tv-card">
            <div class="tv-section-num">01</div>
            <div class="tv-card-header">SECTION 01</div>
            <div class="tv-card-title">TREAD ANALYZER</div>
            <div class="tv-card-desc">
                Assess tread condition using image processing, texture analysis and machine learning.
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
                Evaluate tread condition and explore potential replacement options.
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

    # Dedicated Forensic Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #D8D5CE; border-left: 4px solid #8F1D2C; border-radius: 4px; padding: 24px; margin-top: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
            <div style="max-width: 720px;">
                <div style="font-size: 0.74rem; letter-spacing: 0.1em; text-transform: uppercase; color: #8F1D2C; font-weight: 700; margin-bottom: 4px;">
                    INVESTIGATIVE WORKFLOW
                </div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #171717; margin-bottom: 8px;">
                    FORENSIC TREAD MATCHING
                </div>
                <div style="font-size: 0.92rem; color: #696762; line-height: 1.6;">
                    Compare a photographed tyre impression against reference tread patterns.
                    Evaluates impressions recovered from mud, soil, sand, and asphalt using weighted multi-criteria similarity
                    (edge alignment, GLCM texture, contour moments, and invariant ORB keypoints) to retrieve candidate tyre models
                    and mapped vehicle applications.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    f_col1, f_col2 = st.columns([3, 1])
    with f_col2:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("ANALYZE IMPRESSION", key="btn_home_forensic", use_container_width=True, type="primary"):
            st.session_state["nav_selection"] = "forensic"
            st.rerun()

    # System Overview / Reference Library Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; color: #171717; font-weight: 700; margin-bottom: 12px;">
        REFERENCE LIBRARY
    </div>
    """, unsafe_allow_html=True)

    session = get_db_session()
    try:
        tyre_count = session.query(Tyre).count()
        vehicle_count = session.query(Vehicle).count()
        fitment_count = session.query(Fitment).count()
        pattern_count = session.query(TreadPattern).count()
    except Exception:
        tyre_count, vehicle_count, fitment_count, pattern_count = 0, 0, 0, 0
    finally:
        session.close()

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #D8D5CE; border-radius: 4px; padding: 18px 20px;">
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 600; color: #171717;">
                {pattern_count:,}
            </div>
            <div style="font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #696762; font-weight: 600; margin-top: 4px;">
                REFERENCE PATTERNS
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #D8D5CE; border-radius: 4px; padding: 18px 20px;">
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 600; color: #171717;">
                {vehicle_count:,}
            </div>
            <div style="font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #696762; font-weight: 600; margin-top: 4px;">
                VEHICLE APPLICATIONS
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #D8D5CE; border-radius: 4px; padding: 18px 20px;">
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 600; color: #171717;">
                {fitment_count:,}
            </div>
            <div style="font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #696762; font-weight: 600; margin-top: 4px;">
                FITMENT RELATIONSHIPS
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #D8D5CE; border-radius: 4px; padding: 18px 20px;">
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 600; color: #8F1D2C;">
                3
            </div>
            <div style="font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #696762; font-weight: 600; margin-top: 4px;">
                ANALYSIS MODELS
            </div>
        </div>
        """, unsafe_allow_html=True)
