"""
TreadVision: Home Dashboard Page
"""
import streamlit as st
from app.database.connection import get_db_session
from app.database.models import Tyre, Vehicle, Fitment, TreadPattern
from app.ml.inference import WearInferenceEngine


def render_home():
    """Render main landing page with hero banner and 3 major action cards."""
    # Hero Section
    st.markdown("""
    <div class="tv-hero">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <span class="tv-badge tv-badge-cyan" style="margin-bottom: 10px;">Computer Vision & Forensic Intelligence</span>
                <div class="tv-hero-title">TreadVision</div>
                <div class="tv-hero-tagline">See the tread. Understand the tyre.</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 16px; border-radius: 8px; border: 1px solid #334155;">
                <span style="font-size: 0.8rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">ACADEMIC EDITION</span>
            </div>
        </div>
        <div class="tv-hero-sub">
            <strong>TreadVision: Intelligent Tyre Tread Analysis, Forensic Tread Matching and Vehicle Fitment Intelligence Using Image Processing and Machine Learning</strong>
            <br>
            A classical computer vision pipeline combining spatial filtering, CLAHE contrast enhancement, 
            mathematical morphology, GLCM texture analysis, and multi-criteria similarity metrics to assess wear degradation,
            match crime-scene forensic impressions, and identify verified vehicle fitment options.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3 Primary Action Cards
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown("""
        <div class="tv-card">
            <span class="tv-badge tv-badge-cyan">Module 1</span>
            <div class="tv-card-header" style="margin-top: 10px;">TYRE TREAD ANALYZER</div>
            <div class="tv-card-desc">
                Analyze a photograph of an actual physical tyre tread. Evaluates groove integrity,
                edge definition, GLCM texture, and classifies wear stage (Good, Moderate, Worn).
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 Analyze Tread", use_container_width=True, type="primary"):
            st.session_state["nav_selection"] = "analyzer"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="tv-card">
            <span class="tv-badge tv-badge-amber">Module 2</span>
            <div class="tv-card-header" style="margin-top: 10px;">TYRE REPLACEMENT</div>
            <div class="tv-card-desc">
                Receive practical, transparent replacement guidance based on tread analysis metrics
                and manual depth gauge inputs, matched against compatible replacement tyres.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🛡️ Check Replacement", use_container_width=True):
            st.session_state["nav_selection"] = "replacement"
            st.rerun()

    with col3:
        st.markdown("""
        <div class="tv-card">
            <span class="tv-badge tv-badge-emerald">Module 3</span>
            <div class="tv-card-header" style="margin-top: 10px;">TYRE TREAD DATABASE</div>
            <div class="tv-card-desc">
                Explore tyre models, high-resolution tread patterns, technical specifications,
                and verified vehicle fitment pairings across major automotive manufacturers.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📚 Explore Database", use_container_width=True):
            st.session_state["nav_selection"] = "database"
            st.rerun()

    # Prominent Forensic Matcher Callout Card
    st.markdown("---")
    f_col1, f_col2 = st.columns([2.5, 1], gap="large")

    with f_col1:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #111827 0%, #1e1b4b 100%); border: 1px solid #3730a3; border-radius: 12px; padding: 22px 24px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                <span class="tv-badge tv-badge-amber">Forensic Intelligence</span>
                <span style="font-family: 'Chakra Petch', sans-serif; font-size: 1.2rem; font-weight: 700; color: #ffffff;">
                    Forensic Tread Matcher
                </span>
            </div>
            <p style="color: #cbd5e1; font-size: 0.92rem; margin-bottom: 12px; line-height: 1.5;">
                Upload a photographed tyre impression left in mud, soil, sand, or dirt at a crime scene.
                The system extracts geometric tread blocks, groove orientation, and ORB keypoints to perform
                weighted multi-criteria matching against reference database patterns and retrieve probable vehicle applications.
            </p>
            <span style="color: #94a3b8; font-size: 0.82rem; font-style: italic;">
                Documented formula: S = 0.30·S_edge + 0.25·S_texture + 0.25·S_shape + 0.20·S_local
            </span>
        </div>
        """, unsafe_allow_html=True)

    with f_col2:
        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
        if st.button("🎯 Analyze Crime-Scene Impression", use_container_width=True, type="secondary"):
            st.session_state["nav_selection"] = "forensic"
            st.rerun()

    # Live System Telemetry / Database KPIs
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("System Telemetry & Database Index")

    session = get_db_session()
    engine = WearInferenceEngine()

    try:
        tyre_count = session.query(Tyre).count()
        vehicle_count = session.query(Vehicle).count()
        fitment_count = session.query(Fitment).count()
        pattern_count = session.query(TreadPattern).count()
    except Exception:
        tyre_count, vehicle_count, fitment_count, pattern_count = 0, 0, 0, 0
    finally:
        session.close()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="tv-kpi">
            <div class="tv-kpi-val">{tyre_count}</div>
            <div class="tv-kpi-label">Tyre Models Indexed</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        with kpi2:
            st.markdown(f"""
            <div class="tv-kpi">
                <div class="tv-kpi-val">{pattern_count}</div>
                <div class="tv-kpi-label">Reference Patterns</div>
            </div>
            """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="tv-kpi">
            <div class="tv-kpi-val">{fitment_count}</div>
            <div class="tv-kpi-label">Vehicle Fitments</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        ml_status = engine.model_type.upper() if engine.is_model_trained() else "RULE-CV"
        st.markdown(f"""
        <div class="tv-kpi">
            <div class="tv-kpi-val" style="color: #34d399;">{ml_status}</div>
            <div class="tv-kpi-label">Active Wear Model</div>
        </div>
        """, unsafe_allow_html=True)
