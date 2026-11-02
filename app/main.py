"""
TreadVision: Main Streamlit Application Entrypoint
Full Academic Title:
TreadVision: Intelligent Tyre Tread Analysis, Forensic Tread Matching and Vehicle Fitment Intelligence
Using Image Processing and Machine Learning
"""
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st

# Streamlit Page Configuration
st.set_page_config(
    page_title="TreadVision",
    page_icon="🛞",
    layout="wide",
    initial_sidebar_state="expanded"
)

from app.config import DATABASE_PATH
from app.ui.theme import apply_theme
from app.ui.home import render_home
from app.ui.analyzer import render_analyzer
from app.ui.replacement import render_replacement
from app.ui.database_view import render_database
from app.ui.forensic_view import render_forensic
from app.database.connection import init_db
from app.database.seed_data import seed_database
from scripts.generate_demo_assets import generate_all_demo_assets


def ensure_system_initialized():
    """Ensure database and demo assets exist on startup."""
    if not DATABASE_PATH.exists():
        with st.spinner("Initializing TreadVision database and demo assets..."):
            generate_all_demo_assets()
            seed_database(force_reseed=False)


def main():
    # Apply modern automotive dark theme
    apply_theme()

    # Self-healing database initialization
    ensure_system_initialized()

    # Session State Navigation
    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = "home"

    # Sidebar Navigation & Branding
    with st.sidebar:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <span style="font-size: 2rem;">🛞</span>
            <div>
                <div style="font-family: 'Chakra Petch', sans-serif; font-size: 1.5rem; font-weight: 700; color: #ffffff; letter-spacing: 0.5px;">
                    TreadVision
                </div>
                <div style="color: #38bdf8; font-size: 0.78rem; font-weight: 500;">
                    Vision & Forensic Intelligence
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<p style='font-size: 0.78rem; color: #64748b; text-transform: uppercase; font-weight: 600; margin-bottom: 6px;'>NAVIGATION</p>", unsafe_allow_html=True)

        pages = {
            "home": ("🏠 Dashboard Home", "home"),
            "analyzer": ("🔍 Module 1: Tread Analyzer", "analyzer"),
            "replacement": ("🛡️ Module 2: Tyre Replacement", "replacement"),
            "database": ("📚 Module 3: Tyre Database", "database"),
            "forensic": ("🔬 Forensic Tread Matcher", "forensic"),
        }

        current_key = st.session_state["nav_selection"]
        selected_page = st.radio(
            "Go to:",
            list(pages.keys()),
            format_func=lambda k: pages[k][0],
            index=list(pages.keys()).index(current_key) if current_key in pages else 0,
            label_visibility="collapsed"
        )

        if selected_page != current_key:
            st.session_state["nav_selection"] = selected_page
            st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.5;">
            <strong style="color: #cbd5e1;">Project Foundation:</strong><br>
            • Classical Digital Image Processing<br>
            • Canny / Sobel Edge Statistics<br>
            • Gray-Level Co-occurrence (GLCM)<br>
            • Contour Moments & Hu Descriptors<br>
            • Multi-Criteria Similarity Metric<br>
            • Anti-Leakage Physical Tyre Grouping
        </div>
        <br>
        <div style="font-size: 0.72rem; color: #64748b; border-top: 1px solid #1e293b; padding-top: 10px;">
            Academic & Forensic Prototype<br>
            GitHub: <code>TreadVision</code>
        </div>
        """, unsafe_allow_html=True)

    # Route to Selected Page
    nav = st.session_state["nav_selection"]
    if nav == "home":
        render_home()
    elif nav == "analyzer":
        render_analyzer()
    elif nav == "replacement":
        render_replacement()
    elif nav == "database":
        render_database()
    elif nav == "forensic":
        render_forensic()
    else:
        render_home()


if __name__ == "__main__":
    main()
