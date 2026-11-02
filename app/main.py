"""
TreadVision: Main Application Entrypoint
Academic & Forensic Platform for Tyre Tread Analysis
"""
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st

# Streamlit Page Configuration - Zero emojis
st.set_page_config(
    page_title="TreadVision",
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
        with st.spinner("Initializing reference database and calibrated assets..."):
            generate_all_demo_assets()
            seed_database(force_reseed=False)


def main():
    # Apply warm off-white automotive engineering theme
    apply_theme()

    # Self-healing database initialization
    ensure_system_initialized()

    # Session State Navigation
    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = "home"

    # Sidebar Navigation & Branding
    with st.sidebar:
        st.markdown("""
        <div style="padding-bottom: 1.25rem; margin-bottom: 1.5rem; border-bottom: 1px solid #D8D5CE;">
            <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 1.15rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #171717;">
                TREADVISION
            </div>
            <div style="font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: #8F1D2C; font-weight: 600; margin-top: 2px;">
                TYRE INTELLIGENCE PLATFORM
            </div>
        </div>
        """, unsafe_allow_html=True)

        pages = {
            "home": "OVERVIEW",
            "analyzer": "TREAD ANALYZER",
            "replacement": "TYRE REPLACEMENT",
            "database": "TREAD DATABASE",
            "forensic": "FORENSIC MATCHER",
        }

        current_key = st.session_state["nav_selection"]
        if current_key not in pages:
            current_key = "home"

        selected_page = st.radio(
            "SELECT WORKFLOW:",
            list(pages.keys()),
            format_func=lambda k: pages[k],
            index=list(pages.keys()).index(current_key),
            label_visibility="visible"
        )

        if selected_page != current_key:
            st.session_state["nav_selection"] = selected_page
            st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.78rem; color: #696762; line-height: 1.6;">
            <div style="font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #171717; margin-bottom: 6px;">
                TECHNICAL FOUNDATIONS
            </div>
            Digital Image Processing<br>
            Bilateral & CLAHE Filtering<br>
            Canny & Sobel Derivatives<br>
            GLCM Texture Descriptors<br>
            Hu Invariant Moments<br>
            Stratified Group Splitting<br>
            Multi-Criteria Similarity
        </div>
        <div style="margin-top: 18px; padding-top: 12px; border-top: 1px solid #D8D5CE; font-size: 0.74rem; color: #696762;">
            Repository: <span class="tv-mono" style="color: #171717;">TreadVision</span><br>
            Evaluation: <span class="tv-mono">Validation Set</span>
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
