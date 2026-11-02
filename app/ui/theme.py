"""
TreadVision: Automotive Engineering & Scientific UI Theme
Color Palette:
  Main Background:    #F4F3EF (Warm off-white)
  Primary Surface:    #FFFFFF
  Secondary Surface:  #ECEAE5
  Primary Text:       #171717
  Secondary Text:     #696762
  Borders:            #D8D5CE
  Primary Accent:     #8F1D2C (Deep automotive red, used with restraint)
  Dark Technical:     #1E1E1C
"""
import streamlit as st


def apply_theme():
    """Inject sophisticated automotive engineering CSS stylesheet."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Color System & Typography */
    :root {
        --bg-main: #F4F3EF;
        --surface-primary: #FFFFFF;
        --surface-secondary: #ECEAE5;
        --text-primary: #171717;
        --text-secondary: #696762;
        --border-color: #D8D5CE;
        --accent-red: #8F1D2C;
        --surface-dark: #1E1E1C;
    }

    /* Base Body & App Container */
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'IBM Plex Sans', 'Inter', -apple-system, sans-serif !important;
    }

    /* Hide Default Streamlit Chrome & Decoration */
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    header[data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
    }
    .stDeployButton { display: none !important; }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }

    h1 {
        font-size: 2.1rem !important;
        line-height: 1.25 !important;
        margin-bottom: 0.35rem !important;
    }

    h2 {
        font-size: 1.45rem !important;
        line-height: 1.3 !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.5rem !important;
    }

    h3 {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.4rem !important;
    }

    p, span, label, div {
        color: var(--text-primary);
        font-size: 0.93rem;
    }

    code, pre, .tv-mono {
        font-family: 'IBM Plex Mono', monospace !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--surface-primary) !important;
        border-right: 1px solid var(--border-color) !important;
        padding-top: 1.5rem !important;
    }

    section[data-testid="stSidebar"] div.block-container {
        padding-top: 1.5rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    /* Header Nav & Top Banner */
    .tv-masthead {
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 1.25rem;
        margin-bottom: 2rem;
    }

    .tv-masthead-title {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-primary);
        margin: 0;
    }

    .tv-masthead-sub {
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent-red);
        font-weight: 600;
        margin-top: 0.2rem;
    }

    /* Editorial Cards */
    .tv-card {
        background-color: var(--surface-primary);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 24px;
        margin-bottom: 16px;
    }

    .tv-card-header {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin-bottom: 8px;
    }

    .tv-card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 8px;
    }

    .tv-card-desc {
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.55;
        margin-bottom: 18px;
    }

    /* Dark Technical Surface (for imaging stages & raw technical inspect) */
    .tv-dark-panel {
        background-color: var(--surface-dark);
        border: 1px solid #333330;
        border-radius: 4px;
        padding: 20px;
        color: #ECEAE5;
    }

    .tv-dark-panel h1, .tv-dark-panel h2, .tv-dark-panel h3, .tv-dark-panel h4 {
        color: #FFFFFF !important;
    }

    .tv-dark-panel p, .tv-dark-panel span {
        color: #C5C3BC !important;
    }

    /* Technical Numbered Workflows */
    .tv-section-num {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--accent-red);
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    /* Technical Badges & Tags */
    .tv-tag {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 2px 8px;
        border-radius: 3px;
        background: var(--surface-secondary);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
    }

    .tv-tag-accent {
        border-color: var(--accent-red);
        color: var(--accent-red);
        background: rgba(143, 29, 44, 0.05);
    }

    .tv-tag-dark {
        background: #2A2A28;
        border-color: #444440;
        color: #FFFFFF;
    }

    /* Clean Technical Tables */
    .tv-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
        background: var(--surface-primary);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        overflow: hidden;
    }

    .tv-table th {
        text-align: left;
        padding: 10px 14px;
        background: var(--surface-secondary);
        font-size: 0.76rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-secondary);
        border-bottom: 1px solid var(--border-color);
        font-weight: 600;
    }

    .tv-table td {
        padding: 10px 14px;
        border-bottom: 1px solid var(--border-color);
        color: var(--text-primary);
    }

    .tv-table tr:last-child td {
        border-bottom: none;
    }

    .tv-table td.num {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 500;
    }

    /* Scientific / Forensic Disclaimer Notice */
    .tv-notice {
        background-color: var(--surface-secondary);
        border-left: 3px solid var(--accent-red);
        border-top: 1px solid var(--border-color);
        border-right: 1px solid var(--border-color);
        border-bottom: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 14px 18px;
        margin: 20px 0;
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }

    .tv-notice-title {
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--text-primary);
        margin-bottom: 4px;
        font-size: 0.8rem;
    }

    /* Streamlit Widget Overrides */
    div[data-testid="stRadio"] > label {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        color: var(--text-secondary) !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label {
        background: transparent !important;
    }

    /* Primary & Standard Buttons */
    button[kind="primary"], .stButton > button[kind="primary"] {
        background-color: var(--accent-red) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--accent-red) !important;
        border-radius: 4px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        padding: 8px 18px !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }

    button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
        background-color: #741623 !important;
        border-color: #741623 !important;
    }

    button[kind="secondary"], .stButton > button[kind="secondary"], .stButton > button {
        background-color: var(--surface-primary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 4px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        padding: 8px 18px !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }

    button[kind="secondary"]:hover, .stButton > button:hover {
        background-color: var(--surface-secondary) !important;
        border-color: #B5B2AA !important;
    }

    /* Inputs, Selectboxes, File Upload */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: var(--surface-primary) !important;
        border-color: var(--border-color) !important;
        border-radius: 4px !important;
        color: var(--text-primary) !important;
    }

    div[data-testid="stFileUploader"] {
        background-color: var(--surface-primary) !important;
        border: 1px dashed var(--border-color) !important;
        border-radius: 4px !important;
        padding: 16px !important;
    }

    div[data-testid="stFileUploader"]:hover {
        border-color: var(--accent-red) !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        color: var(--text-secondary) !important;
        background-color: transparent !important;
        border: none !important;
        padding: 8px 16px !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent-red) !important;
        border-bottom: 2px solid var(--accent-red) !important;
    }

    /* Metric Display */
    div[data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.74rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
    }

    /* Progress bar */
    div[data-testid="stProgressBar"] > div > div {
        background-color: var(--accent-red) !important;
    }

    /* Clean Technical Dividers */
    hr {
        border: none !important;
        border-top: 1px solid var(--border-color) !important;
        margin: 24px 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
