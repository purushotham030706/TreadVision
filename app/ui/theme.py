"""
TreadVision: Automotive Engineering & Scientific Dark UI Theme
Color Palette:
  Main Background:    #0F1012
  Primary Surface:    #17191C
  Secondary Surface:  #1E2024
  Elevated Surface:   #24272B
  Primary Text:       #F2F2F0
  Secondary Text:     #A7A8A5
  Muted Text:         #777A78
  Borders:            #34373B
  Accent:             #A51F32
  Accent Hover:       #C12A3E
  Technical Canvas:   #0B0C0E
"""
import streamlit as st


def apply_theme():
    """Inject sophisticated dark automotive engineering CSS stylesheet."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-main: #0F1012;
        --surface-primary: #17191C;
        --surface-secondary: #1E2024;
        --surface-elevated: #24272B;
        --text-primary: #F2F2F0;
        --text-secondary: #A7A8A5;
        --text-muted: #777A78;
        --border-color: #34373B;
        --accent-red: #A51F32;
        --accent-hover: #C12A3E;
        --canvas-technical: #0B0C0E;
    }

    /* Base Body & App Container */
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'IBM Plex Sans', 'Inter', -apple-system, sans-serif !important;
    }

    /* Hide Default Streamlit Chrome */
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
        letter-spacing: -0.01em !important;
    }

    h1 {
        font-size: 1.95rem !important;
        line-height: 1.25 !important;
        margin-bottom: 0.35rem !important;
    }

    h2 {
        font-size: 1.35rem !important;
        line-height: 1.3 !important;
        margin-top: 1.1rem !important;
        margin-bottom: 0.45rem !important;
    }

    h3 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.35rem !important;
    }

    p, span, label, div {
        color: var(--text-primary);
        font-size: 0.92rem;
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

    /* Technical Cards */
    .tv-card {
        background-color: var(--surface-primary);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 22px;
        margin-bottom: 16px;
    }

    .tv-card-header {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 6px;
    }

    .tv-card-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 8px;
    }

    .tv-card-desc {
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.55;
        margin-bottom: 16px;
    }

    /* Technical Numbered Steps */
    .tv-section-num {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--accent-red);
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    /* Badges & Technical Tags */
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
        color: #FFFFFF;
        background: var(--accent-red);
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
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-primary);
        margin-bottom: 4px;
        font-size: 0.8rem;
    }

    /* Streamlit Widget Overrides — Explicit High Contrast Dark Theme */
    
    /* 1. Labels */
    .stSelectbox label, .stTextInput label, .stSlider label, .stRadio label, .stFileUploader label {
        color: var(--text-secondary) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }

    /* 2. BaseWeb Selectbox & Dropdowns (No Black-on-Black) */
    div[data-baseweb="select"] {
        background-color: var(--surface-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 4px !important;
    }

    div[data-baseweb="select"] > div {
        background-color: var(--surface-primary) !important;
        color: var(--text-primary) !important;
        border: none !important;
    }

    div[data-baseweb="select"] span {
        color: var(--text-primary) !important;
    }

    div[data-baseweb="select"] svg {
        fill: var(--text-secondary) !important;
    }

    /* Dropdown Popover Menus */
    div[data-baseweb="popover"], div[data-baseweb="popover"] > div {
        background-color: var(--surface-secondary) !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
    }

    ul[role="listbox"] {
        background-color: var(--surface-secondary) !important;
    }

    li[role="option"] {
        background-color: var(--surface-secondary) !important;
        color: var(--text-primary) !important;
        font-size: 0.88rem !important;
    }

    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: var(--surface-elevated) !important;
        color: #FFFFFF !important;
    }

    /* 3. Text Inputs & Search */
    div[data-baseweb="input"] {
        background-color: var(--surface-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 4px !important;
    }

    div[data-baseweb="input"] input {
        background-color: var(--surface-primary) !important;
        color: var(--text-primary) !important;
        font-size: 0.88rem !important;
    }

    div[data-baseweb="input"] input::placeholder {
        color: var(--text-muted) !important;
    }

    /* 4. Buttons */
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
    }

    button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
        background-color: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
    }

    button[kind="secondary"], .stButton > button[kind="secondary"], .stButton > button {
        background-color: var(--surface-primary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 4px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        padding: 8px 18px !important;
        box-shadow: none !important;
    }

    button[kind="secondary"]:hover, .stButton > button:hover {
        background-color: var(--surface-secondary) !important;
        border-color: #4A4E53 !important;
    }

    /* 5. File Uploader */
    div[data-testid="stFileUploader"] {
        background-color: var(--surface-primary) !important;
        border: 1px dashed var(--border-color) !important;
        border-radius: 4px !important;
        padding: 16px !important;
    }

    div[data-testid="stFileUploader"]:hover {
        border-color: var(--accent-red) !important;
    }

    div[data-testid="stFileUploader"] section {
        background-color: var(--surface-primary) !important;
        color: var(--text-primary) !important;
    }

    div[data-testid="stFileUploader"] span, div[data-testid="stFileUploader"] small {
        color: var(--text-secondary) !important;
    }

    /* 6. Radio & Checkbox */
    div[data-testid="stRadio"] div[role="radiogroup"] label span {
        color: var(--text-primary) !important;
    }

    div[data-testid="stCheckbox"] label span {
        color: var(--text-primary) !important;
    }

    /* 7. Tabs */
    button[data-baseweb="tab"] {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: var(--text-secondary) !important;
        background-color: transparent !important;
        border: none !important;
        padding: 8px 16px !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--text-primary) !important;
        border-bottom: 2px solid var(--accent-red) !important;
    }

    /* 8. DataFrames & Tables */
    div[data-testid="stDataFrame"] {
        background-color: var(--surface-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 4px !important;
    }

    div[data-testid="stDataFrame"] table {
        color: var(--text-primary) !important;
    }

    /* 9. Expanders */
    div[data-testid="stExpander"] {
        background-color: var(--surface-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 4px !important;
    }

    div[data-testid="stExpander"] summary {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    div[data-testid="stExpander"] summary:hover {
        color: #FFFFFF !important;
    }

    /* 10. Metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.55rem !important;
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

    /* Sliders */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: var(--accent-red) !important;
    }

    /* Progress Bar */
    div[data-testid="stProgressBar"] > div > div {
        background-color: var(--accent-red) !important;
    }

    /* Technical Canvas Image Containers */
    div[data-testid="stImage"] {
        background-color: var(--canvas-technical) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 4px !important;
        padding: 4px !important;
    }

    div[data-testid="stImage"] figcaption {
        color: var(--text-secondary) !important;
        font-size: 0.76rem !important;
        margin-top: 4px !important;
    }

    /* Clean Technical Dividers */
    hr {
        border: none !important;
        border-top: 1px solid var(--border-color) !important;
        margin: 20px 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
