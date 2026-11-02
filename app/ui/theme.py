"""
TreadVision: Automotive Dark Modern Design Theme and CSS
"""
import streamlit as st


def apply_theme():
    """Inject polished automotive dark CSS styles."""
    st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Dark Theme Settings */
    :root {
        --bg-dark: #0a0d14;
        --card-bg: #111726;
        --card-border: #1e293b;
        --accent-cyan: #06b6d4;
        --accent-cyan-glow: rgba(6, 182, 212, 0.25);
        --accent-amber: #f59e0b;
        --accent-red: #ef4444;
        --accent-emerald: #10b981;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Headings */
    h1, h2, h3, .tv-brand {
        font-family: 'Chakra Petch', sans-serif;
        letter-spacing: 0.5px;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Hero Banner */
    .tv-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #082f49 100%);
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 36px 32px;
        margin-bottom: 28px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
    }

    .tv-hero::before {
        content: "";
        position: absolute;
        top: 0; right: 0; bottom: 0; left: 0;
        background: radial-gradient(circle at 85% 20%, rgba(6, 182, 212, 0.15) 0%, transparent 60%);
        pointer-events: none;
    }

    .tv-hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
        letter-spacing: 1px;
    }

    .tv-hero-tagline {
        font-size: 1.25rem;
        color: #38bdf8;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .tv-hero-sub {
        font-size: 0.95rem;
        color: #94a3b8;
        line-height: 1.6;
        max-width: 880px;
    }

    /* Automotive Cards */
    .tv-card {
        background-color: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .tv-card:hover {
        border-color: #38bdf8;
        box-shadow: 0 8px 24px -4px rgba(6, 182, 212, 0.15);
    }

    .tv-card-header {
        font-family: 'Chakra Petch', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .tv-card-desc {
        font-size: 0.9rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-bottom: 16px;
    }

    /* Badges */
    .tv-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .tv-badge-cyan {
        background: rgba(6, 182, 212, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(6, 182, 212, 0.4);
    }

    .tv-badge-emerald {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .tv-badge-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }

    .tv-badge-red {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    /* Forensic Notice Callout */
    .tv-forensic-notice {
        background: rgba(15, 23, 42, 0.85);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 20px 0;
        font-size: 0.88rem;
        color: #cbd5e1;
        line-height: 1.55;
    }

    .tv-forensic-notice-title {
        font-family: 'Chakra Petch', sans-serif;
        color: #fbbf24;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 4px;
    }

    /* Stat KPI Box */
    .tv-kpi {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }

    .tv-kpi-val {
        font-family: 'Chakra Petch', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }

    .tv-kpi-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    </style>
    """, unsafe_allow_html=True)
