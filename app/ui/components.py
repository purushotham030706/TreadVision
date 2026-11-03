"""
TreadVision: Shared UI Component Library
Engineering and scientific styling with dark theme and zero emojis.
"""
from typing import Dict, Any, List
import streamlit as st
import pandas as pd


def render_header(title: str, subtitle: str = ""):
    """Standard technical page header."""
    st.markdown(f"""
    <div style="margin-bottom: 24px; padding-bottom: 14px; border-bottom: 1px solid #41434E;">
        <div style="font-size: 0.76rem; letter-spacing: 0.12em; text-transform: uppercase; color: #D8125B; font-weight: 600; margin-bottom: 4px;">
            TREADVISION
        </div>
        <h1 style="color: #F2F2F4; margin: 0 0 6px 0; font-size: 1.85rem; font-weight: 600;">{title}</h1>
        {f'<div style="color: #B4B5BC; font-size: 0.95rem; line-height: 1.5;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_forensic_notice():
    """Professional Forensic Analysis Notice with zero emojis."""
    st.markdown("""
    <div class="tv-notice">
        <div class="tv-notice-title">FORENSIC ANALYSIS NOTICE</div>
        TreadVision provides image-based similarity analysis for research, investigative lead generation,
        and educational purposes. A similarity result does not establish that a particular vehicle or tyre was
        present at a crime scene and should not be treated as definitive forensic evidence without professional
        physical examination, calibrated impression casting, and verified legal chain-of-custody procedures.
    </div>
    """, unsafe_allow_html=True)


def render_feature_table(features_dict: Dict[str, Any]):
    """Format and display edge, texture, and shape features in clean technical tables."""
    tab_edge, tab_tex, tab_shape = st.tabs(["Edge Features", "GLCM Texture", "Shape Features"])

    with tab_edge:
        edge_data = features_dict.get("edge", {})
        df_edge = pd.DataFrame([
            {"Feature": "Total Edge Density", "Value": f"{edge_data.get('total_edge_density', 0):.5f}", "Description": "Ratio of edge boundary pixels to total area"},
            {"Feature": "Horizontal Edge Density", "Value": f"{edge_data.get('horizontal_edge_density', 0):.5f}", "Description": "Circumferentially aligned tread grooves"},
            {"Feature": "Vertical Edge Density", "Value": f"{edge_data.get('vertical_edge_density', 0):.5f}", "Description": "Transverse lateral sipes and water drainage channels"},
            {"Feature": "Orientation Mean", "Value": f"{edge_data.get('edge_orientation_mean', 0):.2f}°", "Description": "Mean gradient angle of detected edges"},
            {"Feature": "Orientation Deviation", "Value": f"{edge_data.get('edge_orientation_std', 0):.2f}°", "Description": "Angular dispersion across tread blocks"}
        ])
        st.dataframe(df_edge, use_container_width=True, hide_index=True)

    with tab_tex:
        tex_data = features_dict.get("texture", {})
        df_tex = pd.DataFrame([
            {"GLCM Property": "Contrast", "Value": f"{tex_data.get('contrast', 0):.4f}", "Interpretation": "Measures intensity variance; higher in deep, unworn grooves"},
            {"GLCM Property": "Dissimilarity", "Value": f"{tex_data.get('dissimilarity', 0):.4f}", "Interpretation": "Linear measure of local intensity discrepancies"},
            {"GLCM Property": "Homogeneity", "Value": f"{tex_data.get('homogeneity', 0):.4f}", "Interpretation": "Measures smoothness; higher in heavily worn, depleted tread"},
            {"GLCM Property": "Energy", "Value": f"{tex_data.get('energy', 0):.4f}", "Interpretation": "Sum of squared elements in co-occurrence matrix"},
            {"GLCM Property": "Correlation", "Value": f"{tex_data.get('correlation', 0):.4f}", "Interpretation": "Linear dependency of gray levels over spatial offsets"},
            {"GLCM Property": "ASM (Angular Second Moment)", "Value": f"{tex_data.get('ASM', 0):.4f}", "Interpretation": "Orderliness and textural uniformity metric"}
        ])
        st.dataframe(df_tex, use_container_width=True, hide_index=True)

    with tab_shape:
        shape_data = features_dict.get("shape", {})
        df_shape = pd.DataFrame([
            {"Shape Property": "Area Ratio", "Value": f"{shape_data.get('area_ratio', 0):.4f}", "Description": "Normalized tread contact area"},
            {"Shape Property": "Perimeter Ratio", "Value": f"{shape_data.get('perimeter_ratio', 0):.4f}", "Description": "Normalized cumulative contour boundary length"},
            {"Shape Property": "Aspect Ratio", "Value": f"{shape_data.get('aspect_ratio', 0):.3f}", "Description": "Width to height ratio of dominant tread block"},
            {"Shape Property": "Solidity", "Value": f"{shape_data.get('solidity', 0):.4f}", "Description": "Ratio of contour area to its convex hull"},
            {"Shape Property": "Extent", "Value": f"{shape_data.get('extent', 0):.4f}", "Description": "Ratio of contour area to bounding box area"},
            {"Shape Property": "Circularity", "Value": f"{shape_data.get('circularity', 0):.4f}", "Description": "Compactness of discrete tread lugs (4π·Area / Perimeter²)"},
            {"Shape Property": "Block Count", "Value": str(shape_data.get('block_count', 0)), "Description": "Discrete segmented tread components detected"}
        ])
        st.dataframe(df_shape, use_container_width=True, hide_index=True)

        hu_vals = shape_data.get("hu_moments", [])
        if hu_vals:
            with st.expander("Inspect 7 Scale/Rotation Invariant Hu Moments", expanded=False):
                hu_df = pd.DataFrame([
                    {"Moment": f"Hu Moment {i+1}", "Log Value": f"{val:.4f}"}
                    for i, val in enumerate(hu_vals)
                ])
                st.dataframe(hu_df, use_container_width=True, hide_index=True)
