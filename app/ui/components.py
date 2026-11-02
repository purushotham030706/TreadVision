"""
TreadVision: Shared UI Component Library
"""
from typing import Dict, Any, List
import streamlit as st
import pandas as pd


def render_header(title: str, subtitle: str = ""):
    """Standard page title and contextual subtitle."""
    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <h1 style="color: #ffffff; margin-bottom: 4px;">{title}</h1>
        <p style="color: #94a3b8; font-size: 1.05rem; margin-top: 0;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_forensic_notice():
    """Prominent Forensic Analysis Notice required by guidelines."""
    st.markdown("""
    <div class="tv-forensic-notice">
        <div class="tv-forensic-notice-title">⚠️ Forensic Analysis Notice</div>
        TreadVision provides image-based similarity analysis for research, investigative lead generation,
        and educational purposes. A similarity result does not establish that a particular vehicle or tyre was
        present at a crime scene and should not be treated as definitive forensic evidence without professional
        physical examination, calibrated impression casting, and verified legal chain-of-custody procedures.
    </div>
    """, unsafe_allow_html=True)


def render_feature_table(features_dict: Dict[str, Any]):
    """Format and display edge, texture, and shape features in neat tabs."""
    tab_edge, tab_tex, tab_shape = st.tabs(["Edge Descriptors", "GLCM Texture", "Shape Geometry"])

    with tab_edge:
        edge_data = features_dict.get("edge", {})
        df_edge = pd.DataFrame([
            {"Descriptor": "Total Edge Density", "Value": f"{edge_data.get('total_edge_density', 0):.5f}", "Description": "Ratio of edge boundary pixels to total area"},
            {"Descriptor": "Horizontal Edge Density", "Value": f"{edge_data.get('horizontal_edge_density', 0):.5f}", "Description": "Edges aligned circumferentially with tread rotation"},
            {"Descriptor": "Vertical Edge Density", "Value": f"{edge_data.get('vertical_edge_density', 0):.5f}", "Description": "Transverse lateral sipes and water displacement grooves"},
            {"Descriptor": "Orientation Mean", "Value": f"{edge_data.get('edge_orientation_mean', 0):.2f}°", "Description": "Mean gradient angle of tread edges"},
            {"Descriptor": "Orientation Std Dev", "Value": f"{edge_data.get('edge_orientation_std', 0):.2f}°", "Description": "Directional dispersion across tread patterns"}
        ])
        st.dataframe(df_edge, use_container_width=True, hide_index=True)

    with tab_tex:
        tex_data = features_dict.get("texture", {})
        df_tex = pd.DataFrame([
            {"GLCM Property": "Contrast", "Value": f"{tex_data.get('contrast', 0):.4f}", "Interpretation": "Measures intensity variance; higher in deep fresh grooves"},
            {"GLCM Property": "Dissimilarity", "Value": f"{tex_data.get('dissimilarity', 0):.4f}", "Interpretation": "Linear measure of local intensity discrepancies"},
            {"GLCM Property": "Homogeneity", "Value": f"{tex_data.get('homogeneity', 0):.4f}", "Interpretation": "Measures smoothness; higher in heavily worn, bald tyres"},
            {"GLCM Property": "Energy", "Value": f"{tex_data.get('energy', 0):.4f}", "Interpretation": "Sum of squared elements in co-occurrence matrix"},
            {"GLCM Property": "Correlation", "Value": f"{tex_data.get('correlation', 0):.4f}", "Interpretation": "Linear dependency of gray levels over spatial offsets"},
            {"GLCM Property": "ASM (Angular Second Moment)", "Value": f"{tex_data.get('ASM', 0):.4f}", "Interpretation": "Orderliness and textural uniformity metric"}
        ])
        st.dataframe(df_tex, use_container_width=True, hide_index=True)

    with tab_shape:
        shape_data = features_dict.get("shape", {})
        df_shape = pd.DataFrame([
            {"Shape Property": "Area Ratio", "Value": f"{shape_data.get('area_ratio', 0):.4f}", "Notes": "Normalized tread contact area"},
            {"Shape Property": "Perimeter Ratio", "Value": f"{shape_data.get('perimeter_ratio', 0):.4f}", "Notes": "Normalized cumulative contour boundary length"},
            {"Shape Property": "Aspect Ratio", "Value": f"{shape_data.get('aspect_ratio', 0):.3f}", "Notes": "Width to height ratio of dominant tread block"},
            {"Shape Property": "Solidity", "Value": f"{shape_data.get('solidity', 0):.4f}", "Notes": "Ratio of contour area to its convex hull"},
            {"Shape Property": "Extent", "Value": f"{shape_data.get('extent', 0):.4f}", "Notes": "Ratio of contour area to bounding box area"},
            {"Shape Property": "Circularity", "Value": f"{shape_data.get('circularity', 0):.4f}", "Notes": "4π * Area / Perimeter² (compactness of tread lugs)"},
            {"Shape Property": "Tread Block Count", "Value": str(shape_data.get('block_count', 0)), "Notes": "Discrete segmented tread components detected"}
        ])
        st.dataframe(df_shape, use_container_width=True, hide_index=True)

        hu_vals = shape_data.get("hu_moments", [])
        if hu_vals:
            with st.expander("Inspect 7 Scale/Rotation Invariant Hu Moments"):
                hu_df = pd.DataFrame([
                    {"Moment": f"Hu Moment #{i+1}", "Log Value": f"{val:.4f}"}
                    for i, val in enumerate(hu_vals)
                ])
                st.dataframe(hu_df, use_container_width=True, hide_index=True)
