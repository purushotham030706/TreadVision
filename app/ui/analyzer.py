"""
TreadVision: Module 1 — Tyre Tread Analyzer
Analyzes a photograph of the physical tyre tread surface, extracts visual processing stages,
measures quantifiable features, and classifies wear status.
"""
from pathlib import Path
import streamlit as st
import numpy as np
import cv2
from PIL import Image

from app.config import WEAR_DATA_DIR
from app.image_processing.pipeline import ImageProcessingPipeline
from app.ml.inference import WearInferenceEngine
from app.ui.components import render_header, render_feature_table


def render_analyzer():
    """Render the Tyre Tread Analyzer interactive page."""
    render_header(
        "Module 1: Tyre Tread Analyzer",
        "Inspect classical digital image processing stages, extract structural features, and evaluate tyre wear condition."
    )

    # Input Selection: File Upload or Sample
    col_input, col_config = st.columns([2, 1], gap="medium")

    with col_input:
        input_mode = st.radio("Select Input Source:", ["Upload Tyre Photograph", "Choose Sample Test Tyre"], horizontal=True)
        image_to_process = None
        image_label = ""

        if input_mode == "Upload Tyre Photograph":
            uploaded = st.file_uploader(
                "Upload a close-up photograph of the tyre tread surface:",
                type=["jpg", "jpeg", "png", "bmp", "webp"]
            )
            if uploaded is not None:
                image_to_process = Image.open(uploaded)
                image_label = uploaded.name
        else:
            sample_options = {
                "Good Condition (Sample Tyre G1)": WEAR_DATA_DIR / "good" / "tyre_g01_view1.jpg",
                "Moderately Worn (Sample Tyre M1)": WEAR_DATA_DIR / "moderate" / "tyre_m01_view1.jpg",
                "Heavily Worn (Sample Tyre W1)": WEAR_DATA_DIR / "worn" / "tyre_w01_view1.jpg",
            }
            selected_sample_name = st.selectbox("Select a calibrated sample tyre image:", list(sample_options.keys()))
            sample_path = sample_options[selected_sample_name]
            if sample_path.exists():
                image_to_process = str(sample_path)
                image_label = selected_sample_name

    with col_config:
        st.markdown("<div style='font-weight: 600; color: #38bdf8; margin-bottom: 8px;'>⚙️ Pipeline Parameters</div>", unsafe_allow_html=True)
        noise_filter = st.selectbox("Noise Reduction Filter:", ["bilateral", "gaussian", "median"], index=0)
        contrast_opt = st.selectbox("Contrast Enhancement:", ["clahe", "histeq"], index=0)
        seg_method = st.selectbox("Segmentation Strategy:", ["otsu", "adaptive"], index=0)
        canny_low = st.slider("Canny Low Threshold:", 10, 100, 50)
        canny_high = st.slider("Canny High Threshold:", 100, 250, 150)

    if image_to_process is None:
        st.info("👆 Please upload a photograph of a tyre tread or select one of the provided sample tyres above to begin.")
        return

    # Execute Processing Pipeline
    pipeline = ImageProcessingPipeline(
        target_size=(512, 512),
        noise_method=noise_filter,
        contrast_method=contrast_opt,
        segmentation_method=seg_method,
        canny_low=canny_low,
        canny_high=canny_high
    )

    with st.spinner("Processing image through 12-stage computer vision pipeline..."):
        result = pipeline.process(image_to_process)

    if not result.success:
        st.error(f"Image Processing Error: {result.error_message}")
        return

    # Classification using Machine Learning Inference Engine
    inference_engine = WearInferenceEngine()
    prediction = inference_engine.predict(result.feature_vector, result.features)

    # Store latest analysis into session_state for Module 2 (Tyre Replacement)
    st.session_state["latest_analysis"] = {
        "image_label": image_label,
        "metrics": result.metrics,
        "features": result.features,
        "prediction": prediction,
        "result_stages": result.stages
    }

    # High-Level Condition & Classification Banner
    pred_class = prediction["predicted_class"]
    badge_color = {
        "good": "tv-badge-emerald",
        "moderate": "tv-badge-amber",
        "worn": "tv-badge-red"
    }.get(pred_class, "tv-badge-cyan")

    st.markdown(f"""
    <div style="background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 20px 24px; margin: 20px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span class="tv-badge {badge_color}">Condition Assessment</span>
                <h2 style="color: #ffffff; margin: 6px 0 0 0;">{prediction['label']}</h2>
                <div style="color: #94a3b8; font-size: 0.88rem; margin-top: 4px;">
                    Classification Engine: {prediction['engine']} &nbsp;|&nbsp; Confidence: {prediction['confidence']*100:.1f}%
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #38bdf8;">
                    Pipeline Execution Time: {result.execution_time_ms} ms
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Core Pattern Integrity Metrics
    m = result.metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Pattern Integrity", f"{m['pattern_integrity_pct']}%")
    with col_m2:
        st.metric("Groove Visibility", f"{m['groove_visibility_pct']}%")
    with col_m3:
        st.metric("Edge Definition", f"{m['edge_definition_pct']}%")
    with col_m4:
        st.metric("Texture Condition", f"{m['texture_condition_pct']}%")

    # Visual Processing Stages Tabs
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Visual Image Processing Stages")

    stage_tabs = st.tabs([
        "Original", "Grayscale", "Denoised", "Enhanced (CLAHE)",
        "Segmentation Mask", "Segmented Tread", "Canny Edges", "Sobel Magnitude", "Laplacian"
    ])

    for i, (stage_name, tab) in enumerate(zip(result.stages.keys(), stage_tabs)):
        with tab:
            img = result.stages[stage_name]
            st.image(img, caption=f"Stage {i+1}: {stage_name}", use_container_width=True)

    # Quantitative Extracted Features
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Quantitative Feature Vector (25 Classical CV Descriptors)")
    render_feature_table(result.features)

    # Probability Distribution (if available)
    probs = prediction.get("probabilities")
    if probs:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Classification Probability Distribution")
        prob_cols = st.columns(3)
        prob_cols[0].progress(probs.get("good", 0.0), text=f"Good / Low Wear: {probs.get('good', 0.0)*100:.1f}%")
        prob_cols[1].progress(probs.get("moderate", 0.0), text=f"Moderately Worn: {probs.get('moderate', 0.0)*100:.1f}%")
        prob_cols[2].progress(probs.get("worn", 0.0), text=f"Heavily Worn: {probs.get('worn', 0.0)*100:.1f}%")

    # Seamless Transition to Module 2
    st.markdown("---")
    c_btn1, c_btn2 = st.columns([2, 1])
    with c_btn1:
        st.markdown("""
        <strong>Need practical replacement advice based on this tyre's condition?</strong><br>
        Proceed to the Tyre Replacement Advisor to check legal thresholds, enter physical depth measurements, and match database replacements.
        """, unsafe_allow_html=True)
    with c_btn2:
        if st.button("Proceed to Tyre Replacement ➡️", use_container_width=True, type="primary"):
            st.session_state["nav_selection"] = "replacement"
            st.rerun()
