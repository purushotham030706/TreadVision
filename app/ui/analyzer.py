"""
TreadVision: Module 1 — Tyre Tread Analyzer
Workstation interface for digital image processing inspection, feature analysis, and wear evaluation.
"""
from pathlib import Path
import streamlit as st
import numpy as np
import cv2
from PIL import Image

from app.config import WEAR_DATA_DIR, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD
from app.image_processing.pipeline import ImageProcessingPipeline
from app.ml.inference import WearInferenceEngine
from app.ui.components import render_header, render_feature_table


def render_analyzer():
    """Render the Tyre Tread Analyzer workstation."""
    render_header(
        "TREAD ANALYZER",
        "Assess the condition, physical relief, and structural characteristics of a tyre tread from an image."
    )

    # Workstation Two-Column Input Layout
    col_input, col_ctrl = st.columns([1.5, 1], gap="large")

    with col_input:
        st.markdown("""
        <div style="font-size: 0.76rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F0; font-weight: 700; margin-bottom: 8px;">
            TREAD IMAGE INPUT
        </div>
        """, unsafe_allow_html=True)

        source_type = st.radio(
            "Input Mode:",
            ["Upload Photograph", "Select Calibrated Sample"],
            horizontal=True,
            label_visibility="collapsed"
        )

        image_to_process = None
        image_label = ""

        if source_type == "Upload Photograph":
            uploaded = st.file_uploader(
                "Upload a close-up photograph of the tyre tread contact patch:",
                type=["jpg", "jpeg", "png", "bmp", "webp"]
            )
            if uploaded is not None:
                image_to_process = Image.open(uploaded)
                image_label = uploaded.name
        else:
            sample_options = {
                "Good / Low Wear — Calibrated Sample G1": WEAR_DATA_DIR / "good" / "tyre_g01_view1.jpg",
                "Moderately Worn — Calibrated Sample M1": WEAR_DATA_DIR / "moderate" / "tyre_m01_view1.jpg",
                "Heavily Worn — Calibrated Sample W1": WEAR_DATA_DIR / "worn" / "tyre_w01_view1.jpg",
            }
            selected_name = st.selectbox(
                "Reference Sample:",
                list(sample_options.keys()),
                label_visibility="collapsed"
            )
            sample_path = sample_options[selected_name]
            if sample_path.exists():
                image_to_process = str(sample_path)
                image_label = selected_name

    with col_ctrl:
        st.markdown("""
        <div style="font-size: 0.76rem; letter-spacing: 0.08em; text-transform: uppercase; color: #F2F2F0; font-weight: 700; margin-bottom: 8px;">
            PIPELINE CONTROLS
        </div>
        """, unsafe_allow_html=True)

        noise_filter = st.selectbox("Noise Reduction Filter:", ["bilateral", "gaussian", "median"], index=0)
        contrast_opt = st.selectbox("Contrast Enhancement:", ["clahe", "histeq"], index=0)
        seg_method = st.selectbox("Tread Segmentation:", ["otsu", "adaptive"], index=0)
        canny_low = st.slider("Canny Low Threshold:", 10, 100, CANNY_LOW_THRESHOLD, step=5)
        canny_high = st.slider("Canny High Threshold:", 50, 200, CANNY_HIGH_THRESHOLD, step=5)

    if image_to_process is None:
        st.markdown("""
        <div class="tv-notice" style="margin-top: 24px;">
            <div class="tv-notice-title">AWAITING IMAGE INPUT</div>
            Upload a tyre tread photograph or select a calibrated reference sample from the controls above to execute analysis.
        </div>
        """, unsafe_allow_html=True)
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

    with st.spinner("Executing digital image processing pipeline..."):
        result = pipeline.process(image_to_process)

    if not result.success:
        st.markdown(f"""
        <div class="tv-notice" style="border-left-color: #A51F32;">
            <div class="tv-notice-title">ANALYSIS UNAVAILABLE</div>
            {result.error_message}. Ensure the input contains an unobstructed tread surface.
        </div>
        """, unsafe_allow_html=True)
        return

    # Machine Learning Inference
    inference_engine = WearInferenceEngine()
    prediction = inference_engine.predict(result.feature_vector, result.features)

    # Cache into session_state for Module 2
    st.session_state["latest_analysis"] = {
        "image_label": image_label,
        "metrics": result.metrics,
        "features": result.features,
        "prediction": prediction,
        "result_stages": result.stages
    }

    # Technical Report Style Analysis Result
    pred_class = prediction["predicted_class"]
    pred_label = prediction["label"].upper()
    confidence_pct = prediction["confidence"] * 100.0

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: #17191C; border: 1px solid #34373B; border-radius: 4px; padding: 24px; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; border-bottom: 1px solid #34373B; padding-bottom: 14px; margin-bottom: 16px;">
            <div>
                <div style="font-size: 0.74rem; letter-spacing: 0.1em; text-transform: uppercase; color: #A51F32; font-weight: 700;">
                    ANALYSIS RESULT
                </div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #F2F2F0; margin-top: 4px;">
                    {pred_label}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #A7A8A5; font-weight: 600;">
                    CLASSIFICATION CONFIDENCE
                </div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600; color: #F2F2F0; margin-top: 2px;">
                    {confidence_pct:.1f}%
                </div>
                <div style="font-size: 0.74rem; color: #A7A8A5;">
                    Execution Time: <span class="tv-mono">{result.execution_time_ms} ms</span>
                </div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px;">
            <div>
                <div style="font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #A7A8A5; font-weight: 600;">
                    PATTERN INTEGRITY
                </div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.35rem; font-weight: 600; color: #F2F2F0; margin-top: 4px;">
                    {result.metrics['pattern_integrity_pct']:.1f}%
                </div>
            </div>
            <div>
                <div style="font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #A7A8A5; font-weight: 600;">
                    GROOVE VISIBILITY
                </div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.35rem; font-weight: 600; color: #F2F2F0; margin-top: 4px;">
                    {result.metrics['groove_visibility_pct']:.1f}%
                </div>
            </div>
            <div>
                <div style="font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #A7A8A5; font-weight: 600;">
                    EDGE DEFINITION
                </div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.35rem; font-weight: 600; color: #F2F2F0; margin-top: 4px;">
                    {result.metrics['edge_definition_pct']:.1f}%
                </div>
            </div>
            <div>
                <div style="font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #A7A8A5; font-weight: 600;">
                    TEXTURE CONDITION
                </div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.35rem; font-weight: 600; color: #F2F2F0; margin-top: 4px;">
                    {result.metrics['texture_condition_pct']:.1f}%
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Image Processing Pipeline Stages Grid
    st.markdown("""
    <div style="font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; color: #F2F2F0; font-weight: 700; margin-bottom: 12px;">
        IMAGE PROCESSING PIPELINE
    </div>
    """, unsafe_allow_html=True)

    stage_descriptions = {
        "Original": ("01", "ORIGINAL IMAGE", "Normalized 512x512 spatial input format"),
        "Grayscale": ("02", "GRAYSCALE CONVERSION", "Luminance weighting (0.299R + 0.587G + 0.114B)"),
        "Denoised": ("03", "NOISE REDUCTION", f"Bilateral edge-preserving smoothing filter"),
        "Enhanced (CLAHE)": ("04", "CONTRAST ENHANCEMENT", "Adaptive histogram equalization with tile limiting"),
        "Segmentation Mask": ("05", "TREAD SEGMENTATION", "Otsu bi-level threshold with component area filtering"),
        "Segmented Tread": ("06", "SEGMENTED TREAD", "Isolated contact patch excluding background noise"),
        "Canny Edges": ("07", "CANNY EDGE MAP", "Hysteresis gradient tracing of groove boundaries"),
        "Sobel Magnitude": ("08", "SOBEL GRADIENT", "First-order horizontal and vertical spatial derivatives"),
        "Laplacian": ("09", "LAPLACIAN OPERATOR", "Second-order derivative edge inflection highlights")
    }

    stage_keys = list(result.stages.keys())

    # Render structured comparison grid (3 stages per row)
    for row_idx in range(0, len(stage_keys), 3):
        cols = st.columns(3, gap="medium")
        for col_idx in range(3):
            k_idx = row_idx + col_idx
            if k_idx < len(stage_keys):
                s_key = stage_keys[k_idx]
                s_num, s_title, s_desc = stage_descriptions.get(s_key, (f"0{k_idx+1}", s_key.upper(), ""))
                with cols[col_idx]:
                    st.markdown(f"""
                    <div style="background: #17191C; border: 1px solid #34373B; border-radius: 4px; padding: 12px; margin-bottom: 16px;">
                        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #A51F32; font-weight: 600;">
                            STAGE {s_num}
                        </div>
                        <div style="font-size: 0.85rem; font-weight: 600; color: #F2F2F0; margin-bottom: 4px;">
                            {s_title}
                        </div>
                        <div style="font-size: 0.74rem; color: #A7A8A5; margin-bottom: 8px; line-height: 1.4;">
                            {s_desc}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(result.stages[s_key], use_container_width=True)

    # Feature Analysis Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; color: #F2F2F0; font-weight: 700; margin-bottom: 12px;">
        FEATURE ANALYSIS
    </div>
    """, unsafe_allow_html=True)
    render_feature_table(result.features)

    # Machine Learning Wear Classification Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; color: #F2F2F0; font-weight: 700; margin-bottom: 12px;">
        WEAR CLASSIFICATION
    </div>
    """, unsafe_allow_html=True)

    ml_col1, ml_col2 = st.columns([1.5, 1], gap="large")

    with ml_col1:
        st.markdown(f"""
        <div style="background: #17191C; border: 1px solid #34373B; border-radius: 4px; padding: 18px 20px;">
            <div style="font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #A7A8A5; font-weight: 600;">
                ACTIVE CLASSIFICATION ENGINE
            </div>
            <div style="font-size: 1.15rem; font-weight: 600; color: #F2F2F0; margin: 4px 0 10px 0;">
                {prediction['engine']}
            </div>
            <div style="font-size: 0.82rem; color: #A7A8A5; line-height: 1.5;">
                Model trained on calibrated physical-tyre stratified splits to prevent identity leakage.
                Input feature vector comprises 25 normalized spatial, texture, and moment coefficients.
            </div>
            <div style="margin-top: 12px; font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; color: #A51F32;">
                DATASET STATUS: CURRENT VALIDATION SET
            </div>
        </div>
        """, unsafe_allow_html=True)

    with ml_col2:
        probs = prediction.get("probabilities")
        if probs:
            st.markdown("""
            <div style="font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #A7A8A5; font-weight: 600; margin-bottom: 8px;">
                CLASS PROBABILITIES
            </div>
            """, unsafe_allow_html=True)
            p_good = probs.get("good", 0.0)
            p_mod = probs.get("moderate", 0.0)
            p_worn = probs.get("worn", 0.0)

            st.write(f"GOOD / LOW WEAR: **{p_good*100:.1f}%**")
            st.progress(p_good)
            st.write(f"MODERATELY WORN: **{p_mod*100:.1f}%**")
            st.progress(p_mod)
            st.write(f"HEAVILY WORN: **{p_worn*100:.1f}%**")
            st.progress(p_worn)

    # Workflow Action Link to Module 2
    st.markdown("---")
    act_col1, act_col2 = st.columns([2.5, 1])
    with act_col1:
        st.markdown("""
        <div style="font-size: 0.9rem; color: #F2F2F0; font-weight: 500;">
            Continue to Tyre Replacement Advisor
        </div>
        <div style="font-size: 0.82rem; color: #A7A8A5;">
            Evaluate tread depth safety thresholds, inspect legal minimum benchmarks, and review database replacement options.
        </div>
        """, unsafe_allow_html=True)
    with act_col2:
        if st.button("VIEW REPLACEMENT", key="btn_analyzer_to_replacement", use_container_width=True, type="primary"):
            st.session_state["nav_selection"] = "replacement"
            st.rerun()
