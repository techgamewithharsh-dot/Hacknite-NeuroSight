"""
Neuro-Oncology AI Diagnostic Suite:
Real-Time Multi-Modal Brain Tumor Progression & Pseudo-Progression Analyzer.

Hacknite 2026.
Powered by NVIDIA MONAI & NVIDIA Triton Inference Server.
"""

import os
import time
from pathlib import Path
import streamlit as st
import numpy as np

from core.monai_pipeline import BrainMRIPipeline
from core.progression_analyzer import ProgressionAnalyzer
from triton.triton_client import TritonInferenceBridge
from api.schemas import DiagnosticVerdict
from ui.visualizer import (
    create_2d_slice_figure,
    create_3d_volumetric_figure,
    create_ppri_gauge_figure,
    create_volumetric_comparison_chart,
)

# Set page configuration
st.set_page_config(
    page_title="Neuro-Oncology AI | Progression vs. Pseudo-Progression Terminal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Cyber-Clinical CSS Theme
st.markdown("""
<style>
    .main {
        background-color: #070A11;
        color: #F1F5F9;
    }
    .metric-card {
        background-color: #131B30;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .status-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        font-family: monospace;
        letter-spacing: 0.5px;
    }
    .status-triton {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid #10B981;
    }
    .status-local {
        background-color: rgba(56, 189, 248, 0.2);
        color: #38BDF8;
        border: 1px solid #38BDF8;
    }
    .verdict-banner-red {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 2px solid #EF4444;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .verdict-banner-green {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 2px solid #10B981;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .terminal-box {
        background-color: #03060C;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 12px;
        font-family: monospace;
        font-size: 0.8rem;
        color: #38BDF8;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    return BrainMRIPipeline()

@st.cache_resource
def get_analyzer():
    return ProgressionAnalyzer()

@st.cache_resource
def get_triton_bridge():
    return TritonInferenceBridge()


def load_demo_scans(case_dir: Path):
    """Loads 4 modalities for Baseline and Follow-up from benchmark directory."""
    pipeline = get_pipeline()
    scan_a_dict, scan_b_dict = {}, {}
    dir_a = case_dir / "scan_A_baseline"
    dir_b = case_dir / "scan_B_followup"
    
    for seq in ["t1", "t1ce", "t2", "flair"]:
        path_a = dir_a / f"baseline_{seq}.nii.gz"
        path_b = dir_b / f"followup_{seq}.nii.gz"
        if path_a.exists():
            data, _ = pipeline.load_nifti(str(path_a))
            scan_a_dict[seq] = data
        if path_b.exists():
            data, _ = pipeline.load_nifti(str(path_b))
            scan_b_dict[seq] = data
            
    return scan_a_dict, scan_b_dict


def main():
    pipeline = get_pipeline()
    analyzer = get_analyzer()
    triton_bridge = get_triton_bridge()

    sample_base = Path("data/samples")
    if not (sample_base / "case_001_true_progression").exists():
        from data_generator.generate_synthetic_cases import generate_all_cases
        with st.spinner("Synthesizing benchmark 3D multi-modal NIfTI cases..."):
            generate_all_cases(str(sample_base))

    # --- Sidebar Controls ---
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/brain.png", width=64)
        st.markdown("### CLINICAL HUD")
        
        # Server Status Badge
        is_triton_up = triton_bridge.is_triton_live()
        if is_triton_up:
            st.markdown('<span class="status-badge status-triton">&#9889; TRITON SERVER ACTIVE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-local">&#128187; LOCAL MONAI ENGINE</span>', unsafe_allow_html=True)

        st.markdown("---")
        
        # Case Selection
        st.markdown("#### Patient Scans Selection")
        case_options = [
            "Patient 001 - Glioblastoma Recurrence (True Progression)",
            "Patient 002 - Radiation Necrosis (Pseudo-Progression)",
            "Custom Upload (Pair of NIfTI Volumes)"
        ]
        selected_case = st.selectbox("Preset Case / Upload Mode", case_options)
        
        patient_id_input = st.text_input("Patient Record ID", value="PT-84920" if "001" in selected_case else "PT-31045")
        days_between = st.slider("Scan Interval (Days)", min_value=30, max_value=360, value=90, step=15)
        
        st.markdown("---")
        st.markdown("#### Slice View Controls")
        chosen_modality = st.selectbox("Background Modality", ["t1ce", "flair", "t2", "t1"], index=0)
        plane = st.radio("Slice Plane", ["Axial", "Coronal", "Sagittal"], horizontal=True)

    # Ingestion Logic
    if "Patient 001" in selected_case:
        case_dir = sample_base / "case_001_true_progression"
        scan_a_dict, scan_b_dict = load_demo_scans(case_dir)
    elif "Patient 002" in selected_case:
        case_dir = sample_base / "case_002_pseudo_progression"
        scan_a_dict, scan_b_dict = load_demo_scans(case_dir)
    else:
        st.info("Upload sequential 3D NIfTI scans (.nii.gz) for custom progression analysis.")
        c1, c2 = st.columns(2)
        with c1:
            up_a = st.file_uploader("Upload Timepoint A (Baseline .nii.gz)", type=["nii", "gz"])
        with c2:
            up_b = st.file_uploader("Upload Timepoint B (Follow-up .nii.gz)", type=["nii", "gz"])
            
        if up_a and up_b:
            da, _ = pipeline.load_nifti(up_a)
            db, _ = pipeline.load_nifti(up_b)
            scan_a_dict = {"t1ce": da, "t1": da, "t2": da, "flair": da}
            scan_b_dict = {"t1ce": db, "t1": db, "t2": db, "flair": db}
        else:
            st.stop()

    # MONAI Preprocessing
    tensor_4d_a = pipeline.preprocess_multimodal_dict(scan_a_dict)
    tensor_4d_b = pipeline.preprocess_multimodal_dict(scan_b_dict)

    # Triton Inference & Analysis
    with st.spinner("Processing 3D tensors through MONAI & Triton Inference Server..."):
        mask_a, meta_a = triton_bridge.infer(tensor_4d_a)
        mask_b, meta_b = triton_bridge.infer(tensor_4d_b)
        report = analyzer.analyze_progression(mask_a, mask_b, days_between_scans=days_between)

    # Compute Growth Velocity (cm3 / month)
    months = max(0.5, days_between / 30.0)
    growth_velocity = round(report.delta_total_volume_cm3 / months, 2)

    # Determine Diagnostic Verdict
    if report.ppri_score >= 65.0:
        verdict = DiagnosticVerdict.RADIATION_NECROSIS_PSEUDOPROGRESSION
        verdict_title = "RADIATION NECROSIS / PSEUDO-PROGRESSION"
        banner_class = "verdict-banner-green"
        badge_text = "SAFE: BENIGN INFLAMMATORY FLARE"
        badge_color = "#10B981"
        confidence_score = report.ppri_score
        action_text = "Do NOT re-operate. Maintain active chemo-radiation; schedule 4-6 week DSC-perfusion MRI follow-up."
    elif report.ppri_score <= 35.0:
        verdict = DiagnosticVerdict.TRUE_TUMOR_PROGRESSION
        verdict_title = "CRITICAL: TRUE TUMOR PROGRESSION"
        banner_class = "verdict-banner-red"
        badge_text = "CRITICAL: ACTIVE GLIOBLASTOMA RECURRENCE"
        badge_color = "#EF4444"
        confidence_score = round(100.0 - report.ppri_score, 1)
        action_text = "Urgent neuro-oncology tumor board review. Evaluate surgical re-resection or second-line therapy."
    else:
        verdict = DiagnosticVerdict.EQUIVOCAL_RESPONSE
        verdict_title = "EQUIVOCAL / INDETERMINATE PROGRESSION"
        banner_class = "verdict-banner-red"
        badge_text = "INDETERMINATE: PET SCAN REQUIRED"
        badge_color = "#F59E0B"
        confidence_score = 55.0
        action_text = "Recommend Advanced DSC-MRI perfusion or 18F-FET PET scan to distinguish tumor from radiation effect."

    # --- Header ---
    st.title("NEURO-ONCOLOGY PROGRESSION TERMINAL")
    st.caption(f"Patient: **{patient_id_input}** | Interval: **{days_between} days** | Triton Serving Latency: **{meta_b['latency_ms']} ms**")

    # --- Live Telemetry Strip ---
    with st.expander("Telemetry Stream & Triton Dispatch Log", expanded=False):
        st.markdown(f"""
        <div class="terminal-box">
            [MONAI.Spacingd] Resampled Baseline Scan A to isotropic 1.0mm³ spacing [4, {tensor_4d_a.shape[1]}, {tensor_4d_a.shape[2]}, {tensor_4d_a.shape[3]}]<br>
            [MONAI.Orientationd] Aligned Follow-up Scan B to RAS canonical orientation; normalized intensity.<br>
            [Triton.DynamicBatching] Dispatched 4D tensors to backend: {meta_b['backend']} ({meta_b['model']}).<br>
            [Triton.InferenceServer] Generated 3D sub-region segmentation masks (Latency: {meta_b['latency_ms']} ms).<br>
            [Analysis.PPRICalculator] Calculated volumetric delta and PPRI score: {report.ppri_score}%.
        </div>
        """, unsafe_allow_html=True)

    # --- High-Impact Verdict Banner ---
    st.markdown(f"""
    <div class="{banner_class}">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <span style="background:{badge_color}; color:#FFFFFF; font-weight:800; font-family:monospace; font-size:0.85rem; padding:4px 12px; border-radius:4px;">
                {badge_text}
            </span>
            <span style="font-family:monospace; font-size:1.3rem; font-weight:800; color:#FFFFFF;">
                {confidence_score:.1f}% CONFIDENCE
            </span>
        </div>
        <h1 style="margin:0 0 10px 0; font-size:2.4rem; font-weight:800; letter-spacing:-0.5px;">
            {verdict_title}
        </h1>
        <p style="font-size:1.05rem; margin:0 0 14px 0; color:#E2E8F0;">
            {report.clinical_recommendation}
        </p>
        <div style="background:rgba(0,0,0,0.3); padding:8px 14px; border-radius:6px; font-family:monospace; font-size:0.85rem;">
            <strong>CLINICAL ACTION:</strong> {action_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Volumetric Metrics Grid ---
    c_m1, c_m2, c_m3, c_m4, c_m5, c_m6 = st.columns(6)
    with c_m1:
        st.metric("Growth Velocity", f"{growth_velocity:+.2f} cm³/mo")
    with c_m2:
        st.metric("Total Tumor Delta", f"{report.delta_total_volume_cm3:+.2f} cm³", f"{report.delta_total_pct:+.1f}%")
    with c_m3:
        st.metric("Enhancing (ET) Delta", f"{report.delta_enhancing_volume_cm3:+.2f} cm³", f"{report.delta_enhancing_pct:+.1f}%")
    with c_m4:
        st.metric("Edema (ED) Delta", f"{report.delta_edema_volume_cm3:+.2f} cm³", f"{report.delta_edema_pct:+.1f}%")
    with c_m5:
        st.metric("Centroid Shift", f"{report.centroid_displacement_mm} mm")
    with c_m6:
        st.metric("PPRI Score", f"{report.ppri_score:.1f}%")

    # --- Visual Comparison Dashboard ---
    col_viz, col_gauges = st.columns([1.6, 1.0])

    with col_viz:
        tab_2d, tab_3d = st.tabs(["Synchronized 2D Orthogonal Slices", "3D Volumetric Mesh"])
        
        with tab_2d:
            bg_vol_a = scan_a_dict[chosen_modality]
            bg_vol_b = scan_b_dict[chosen_modality]
            
            max_slices = {
                "Axial": bg_vol_a.shape[0],
                "Coronal": bg_vol_a.shape[1],
                "Sagittal": bg_vol_a.shape[2]
            }
            default_slice = max_slices[plane] // 2
            slice_idx = st.slider(f"{plane} Slice Index", 0, max_slices[plane] - 1, default_slice)

            if plane == "Axial":
                slice_a, mask_slice_a = bg_vol_a[slice_idx, :, :], mask_a[slice_idx, :, :]
                slice_b, mask_slice_b = bg_vol_b[slice_idx, :, :], mask_b[slice_idx, :, :]
            elif plane == "Coronal":
                slice_a, mask_slice_a = bg_vol_a[:, slice_idx, :], mask_a[:, slice_idx, :]
                slice_b, mask_slice_b = bg_vol_b[:, slice_idx, :], mask_b[:, slice_idx, :]
            else:
                slice_a, mask_slice_a = bg_vol_a[:, :, slice_idx], mask_a[:, :, slice_idx]
                slice_b, mask_slice_b = bg_vol_b[:, :, slice_idx], mask_b[:, :, slice_idx]

            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                fig_a = create_2d_slice_figure(slice_a, mask_slice_a, title="Scan A (Baseline)", plane_name=plane)
                st.plotly_chart(fig_a, use_container_width=True)
            with sub_c2:
                fig_b = create_2d_slice_figure(slice_b, mask_slice_b, title="Scan B (Follow-up)", plane_name=plane)
                st.plotly_chart(fig_b, use_container_width=True)

            st.caption("🔴 Red = Enhancing Tumor | 🟢 Green = Peritumoral Edema | 🔵 Blue = Necrotic Core")

        with tab_3d:
            fig_3d = create_3d_volumetric_figure(mask_a, mask_b, title="3D Follow-up Tumor Isosurface")
            st.plotly_chart(fig_3d, use_container_width=True)

    with col_gauges:
        fig_gauge = create_ppri_gauge_figure(report.ppri_score)
        st.plotly_chart(fig_gauge, use_container_width=True)

        fig_bar = create_volumetric_comparison_chart(report)
        st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    main()
