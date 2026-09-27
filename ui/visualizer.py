"""
Interactive Medical Image & Volumetric Visualization Components.

Includes:
- Synchronized 2D Orthogonal Slice Viewer (Axial, Coronal, Sagittal) with color-coded tumor masks.
- Interactive Plotly 3D Isosurface Mesh Rendering.
- Volumetric Delta Waterfall & Bar Charts.
- Pseudo-Progression Risk Index (PPRI) Gauge.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Color palette for tumor sub-regions
SUBREGION_COLORS = {
    1: "rgba(30, 144, 255, 0.75)",   # Necrotic Core - Deep Sky Blue
    2: "rgba(50, 205, 50, 0.65)",    # Peritumoral Edema - Lime Green
    3: "rgba(220, 20, 60, 0.85)"     # Enhancing Tumor - Crimson Red
}


def create_2d_slice_figure(scan_slice: np.ndarray, mask_slice: Optional[np.ndarray],
                           title: str = "MRI Slice", plane_name: str = "Axial") -> go.Figure:
    """Creates a high-contrast 2D grayscale MRI slice with color-coded segmentation overlay."""
    fig = go.Figure()

    # Grayscale anatomical MRI layer
    fig.add_trace(go.Heatmap(
        z=scan_slice,
        colorscale="Gray",
        showscale=False,
        hoverinfo="z"
    ))

    # Add segmentation contours/overlays if present
    if mask_slice is not None and np.any(mask_slice > 0):
        # Edema mask (Label 2)
        ed_mask = (mask_slice == 2).astype(np.float32)
        if np.any(ed_mask):
            fig.add_trace(go.Heatmap(
                z=np.where(ed_mask > 0, 1.0, np.nan),
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(46, 204, 113, 0.45)"]],
                showscale=False,
                name="Edema",
                hoverinfo="skip"
            ))

        # Enhancing Tumor mask (Label 3)
        et_mask = (mask_slice == 3).astype(np.float32)
        if np.any(et_mask):
            fig.add_trace(go.Heatmap(
                z=np.where(et_mask > 0, 1.0, np.nan),
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(231, 76, 60, 0.75)"]],
                showscale=False,
                name="Enhancing Tumor",
                hoverinfo="skip"
            ))

        # Necrotic Core mask (Label 1)
        ncr_mask = (mask_slice == 1).astype(np.float32)
        if np.any(ncr_mask):
            fig.add_trace(go.Heatmap(
                z=np.where(ncr_mask > 0, 1.0, np.nan),
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(52, 152, 219, 0.8)"]],
                showscale=False,
                name="Necrotic Core",
                hoverinfo="skip"
            ))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b> ({plane_name})", x=0.5, font=dict(size=14, color="#E0E6ED")),
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.9)",
        height=320,
        showlegend=False
    )
    return fig


def create_3d_volumetric_figure(mask_a: np.ndarray, mask_b: np.ndarray, title: str = "3D Tumor Evolution") -> go.Figure:
    """
    Renders 3D interactive isosurface representation of tumor volume
    comparing Baseline Scan A (ghosted wireframe) vs Follow-up Scan B (solid colored).
    """
    fig = go.Figure()

    # Downsample slightly for fast WebGL rendering in browser
    step = 2
    sub_a = mask_a[::step, ::step, ::step]
    sub_b = mask_b[::step, ::step, ::step]

    # Enhancing tumor in Scan B
    et_b = (sub_b == 3).astype(np.float32)
    if np.any(et_b):
        X, Y, Z = np.mgrid[0:et_b.shape[0], 0:et_b.shape[1], 0:et_b.shape[2]]
        fig.add_trace(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=et_b.flatten(),
            isomin=0.5, isomax=1.0,
            surface_count=2,
            colorscale=[[0, "#E74C3C"], [1, "#E74C3C"]],
            showscale=False,
            name="Enhancing Tumor (Follow-up)",
            opacity=0.85
        ))

    # Edema in Scan B
    ed_b = (sub_b == 2).astype(np.float32)
    if np.any(ed_b):
        X, Y, Z = np.mgrid[0:ed_b.shape[0], 0:ed_b.shape[1], 0:ed_b.shape[2]]
        fig.add_trace(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=ed_b.flatten(),
            isomin=0.5, isomax=1.0,
            surface_count=1,
            colorscale=[[0, "#2ECC71"], [1, "#2ECC71"]],
            showscale=False,
            name="Peritumoral Edema (Follow-up)",
            opacity=0.25
        ))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=16, color="#E0E6ED")),
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=""),
            yaxis=dict(showbackground=False, showticklabels=False, title=""),
            zaxis=dict(showbackground=False, showticklabels=False, title=""),
            bgcolor="rgba(15, 23, 42, 0.95)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        height=450
    )
    return fig


def create_ppri_gauge_figure(ppri_score: float) -> go.Figure:
    """Renders the Pseudo-Progression Risk Index (PPRI) Speedometer Gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ppri_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "<b>Pseudo-Progression Risk Index (PPRI)</b><br><span style='font-size:0.8em;color:#94A3B8;'>0 = True Tumor Recurrence | 100 = Radiation Necrosis</span>", 'font': {'size': 15, 'color': '#E0E6ED'}},
        number={'suffix': "%", 'font': {'size': 44, 'color': '#FFFFFF'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': "#38BDF8", 'thickness': 0.28},
            'bgcolor': "rgba(30, 41, 59, 0.6)",
            'borderwidth': 1,
            'bordercolor': "#475569",
            'steps': [
                {'range': [0, 35], 'color': 'rgba(239, 68, 68, 0.65)'},      # Red - True Progression
                {'range': [35, 65], 'color': 'rgba(245, 158, 11, 0.55)'},    # Amber - Indeterminate
                {'range': [65, 100], 'color': 'rgba(16, 185, 129, 0.65)'}    # Green - Pseudo-progression
            ],
            'threshold': {
                'line': {'color': "#F8FAFC", 'width': 4},
                'thickness': 0.8,
                'value': ppri_score
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#F8FAFC", 'family': "sans-serif"},
        margin=dict(l=25, r=25, t=60, b=20),
        height=260
    )
    return fig


def create_volumetric_comparison_chart(report) -> go.Figure:
    """Builds side-by-side sub-region volumetric comparison bar chart."""
    subregions = ["Necrotic Core (NCR)", "Peritumoral Edema (ED)", "Enhancing Tumor (ET)", "Total Lesion"]
    
    vol_a = [
        report.scan_a_metrics.necrotic_volume_cm3,
        report.scan_a_metrics.edema_volume_cm3,
        report.scan_a_metrics.enhancing_volume_cm3,
        report.scan_a_metrics.total_lesion_volume_cm3
    ]
    
    vol_b = [
        report.scan_b_metrics.necrotic_volume_cm3,
        report.scan_b_metrics.edema_volume_cm3,
        report.scan_b_metrics.enhancing_volume_cm3,
        report.scan_b_metrics.total_lesion_volume_cm3
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=subregions,
        y=vol_a,
        name="Scan A (Baseline)",
        marker_color="#64748B",
        text=[f"{v:.2f} cm³" for v in vol_a],
        textposition="auto"
    ))

    fig.add_trace(go.Bar(
        x=subregions,
        y=vol_b,
        name="Scan B (Follow-up)",
        marker_color="#0284C7",
        text=[f"{v:.2f} cm³" for v in vol_b],
        textposition="auto"
    ))

    fig.update_layout(
        barmode="group",
        title=dict(text="<b>Volumetric Trajectory by Compartment (cm³)</b>", x=0.5, font=dict(color="#E0E6ED")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#CBD5E1")),
        xaxis=dict(tickfont=dict(color="#CBD5E1"), showgrid=False),
        yaxis=dict(title="Volume (cm³)", tickfont=dict(color="#CBD5E1"), gridcolor="rgba(71, 85, 105, 0.3)"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=320
    )
    return fig
