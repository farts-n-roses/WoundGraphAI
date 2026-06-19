# =========================================================
# TEMPORAL ANALYSIS PAGE
# =========================================================

import sys
from pathlib import Path

# =========================================================
# PROJECT ROOT FIX
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# =========================================================
# IMPORTS
# =========================================================

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.helpers import binarize_mask
from processing.metrics import compute_wound_percentage
from processing.temporal_simulator import (
    generate_healing_sequence,
    compute_healing_percentages
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Temporal Healing Analysis",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("📈 Temporal Wound Healing Analysis")

st.markdown("""
This module simulates wound healing progression over time
using computational temporal modeling and exponential
healing dynamics.
""")

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_mask = st.file_uploader(
    "Upload Wound Mask",
    type=["png", "jpg", "jpeg"]
)

# =========================================================
# PROCESSING
# =========================================================

if uploaded_mask is not None:

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    mask = cv2.imdecode(
        #file_bytes,
        np.frombuffer(uploaded_mask.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    mask = binarize_mask(mask)
    mask = mask.astype(np.uint8)
    
    initial_wound = compute_wound_percentage(mask)

    # =====================================================
    # GENERATE TEMPORAL SEQUENCE
    # =====================================================

    sequence = generate_healing_sequence(mask, healing_rate=0.12)
    remaining = compute_healing_percentages(sequence)
    
    days = [0, 3, 5, 7, 10, 14]

    # =====================================================
    # DISPLAY HEALING TIMELINE
    # =====================================================

    st.subheader("🩹 Healing Progression Timeline")

    cols = st.columns(len(sequence))

    healing_values = []

    for i, seq_mask in enumerate(sequence):

        healing_values.append(remaining[i])

        with cols[i]:
            st.image(
                seq_mask,
                caption=(
                    f"Day {days[i]}\n"
                    f"{remaining[i]:.2f}%"
                ),
                width="stretch"
            )

    # =====================================================
    # HEALING CURVE
    # =====================================================

    st.subheader("📉 Healing Progression Curve")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(days, healing_values, marker="o", linewidth=3)

    ax.set_xlabel("Healing Day")
    ax.set_ylabel("Wound Area Recovery (%)")
    ax.set_title("Temporal (Exponential) Wound Healing Progression")
    ax.grid(True)

    st.pyplot(fig)

    # =====================================================
    # METRICS
    # =====================================================

    st.subheader("📊 Healing Metrics")

    initial_area = remaining[0]
    final_area = remaining[-1]

    total_reduction = ((initial_area - final_area) / initial_area) * 100
    avg_healing_rate = (total_reduction / (days[-1] - days[0]))

    c1, c2, c3 = st.columns(3)

    c1.metric("Initial Wound Area", f"{initial_area:.2f}%")

    c2.metric("Final Wound Area", f"{final_area:.2f}%")

    c3.metric("Total Healing", f"{total_reduction:.2f}%")

    # =====================================================
    # HEALING INTERPRETATION
    # =====================================================

    st.subheader("🧠 Biological Interpretation")

    if total_reduction > 70:
        st.success("""
        Rapid wound contraction observed.
        Healing trajectory appears favorable with
        strong epithelial recovery dynamics.
        """)

    elif total_reduction > 40:
        st.info("""
        Moderate wound healing progression detected.
        Tissue recovery appears stable but incomplete.
        """)

    else:
        st.warning("""
        Delayed healing progression detected.
        Possible abnormal tissue recovery dynamics.
        """)

    # =====================================================
    # TEMPORAL INSIGHTS
    # =====================================================

    st.subheader("🔬 Temporal Insights")

    st.markdown(f"""
    ### Computational Analysis Summary

    - Initial wound coverage: **{initial_area:.2f}%**
    - Final wound coverage: **{final_area:.2f}%**
    - Total simulated healing: **{total_reduction:.2f}%**
    - Average healing rate: **{avg_healing_rate:.2f}% per stage**

    ### Biological Interpretation

    The wound demonstrates progressive contraction over
    simulated healing intervals with nonlinear temporal
    dynamics consistent with exponential epithelial
    recovery behavior.

    The decreasing healing slope suggests:
    - rapid early tissue contraction
    - slower late-stage remodeling
    - asymptotic wound closure behavior

    This temporal behavior resembles biologically
    plausible epithelial wound healing kinetics.
    """)

else:
    st.info("Please upload a wound mask to begin temporal analysis.")
