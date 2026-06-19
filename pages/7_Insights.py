# =========================================================
# INSIGHTS PAGE
# =========================================================

import streamlit as st
import numpy as np
import cv2
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from processing.temporal_simulator import binarize_mask, compute_healing_percentages
from processing.temporal_simulator import generate_healing_sequence
from processing.feature_extractor import extract_patch_features
from processing.metrics import compute_wound_percentage
from utils.helpers import binarize_mask

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Biological Insights",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🔬 Biological & Computational Insights")

st.markdown(
    """
This module translates computational outputs into
**biological interpretation of wound healing dynamics**.
"""
)

# =========================================================
# INPUT
# =========================================================

st.subheader("📁 Upload Wound Mask for Insight Analysis")

mask_file = st.file_uploader(
    "Upload Binary Wound Mask",
    type=["png", "jpg", "jpeg"]
)

# =========================================================
# ANALYSIS
# =========================================================

if mask_file:

    mask = cv2.imdecode(
        np.frombuffer(mask_file.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    mask = binarize_mask(mask)
    
    node_features, patches = extract_patch_features(mask, mask)

    # simulate healing
    sequence = generate_healing_sequence(mask, healing_rate=0.12)
    remaining = compute_healing_percentages(sequence)

    days = [0, 3, 5, 7, 10, 14]

    wound_remaining = remaining
    healing_progress = [100 - r for r in remaining]

    # =====================================================
    # METRICS
    # =====================================================

    st.subheader("📊 Healing Summary Metrics")

    col1, col2, col3 = st.columns(3)

    initial_wound = remaining[0]
    final_wound = remaining[-1]
    healing_progress = [100 - r for r in remaining]
    total_healing = initial_wound - final_wound

    with col1:
        st.metric("Initial Wound Load", f"{initial_wound:.2f}%")

    with col2:
        st.metric("Final Wound Load", f"{final_wound:.2f}%")

    with col3:
        st.metric("Total Healing Achieved", f"{total_healing:.2f}%")

    real_wound_pct = compute_wound_percentage(mask)
    healing_progress = [100 - r for r in remaining]

    df = pd.DataFrame({
        "Day": [0, 3, 5, 7, 10, 14],
        "Wound Remaining (%)": wound_remaining,
        "Healing (%)": healing_progress
    })

    fig, ax = plt.subplots()

    ax.plot(df["Day"], df["Wound Remaining (%)"], label="Wound Remaining")
    ax.plot(df["Day"], df["Healing (%)"], label="Healing Progress")

    ax.set_xlabel("Days")
    ax.set_ylabel("Percentage")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
    
    # =====================================================
    # INTERPRETATION
    # =====================================================

    st.markdown("---")
    st.subheader("🧬 Biological Interpretation")

    final_healing = healing_progress[-1]

    if final_healing > 70:
        st.success(
            """
🟢 Strong healing response detected.

- Rapid epithelial regeneration  
- Efficient wound contraction  
- Stable tissue remodeling  

Overall: Favorable biological recovery trajectory.
"""
        )

    elif final_healing > 40:
        st.warning(
            """
🟡 Moderate healing progression observed.

- Partial epithelial closure  
- Slower tissue contraction phases  
- Possible delayed remodeling activity  

Overall: Stable but monitored recovery state.
"""
        )

    else:
        st.error(
            """
🔴 Delayed or impaired healing pattern detected.

- Weak contraction dynamics  
- Reduced tissue regeneration efficiency  
- Possible abnormal healing behavior  

Overall: High-risk recovery trajectory.
"""
        )

    # =====================================================
    # COMPUTATIONAL INTERPRETATION
    # =====================================================

    st.markdown("---")
    st.subheader("🧠 Computational Interpretation")

    st.markdown(
        f"""
The simulation models wound healing as a **spatio-temporal decay process**
where tissue recovery follows a nonlinear trajectory.

### Key Observations
- Initial wound load: **{remaining[0]:.2f}%**
- Final wound load: **{remaining[-1]:.2f}%**
- Model indicates nonlinear decay-based contraction behavior (simulated healing dynamics)

### Interpretation
The system approximates epithelial healing using:
- spatial patch evolution
- decay-based contraction dynamics
- temporal sequence modeling

This allows translation of image-based data into
predictive biological trajectories.
"""
    )

else:
    st.info("Upload a wound mask to generate biological insights.")
