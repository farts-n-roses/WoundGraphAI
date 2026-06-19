# =========================================================
# EXPLAINABLE AI PAGE
# =========================================================

import streamlit as st
import cv2
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utils.helpers import resize_image, binarize_mask
from processing.feature_extractor import extract_patch_features
from models.explainability import (
    generate_importance_map,
    overlay_heatmap
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Explainable AI",
    layout="wide"
)

st.title("🧠 Explainable AI - Wound Analysis")

st.markdown(
    """
This module visualizes **why the model focuses on
specific wound regions** using spatial importance mapping.
"""
)

# =========================================================
# UPLOAD SECTION
# =========================================================

col1, col2 = st.columns(2)

with col1:
    img_file = st.file_uploader(
        "Upload Wound Image",
        type=["png", "jpg", "jpeg"]
    )

with col2:
    mask_file = st.file_uploader(
        "Upload Wound Mask",
        type=["png", "jpg", "jpeg"]
    )

if img_file and mask_file:

    # =====================================================
    # READ INPUTS
    # =====================================================

    image = cv2.imdecode(
        np.frombuffer(
            img_file.read(),
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    mask = cv2.imdecode(
        np.frombuffer(
            mask_file.read(),
            np.uint8
        ),
        cv2.IMREAD_GRAYSCALE
    )

    # =====================================================
    # PREPROCESS
    # =====================================================
    
    image = resize_image(image, 256)
    
    mask = resize_image(mask, 256)
    mask = binarize_mask(mask)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # =====================================================
    # EXPLAINABILITY PIPELINE
    # =====================================================

    node_features, patches = extract_patch_features(image, mask)
    importance_map = generate_importance_map(image, mask)
    overlay = overlay_heatmap(image, importance_map)

    # =====================================================
    # DISPLAY
    # =====================================================

    st.subheader("🖼️ Explainability Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(image, caption="Original Image")

    with col2:
        st.image(mask, caption="Binary Mask")

    with col3:
        st.image(importance_map, caption="Importance Heatmap")

    with col4:
        st.image(overlay, caption="Overlay")

    # =====================================================
    # INTERPRETATION
    # =====================================================

    st.markdown("---")

    st.subheader("🧬 Biological Interpretation")

    st.info(
        """
The model highlights regions with higher wound density
as more influential in healing prediction.

Red/yellow regions → high biological impact  
Blue regions → low influence

This simulates attention-based interpretability used
in medical imaging AI systems.
"""
    )

else:
    st.warning("Please upload both image and mask to continue.")
