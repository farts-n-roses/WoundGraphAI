# =========================================================
# SEGMENTATION PAGE (STREAMLIT UI)
# =========================================================

import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
import numpy as np
import cv2

from utils.helpers import resize_image, binarize_mask, overlay_images
from processing.metrics import (
    compute_wound_percentage,
    compute_perimeter
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Segmentation Analysis",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Wound Segmentation Analysis")
st.markdown("Upload a wound image + mask to analyze segmentation results.")

# =========================================================
# FILE UPLOAD
# =========================================================

col1, col2 = st.columns(2)

with col1:
    image_file = st.file_uploader("Upload Wound Image", type=["png", "jpg", "jpeg"])

with col2:
    mask_file = st.file_uploader("Upload Mask Image", type=["png", "jpg", "jpeg"])

# =========================================================
# RUN PIPELINE
# =========================================================

if image_file and mask_file:

    # Convert uploaded files to arrays

    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    mask = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    # CLEANING STEP (helpers.py)
    image = resize_image(image, 256)
    mask = binarize_mask(mask)
    mask = cv2.resize(mask, (256, 256))
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = mask.astype(np.uint8)

    # contour extraction (inline for UI simplicity)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    wound_percent = compute_wound_percentage(mask)
    perimeter = compute_perimeter(mask)

    # =========================================================
    # DISPLAY RESULTS
    # =========================================================

    st.subheader("📊 Segmentation Metrics")

    c1, c2 = st.columns(2)
  
    with c1:
        st.metric("Wound Area (%)", f"{wound_percent:.2f}%")
      
    with c2:
        st.metric("Perimeter", f"{perimeter:.2f}")

    # =========================================================
    # VISUALIZATION
    # =========================================================

    st.subheader("🖼️ Visual Analysis")

    overlay = overlay_images(image, mask)
    colA, colB, colC = st.columns(3)

    with colA:
        st.image(image, caption="Original Image", use_container_width=True)

    mask_vis = mask.copy()
    if mask_vis.max() <= 1:
        mask_vis = (mask_vis * 255).astype(np.uint8)
    with colB:
        st.image(mask_vis, caption="Binary Mask", use_container_width=True, clamp=True)

    with colC:
        st.image(overlay, caption="Contour Overlay", use_container_width=True)

    # =========================================================
    # EXTRA INSIGHT BLOCK
    # =========================================================

    st.subheader("🧠 Interpretation")

    if wound_percent > 10:
        st.warning("High wound coverage detected — severe case.")
    elif wound_percent > 5:
        st.info("Moderate wound region — healing stage likely ongoing.")
    else:
        st.success("Low wound area — near recovery or minor wound.")

else:
    st.info("Please upload both image and mask to begin analysis.")
