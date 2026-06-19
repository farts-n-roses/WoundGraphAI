# =========================================================
# OVERVIEW PAGE
# =========================================================

import streamlit as st
import cv2
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from processing.preprocess import preprocess_image
from utils.data_loader import load_dataset, dataset_summary, train_test_split
from utils.helpers import resize_image, binarize_mask
from processing.metrics import compute_wound_percentage, compute_perimeter

@st.cache_data
def load_data():
    dataset = load_dataset(
        "dataset/raw_images/train_images",
        "dataset/masks/train_masks"
    )
    train_data, test_data = train_test_split(dataset)
    return train_data, test_data
train_data, test_data = load_data()
st.write(dataset_summary(train_data))

# optional sanity check using metrics module
sample_img, sample_mask = train_data[0]

wound_pct = compute_wound_percentage(sample_mask)
perimeter = compute_perimeter(sample_mask)

print("\n================ QUICK METRICS CHECK ================")
print(f"Wound %: {wound_pct:.2f}")
print(f"Perimeter: {perimeter:.2f}")
print("====================================================\n")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Wound Healing System - Overview",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🧬 Wound Healing Analysis System")

st.markdown("""
A **Spatio-Temporal Graph Neural Network framework**
for computational modeling of epithelial wound healing.

This system performs:
- Image preprocessing
- Wound segmentation
- Temporal healing simulation
- Graph-based tissue modeling
- Predictive analysis
- Explainable AI visualization
""")

# =========================================================
# SIDEBAR INFO
# =========================================================

st.header("📊 System Summary")

st.markdown(
    """
- Segmentation: OpenCV / thresholding  
- Temporal Modeling: Simulated healing progression  
- Graph Model: Patch-based tissue graph  
- AI Model: Temporal GNN  
- Explainability: Heatmap-based importance mapping  
"""
)

# =========================================================
# OPTIONAL QUICK DEMO INPUT
# =========================================================

st.markdown("---")

st.subheader("📁 Quick Demo Upload (Optional)")

img_file = st.file_uploader(
    "Upload a wound image to preview pipeline",
    type=["png", "jpg", "jpeg"]
)

if img_file:
    # READ IMAGE
    image = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # CLEAN USING HELPERS
    image_resized = resize_image(image, 256)
    image_processed = preprocess_image(image_resized)

    # OPTIONAL MASK SIMULATION (for demo only)
    dummy_mask = np.zeros((256, 256), dtype=np.uint8)
    dummy_mask = binarize_mask(dummy_mask)

    # DISPLAY/SHOW RESULTS
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_file, caption="Original Image", use_container_width=True)
    with col2:
        st.image(image_processed, caption="Preprocessed Image", use_container_width=True)
    st.success("Basic preprocessing pipeline executed successfully.")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.info(
    """
This system is a computational research prototype
for modeling wound healing dynamics using AI.

It is not intended for clinical diagnosis.
"""
)
