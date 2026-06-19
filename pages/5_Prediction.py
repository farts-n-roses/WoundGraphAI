# =========================================================
# STREAMLIT PAGE:
# HEALING PREDICTION ANALYSIS
# =========================================================

# Displays:
# - healing prediction
# - risk analysis
# - recovery estimation
# - clinical interpretation
# - healing progression visualization

# =========================================================
# IMPORTS
# =========================================================

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utils.helpers import resize_image, binarize_mask

from processing.feature_extractor import extract_patch_features
from processing.graph_builder import build_graph, convert_to_pyg
from processing.metrics import compute_wound_percentage
from processing.temporal_simulator import (
    generate_healing_sequence,
    compute_healing_percentages,
)

from models.healing_predictor import classify_healing
from models.temporal_gnn import TemporalGNN
from config import DEVICE

# =========================================================
# PAGE CONFIG
# =========================================================

@st.cache_resource
def load_model():
    model = TemporalGNN().to(DEVICE)
    model.eval()
    return model

model = load_model()

st.set_page_config(
    page_title="Healing Prediction",
    layout="wide"
)

# =========================================================
# PAGE TITLE
# =========================================================

st.title("🧠 Predictive Wound Healing Analysis")

st.markdown("""
This module performs computational prediction of
future wound healing progression using a
Spatio-Temporal Graph Neural Network framework.
""")

# =========================================================
# HEALING PROGRESSION CURVE
# =========================================================

st.markdown("---")

st.subheader("📉 Predicted Healing Trajectory")

img_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
mask_file = st.file_uploader("Upload Mask", type=["png","jpg","jpeg"])

if img_file and mask_file:
    # Load and Preprocess
    image = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
    mask = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    image = resize_image(image, 256)
    mask = binarize_mask(mask)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    graph_sequence = []
    
    masks = generate_healing_sequence(mask, healing_rate=0.12)
    
    for m in masks:
        # Feature Extraction
        node_features, _ = extract_patch_features(image, m)
        node_features = np.asarray(node_features, dtype=np.float32)
        
        # Graph Building
        G = build_graph(node_features)
        
        if G is None:
            st.error("Graph construction failed")
            st.stop()
        
        # Convert to PyTorch Geometric
        pyg_graph = convert_to_pyg(G, node_features)

        graph_sequence.append(pyg_graph.to(DEVICE))
        # graph_sequence.append(g.to(DEVICE))
    
    with torch.no_grad():
        prediction = model(graph_sequence)
    
    healing_score = prediction.item()
    
    # Metrics
    wound_pct = compute_wound_percentage(mask)
    
    risk = classify_healing(healing_score)[1]

    # Temporal Simulation
    remaining = compute_healing_percentages(masks)

    days = [0, 3, 5, 7, 10, 14]
    healed = [100 - r for r in remaining]
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(days, healed, marker="o", linewidth=3)
    ax.set_title("Healing Trajectory")
    ax.set_xlabel("Days")
    ax.set_ylabel("Wound Recovery %")
    ax.grid(True)

    st.pyplot(fig)
    
    # Metrics Display
    st.subheader("📊 Prediction Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Healing Score", f"{healing_score:.2f}")
    col2.metric("Wound Area (%)", f"{wound_pct:.2f}%")
    col3.metric("Risk Level", risk)
    
    # Interpretation
    if healing_score > 0.7:
        st.success("""
        Strong wound healing progression detected.

        - Efficient epithelial closure
        - Stable spatial recovery dynamics
        - Low risk of delayed healing
        """)

    elif healing_score > 0.4:
        st.warning("""
        Moderate healing progression observed.

        - Partial tissue recovery
        - Slower contraction behavior
        - Monitoring recommended
        """)

    else:
        st.error("""
        High delayed-healing risk detected.

        - Impaired contraction dynamics
        - Abnormal spatial recovery
        - Elevated risk of complications
        """)
