# =========================================================
# GRAPH ANALYSIS PAGE
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

from matplotlib import patches
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import networkx as nx

# from processing.graph_builder import (
#     load_image,
#     load_mask,
#     binarize_mask,
#     extract_patches,
#     build_graph
# )
from utils.helpers import resize_image, binarize_mask
from processing.feature_extractor import extract_patch_features
from processing.metrics import compute_wound_percentage
from processing.graph_builder import build_graph
#, extract_single_patch_features

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Graph Tissue Analysis",
    page_icon="🕸️",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🕸️ Spatial Tissue Graph Analysis")

st.markdown("""
This module converts wound tissue regions into
graph-based spatial representations for
computational biological modeling and
Graph Neural Network analysis.
""")

# =========================================================
# FILE UPLOAD
# =========================================================

col1, col2 = st.columns(2)

with col1:
    uploaded_image = st.file_uploader(
        "Upload Wound Image",
        type=["png", "jpg", "jpeg"]
    )

with col2:
    uploaded_mask = st.file_uploader(
        "Upload Wound Mask",
        type=["png", "jpg", "jpeg"]
    )

# =========================================================
# PROCESSING
# =========================================================

if uploaded_image and uploaded_mask:

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    image = cv2.imdecode(
        np.frombuffer(uploaded_image.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    mask = cv2.imdecode(
        np.frombuffer(uploaded_mask.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    # CLEAN USING HELPERS (IMPORTANT)
    image = resize_image(image, 256)
    mask = resize_image(mask, 256)
    mask = binarize_mask(mask)
    
    # AUTO-FIX INVERTED MASKS
    white_ratio = np.mean(mask > 0)

    # if most of image is white, mask is probably inverted
    if white_ratio > 0.5:
        mask = 255 - mask

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # =====================================================
    # PATCH EXTRACTION
    # =====================================================

    node_features, patches = extract_patch_features(image, mask)

    # =====================================================
    # BUILD GRAPH
    # =====================================================

    G = build_graph(node_features)

    # =====================================================
    # GRAPH METRICS
    # =====================================================

    num_nodes = G.number_of_nodes()

    num_edges = G.number_of_edges()

    if num_nodes > 0:
        avg_degree = sum(dict(G.degree()).values()) / num_nodes
    else:
        avg_degree = 0

    # =====================================================
    # DASHBOARD METRICS
    # =====================================================

    st.subheader("📊 Graph Statistics")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Nodes", num_nodes)

    c2.metric("Total Edges", num_edges)

    c3.metric("Average Degree", f"{avg_degree:.2f}")

    # =====================================================
    # IMAGE + MASK
    # =====================================================

    st.subheader("🖼️ Tissue Inputs")

    colA, colB = st.columns(2)

    with colA:
        st.image(image, caption="Original Wound Image", use_container_width=True)

    with colB:
        st.image(mask, caption="Binary Wound Mask", use_container_width=True)

    # =====================================================
    # GRAPH VISUALIZATION
    # =====================================================

    st.subheader("🕸️ Spatial Tissue Graph")

    fig, ax = plt.subplots(figsize=(10, 10))
    
    grid_size = int(np.ceil(np.sqrt(num_nodes)))
    pos = {}
    
    for idx in range(num_nodes):

        row = idx // grid_size
        col = idx % grid_size

        pos[idx] = (col, -row)

    # =====================================================
    # NODE COLORS
    # =====================================================

    node_colors = []

    WOUND_DENSITY_IDX = 2
    node_colors = np.array([f[WOUND_DENSITY_IDX] for f in node_features], dtype=np.float32)
    node_colors = node_colors ** 0.5  # enhance contrast and amplify visibility of low-density nodes
    node_sizes = [
        700 * f[2] + 40
        for f in node_features
    ]
    
    nx.draw(G, pos, ax=ax, with_labels=False, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.Reds, vmin=0, vmax=1)

    ax.set_title("Spatial Tissue Interaction Graph")

    ax.axis("equal")

    st.pyplot(fig)

    # =====================================================
    # FEATURE INSIGHTS
    # =====================================================

    st.subheader("🧠 Graph-Based Tissue Insights")

    high_density_nodes = sum(1 for f in node_features if f[2] > 0.05)

    st.markdown(f"""
    ### Spatial Graph Interpretation

    - Total tissue patches analyzed: **{num_nodes}**
    - Spatial tissue interactions: **{num_edges}**
    - High wound-density regions: **{high_density_nodes}**

    ### Biological Interpretation

    Each graph node represents a localized tissue
    microenvironment extracted from the wound image.

    Edges represent spatial tissue adjacency,
    enabling computational modeling of:

    - local tissue interactions
    - wound propagation structure
    - spatial healing relationships
    - tissue neighborhood dynamics

    Regions with stronger red coloration indicate:
    - higher wound density
    - stronger tissue damage presence
    - greater biological abnormality

    This graph structure forms the foundation for:
    - Graph Neural Networks (GNNs)
    - spatio-temporal tissue modeling
    - computational wound progression analysis
    """)

else:
    st.info(
        "Please upload both wound image and mask "
        "to begin graph analysis."
    )
