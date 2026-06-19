# =========================================================
# MAIN DASHBOARD / LANDING PAGE / APP ENTRY POINT
# =========================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Wound Healing AI System",
    layout="wide",
    page_icon="🧬"
)

# =========================================================
# HOME LANDING
# =========================================================

st.markdown("---")

st.title("🧬 AI-Based Spatio-Temporal Wound Healing Analysis Platform")

st.markdown("""Welcome to the **Wound Healing AI Analysis platform** - a comprehensive tool for analyzing and predicting wound healing progression using advanced computational techniques.""")

st.markdown("""
This platform is a modular **Spatio-Temporal Graph Neural Network framework** for computational modeling of epithelial wound healing.
It combines image processing, graph-based tissue representation, temporal modeling, and deep learning for computational wound analysis.

This integrated computational framework for analyzing epithelial wound healing includes the modules of:

- Image preprocessing
- Segmentation analysis
- Temporal healing simulation
- Temporal tissue dynamics and analysis
- Spatial graph modeling
- Graph Neural Network prediction
- Explainable AI heatmaps
- Biological interpretation and insights
""")

st.info("Note: This system is a research prototype and not intended for clinical diagnosis.")

st.markdown("---")

# =========================================================
# PROJECT OBJECTIVE
# =========================================================

st.header("🎯 Project Objective")

st.markdown("""
The objective of this project is to develop a computational framework capable of analyzing wound healing dynamics using spatial and temporal modeling.

The system integrates:

- image-based wound segmentation
- patch-level tissue feature extraction
- graph-based spatial tissue representation
- temporal healing simulation
- Graph Neural Network prediction
- explainable AI visualization

to model biologically relevant wound healing behavior.
""")

# =========================================================
# SYSTEM WORKFLOW
# =========================================================

st.header("⚙️ System Workflow")

st.markdown("""
### Computational Pipeline

1. Upload wound image and binary mask  
2. Perform segmentation and wound quantification  
3. Extract patch-level tissue features  
4. Construct spatial tissue graphs  
5. Simulate temporal healing progression  
6. Predict healing trajectory using Temporal GNN  
7. Generate explainability heatmaps  
8. Produce biological interpretation and insights
""")

# =========================================================
# MODULE OVERVIEW
# =========================================================

st.header("🧩 System Modules")

col1, col2 = st.columns(2)

with col1:

    st.subheader("🩹 Segmentation")
    st.write("""
    Quantifies wound area and extracts
    primary wound morphology information.
    """)

    st.subheader("📈 Temporal Analysis")
    st.write("""
    Simulates wound healing progression
    using nonlinear temporal decay dynamics.
    """)

    st.subheader("🕸️ Graph Analysis")
    st.write("""
    Converts tissue regions into graph
    structures for spatial modeling.
    """)

with col2:

    st.subheader("🧠 Prediction")
    st.write("""
    Predicts healing trajectory using
    a Spatio-Temporal Graph Neural Network.
    """)

    st.subheader("🔥 Explainable AI")
    st.write("""
    Visualizes biologically influential
    wound regions using importance mapping.
    """)

    st.subheader("🔬 Insights")
    st.write("""
    Translates computational outputs into
    biological healing interpretations.
    """)

st.markdown("---")

# =========================================================
# DATASET INFORMATION
# =========================================================

st.header("🗂️ Dataset Information")

st.markdown("""
The system operates on wound image datasets
containing:

- wound images
- binary segmentation masks

Masks are used to identify wound regions and
enable spatial tissue analysis.

Patch-wise extraction is used to convert
tissue regions into graph-compatible features.
""")

# =========================================================
# MODEL SUMMARY
# =========================================================

st.header("🧠 Model Architecture Summary")

st.markdown("""
### Core Components

- Image preprocessing pipeline
- Patch-wise feature extraction
- Spatial graph construction
- Temporal healing simulator
- Temporal Graph Neural Network
- Explainability heatmap generation

### Node Features

Each graph node contains:
- intensity information
- texture variation
- wound density
- spatial coordinates

### Graph Representation

- nodes → tissue patches
- edges → spatial adjacency relationships

### Temporal Modeling

Healing progression is modeled using:
- temporal graph sequences
- nonlinear healing dynamics
- simulated epithelial contraction behavior
""")

# =========================================================
# BIOLOGICAL RELEVANCE
# =========================================================

st.header("🧬 Biological Relevance")

st.markdown("""
The framework attempts to computationally
approximate biological wound healing behavior,
including:

- epithelial contraction
- spatial tissue interaction
- nonlinear healing progression
- tissue remodeling dynamics

The graph-based formulation enables modeling of
localized tissue microenvironments and their
spatial relationships during healing progression.
""")

st.markdown("---")

# =========================================================
# NAVIGATION GUIDE
# =========================================================

st.header("🧭 Navigation")

st.info("""Use the sidebar to navigate through the
different computational analysis modules.""")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("""Final Year Project — WoundGraphAI""")
st.caption("""Developed by: Fida Fathima""")
