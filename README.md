# 🩹 WoundGraphAI: Spatio-Temporal Graph Neural Network Framework for Epithelial Wound Healing Analysis

## Overview

WoundGraphAI is an AI-powered wound healing analysis platform that combines computer vision, graph representation learning, temporal modelling, and explainable artificial intelligence (XAI) to analyse epithelial wound images and predict healing progression.

The system transforms wound images into spatial tissue graphs, simulates temporal wound contraction, and processes sequential graph representations using a Spatio-Temporal Graph Neural Network (ST-GNN). The framework generates healing predictions, risk assessments, and visual explanations that highlight the tissue regions influencing model decisions.

The project demonstrates how modern graph deep learning techniques can be applied to biomedical image analysis and wound healing research.

---

## Problem Statement

Can wound healing progression be predicted from wound images by modelling tissue structure as a graph and learning both spatial tissue relationships and temporal healing dynamics using Spatio-Temporal Graph Neural Networks?

---

## Objectives

- Perform wound image preprocessing and segmentation
- Extract quantitative wound metrics
- Represent tissue structure as a graph
- Simulate temporal wound progression
- Learn spatial and temporal healing patterns using ST-GNNs
- Predict wound healing outcomes and recovery trajectories
- Classify risk of delayed healing
- Provide explainable AI visualisations for model interpretability
- Develop an interactive clinical analytics dashboard

---

## Tech Stack

### Programming Language

- Python

### Front-End Interface

- Streamlit

### Computer Vision

- OpenCV
- Pillow
- Scikit-Image

### Data Processing

- NumPy
- Pandas

### Graph Learning

- PyTorch
- PyTorch Geometric
- NetworkX

### Machine Learning & Deep Learning

- Scikit-learn
- PyTorch

### Explainable AI

- Grad-CAM-inspired Graph Attribution
- Feature Importance Mapping

### Visualisation

- Matplotlib
- Plotly
- Seaborn

---

## System Architecture

The framework follows a multi-stage pipeline:

1. Image Acquisition
2. Image Preprocessing & Normalisation
3. Wound Segmentation
4. Metric Extraction
5. Temporal Healing Simulation
6. Graph Construction
7. Spatio-Temporal Graph Neural Network Processing
8. Healing Prediction & Risk Assessment
9. Explainable AI Analysis
10. Visualisation & Clinical Insights

---

## Project Structure

```text
WoundGraphAI/
│
├── README.md
├── requirements.txt
├── app.py
├── config.py
│
├── dataset/
│   ├── masks/
│   │   ├── test_masks/
│   │   └── train_masks/
│   ├── precomputed/
│   ├── images/
│   │   ├── test_images/
│   │   └── train_images/
│   └── sample_simulated_sequence/
│
├── pages/
│   ├── 1_Overview.py
│   ├── 2_Segmentation.py
│   ├── 3_Temporal_Analysis.py
│   ├── 4_Graph_Analysis.py
│   ├── 5_Prediction.py
│   ├── 6_Explainable_AI.py
│   └── 7_Insights.py
│
├── models/
│   ├── explainability.py
│   ├── healing_predictor.py
│   ├── segmentation_model.py
│   └── temporal_gnn.py
│
├── processing/
│   ├── feature_extractor.py
│   ├── graph_builder.py
│   ├── metrics.py
│   ├── preprocess.py
│   └── temporal_simulator.py
│
└── utils/
    ├── data_loader.py
    ├── dataset_inspector.py
    ├── helpers.py
    └── visualisation.py
```

---

## Core Features

### 1. Image Preprocessing

- Image resizing and normalisation
- Noise reduction
- Binary mask processing
- Input standardisation

### 2. Wound Segmentation

- Automatic wound region extraction
- Binary wound mask generation
- Region boundary identification

### 3. Quantitative Metric Extraction

- Wound area estimation
- Perimeter calculation
- Tissue density analysis
- Structural feature extraction

### 4. Temporal Healing Simulation

- Progressive wound contraction
- Synthetic healing sequence generation
- Temporal state modelling

### 5. Graph Construction

The wound image is transformed into a tissue graph:

- Nodes → Image patches
- Edges → Spatial adjacency relationships
- Features → Intensity, texture, wound density, spatial position

### 6. Spatio-Temporal Graph Neural Network

The ST-GNN learns:

- Local tissue interactions
- Global wound structure
- Temporal healing progression
- Dynamic tissue evolution patterns

Outputs include:

- Healing score
- Recovery trajectory
- Risk classification

### 7. Prediction Engine

The system predicts:

- Healing progression
- Recovery potential
- Delayed healing risk

Risk Levels:

- Low Risk
- Moderate Risk
- High Risk

### 8. Explainable AI (XAI)

Provides:

- Patch-level importance maps
- Region contribution analysis
- Heatmap overlays
- Prediction interpretability

### 9. Interactive Dashboard

Built using Streamlit.

Includes:

- Original image display
- Segmentation visualisation
- Temporal healing curves
- Graph visualisation
- ST-GNN predictions
- Explainability maps
- Clinical insights

---

## Machine Learning Pipeline

### Stage 1: Preprocessing

- Image normalisation
- Mask generation
- Feature preparation

### Stage 2: Temporal Simulation

- Healing sequence generation
- Multi-time-step representation

### Stage 3: Graph Construction

- Patch extraction
- Node feature computation
- Edge generation

### Stage 4: ST-GNN Learning

- Spatial feature aggregation
- Temporal pattern learning
- Latent representation generation

### Stage 5: Prediction

- Healing score estimation
- Recovery forecasting
- Risk classification

### Stage 6: Explainability

- Feature attribution
- Importance scoring
- Visual explanation generation

---

## Research Contributions

This project introduces:

- Graph-based tissue representation for wound analysis
- Temporal wound progression modelling
- Integration of ST-GNNs with biomedical image analysis
- Explainable AI for clinical interpretability
- Interactive visual analytics for wound assessment

The framework bridges computer vision, graph learning, and biomedical informatics within a unified architecture.

---

## How to Run the Project

### Clone Repository

```bash
git clone https://github.com/farts-n-roses/WoundGraphAI.git
cd WoundGraphAI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## Input Requirements

Supported image formats:

- PNG
- JPG
- JPEG

Required inputs:

- Wound image
- Binary wound mask (optional if automatic segmentation is enabled)

---

## Outputs

The system generates:

- Segmentation masks
- Wound metrics
- Healing progression curves
- Tissue graph visualisations
- Healing predictions
- Risk classifications
- Explainability heatmaps
- Clinical insight reports

---

## Future Improvements

### AI & Deep Learning

- Graph Attention Networks (GAT)
- Graph Transformers
- Dynamic Graph Neural Networks
- Self-Supervised Graph Learning

### Biomedical Extensions

- Multi-class tissue segmentation
- Chronic wound analysis
- Diabetic ulcer assessment
- Histopathology integration

### Explainability

- SHAP for Graph Neural Networks
- GNNExplainer integration
- Counterfactual explanations

### Deployment

- Docker support
- Cloud deployment
- REST API integration
- Clinical decision-support interface

---

## Author

**Fida Fathima**

B.Tech Computer Science Engineering

Computational Biology • Biomedical AI • Graph Neural Networks • Computer Vision • Explainable AI

---

## Academic Context

This project was developed as a Final Year Research Project exploring the application of Spatio-Temporal Graph Neural Networks and Explainable AI for epithelial wound healing analysis and prediction from biomedical images.
