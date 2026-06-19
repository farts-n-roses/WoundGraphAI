# =========================================================
# GRAPH BUILDER
# =========================================================

# Converts wound tissue regions into graph structures for spatio-temporal GNN analysis.

# Nodes  -> image patches
# Edges  -> neighboring tissue regions
# Features:
#   - intensity
#   - texture variation
#   - spatial coordinates
#   - tissue density

# Output: torch_geometric.data.Data object

# =========================================================
# IMPORTS
# =========================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch

from torch_geometric.data import Data

# =========================================================
# SETTINGS
# =========================================================

PATCH_SIZE = 32

# =========================================================
# LOAD IMAGE
# =========================================================

def load_image(image_path):

    image = cv2.imread(str(image_path))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image

# =========================================================
# LOAD MASK
# =========================================================

def load_mask(mask_path):

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    return mask

# =========================================================
# BINARIZE MASK
# =========================================================

def binarize_mask(mask):

    _, binary = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return binary

# =========================================================
# PATCH EXTRACTION
# =========================================================

def extract_patches(image, mask):

    patches = []

    h, w = image.shape[:2]

    for y in range(0, h, PATCH_SIZE):

        for x in range(0, w, PATCH_SIZE):

            img_patch = image[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            mask_patch = mask[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            if img_patch.shape[0] != PATCH_SIZE:
                continue

            if img_patch.shape[1] != PATCH_SIZE:
                continue

            patches.append(
                {
                    "image": img_patch,
                    "mask": mask_patch,
                    "x": x,
                    "y": y
                }
            )

    return patches

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_single_patch_features(patch):

    image = patch["image"]

    mask = patch["mask"]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # =====================================================
    # FEATURE 1 — MEAN INTENSITY
    # =====================================================

    mean_intensity = np.mean(gray)

    # =====================================================
    # FEATURE 2 — TEXTURE / STD
    # =====================================================

    texture_std = np.std(gray)

    # =====================================================
    # FEATURE 3 — WOUND DENSITY
    # =====================================================

    wound_density = (np.sum(mask > 0) / (PATCH_SIZE * PATCH_SIZE))

    # =====================================================
    # FEATURE 4 — X COORDINATE
    # =====================================================

    x_coord = patch["x"]

    # =====================================================
    # FEATURE 5 — Y COORDINATE
    # =====================================================

    y_coord = patch["y"]

    features = [
        mean_intensity,
        texture_std,
        wound_density,
        x_coord,
        y_coord
    ]

    return features

# =========================================================
# GRAPH CONSTRUCTION
# =========================================================

def build_graph(node_features):
    """
    Builds spatial tissue graph from patch feature vectors.
    """
    G = nx.Graph()
    num_nodes = len(node_features)
    
    if num_nodes == 0:
        return G

    # =====================================================
    # ADD NODES
    # =====================================================

    for idx in range(num_nodes):
        G.add_node(idx, features=node_features[idx])
        
    # =====================================================
    # COMPUTE GRID SIZE
    # =====================================================
    
    grid_size = int(np.ceil(np.sqrt(num_nodes)))
    
    # =====================================================
    # CONNECT NEIGHBORS (4-ADJACENCY)
    # =====================================================

    for idx in range(num_nodes):
        row = idx // grid_size
        col = idx % grid_size
        
        # right neighbor
        if col < grid_size - 1:
            neighbor = idx + 1
            if neighbor < num_nodes:
                G.add_edge(idx, neighbor)
        
        # bottom neighbor
        if row < grid_size - 1:
            neighbor = idx + grid_size
            if neighbor < num_nodes:
                G.add_edge(idx, neighbor)

    return G

# =========================================================
# NETWORKX -> PYTORCH GEOMETRIC
# =========================================================

def convert_to_pyg(G, node_features):

    # =====================================================
    # EDGE INDEX
    # =====================================================

    edge_index = list(G.edges())
    
    edge_index = torch.tensor(
        edge_index,
        dtype=torch.long
    ).t().contiguous()

    # Add reverse edges
    reverse_edges = edge_index.flip(0)

    edge_index = torch.cat(
        [edge_index, reverse_edges],
        dim=1
    )

    # =====================================================
    # NODE FEATURES
    # =====================================================

    x = torch.tensor(
        node_features,
        dtype=torch.float
    )

    # =====================================================
    # CREATE DATA OBJECT
    # =====================================================

    data = Data(
        x=x,
        edge_index=edge_index
    )

    return data

# =========================================================
# GRAPH VISUALIZATION
# =========================================================

def visualize_graph(G, patches):

    plt.figure(figsize=(10, 10))

    pos = {}

    # =====================================================
    # USE REAL SPATIAL COORDINATES
    # =====================================================

    for idx, patch in enumerate(patches):

        x = patch["x"]

        y = -patch["y"]

        pos[idx] = (x, y)

    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=80
    )

    plt.title("Spatial Tissue Graph")

    plt.axis("equal")

    plt.show()

# =========================================================
# FULL GRAPH PIPELINE
# =========================================================

def run_graph_builder(
    image_path,
    mask_path,
    visualize=False,
    verbose=False
):

    # =====================================================
    # LOAD DATA
    # =====================================================

    image = load_image(image_path)

    mask = load_mask(mask_path)

    mask = binarize_mask(mask)

    # =====================================================
    # PATCH EXTRACTION
    # =====================================================

    patches = extract_patches(image, mask)

    if verbose:
        print(f"\nTotal Patches: {len(patches)}")

    # =====================================================
    # GRAPH CONSTRUCTION
    # =====================================================

    G, node_features = build_graph(patches)

    if verbose:
        print(f"Total Nodes: {G.number_of_nodes()}")

    print(f"Total Edges: {G.number_of_edges()}")

    # =====================================================
    # CONVERT TO PYG
    # =====================================================

    pyg_data = convert_to_pyg(G, node_features)

    print("\nPyTorch Geometric Data Object:\n")

    print(pyg_data)

    # =====================================================
    # VISUALIZE GRAPH
    # =====================================================

    if visualize:
        visualize_graph(G, patches)

    return pyg_data

# =========================================================
# BUILD GRAPH FROM NUMPY ARRAYS
# =========================================================

def run_graph_builder_from_arrays(
    image,
    mask,
    visualize=False,
    verbose=False
):

    # =====================================================
    # BINARIZE MASK
    # =====================================================

    mask = binarize_mask(mask)

    # =====================================================
    # PATCH EXTRACTION
    # =====================================================

    patches = extract_patches(image, mask)

    if verbose:
        print(f"\nTotal Patches: {len(patches)}")

    # =====================================================
    # GRAPH CONSTRUCTION
    # =====================================================

    G, node_features = build_graph(patches)

    if verbose:
        print(f"Total Nodes: {G.number_of_nodes()}")

    if verbose:
        print(f"Total Edges: {G.number_of_edges()}")

    # =====================================================
    # CONVERT TO PYG
    # =====================================================

    pyg_data = convert_to_pyg(G, node_features)

    if verbose:
        print("\nPyTorch Geometric Data Object:\n")

    if verbose:
        print(pyg_data)

    # =====================================================
    # OPTIONAL VISUALIZATION
    # =====================================================

    if visualize:
        visualize_graph(G, patches)

    return pyg_data

# =========================================================
# TEST RUN
# =========================================================

if __name__ == "__main__":

    image_path = ("dataset/images/train_images/fusc_0002.png")

    mask_path = ("dataset/masks/train_masks/fusc_0002.png")

    run_graph_builder(
        image_path,
        mask_path
    )
