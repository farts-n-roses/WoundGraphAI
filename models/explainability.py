# =========================================================
# EXPLAINABILITY MODULE
# =========================================================

# Simulates explainable AI analysis for:
# - influential tissue regions
# - abnormal wound zones
# - healing importance heatmaps
# - computational tissue attention

# =========================================================
# IMPORTS
# =========================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from processing.preprocess import preprocess_image
from processing.temporal_simulator import binarize_mask

# =========================================================
# GENERATE PATCH IMPORTANCE MAP
# =========================================================

def generate_importance_map(
    mask,
    patch_size=32
):

    """
    Generates a simulated tissue importance map.
    Higher wound-density regions receive
    stronger importance values.
    """

    height, width = mask.shape

    importance_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # =====================================================
    # PATCH ANALYSIS
    # =====================================================

    for y in range(0, height, patch_size):

        for x in range(0, width, patch_size):

            patch = mask[
                y:y + patch_size,
                x:x + patch_size
            ]

            # wound density
            wound_ratio = np.mean(patch > 0)

            # simulate attention weighting
            importance_score = (wound_ratio ** 1.5)

            importance_map[
                y:y + patch_size,
                x:x + patch_size
            ] = importance_score

    # =====================================================
    # NORMALIZE
    # =====================================================

    importance_map = cv2.GaussianBlur(
        importance_map,
        (21, 21),
        0
    )

    importance_map = cv2.normalize(
        importance_map,
        None,
        0,
        1,
        cv2.NORM_MINMAX
    )

    return importance_map

# =========================================================
# OVERLAY HEATMAP
# =========================================================

def overlay_heatmap(
    image,
    heatmap,
    alpha=0.5
):

    """
    Overlays heatmap onto wound image.
    """
    
    # Ensure image is uint8 rgb
    image = np.asarray(image, dtype=np.uint8)

    # convert grayscale only if truly single-channel
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # otherwise RGBA
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # RGBA -> RGB safety
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    image = image.astype(np.uint8)
    
    # Heatmap Safety
    
    # Normalize Heatmap
    heatmap = np.asarray(heatmap, dtype=np.float32)
    
    # ensure same spatial size
    # resize to image dimensions
    heatmap = cv2.resize(
        heatmap,
        (image.shape[1], image.shape[0])
    )
    
    # Normalize to 0-255
    heatmap = cv2.normalize(
        heatmap,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )
    
    heatmap = np.uint8(heatmap)

    # Convert heatmap to color map
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Final Blend
    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return overlay

# =========================================================
# VISUALIZE EXPLAINABILITY
# =========================================================

def visualize_explainability(
    image,
    mask,
    importance_map,
    overlay
):

    """
    Displays:
    - original image
    - wound mask
    - importance heatmap
    - overlay visualization
    """

    _, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5)
    )

    image_uint8 = np.uint8(image)

    axes[0].imshow(
        cv2.cvtColor(
            image_uint8,
            cv2.COLOR_BGR2RGB
        )
    )

    axes[0].set_title("Original Image")

    axes[0].axis("off")

    # =====================================================

    axes[1].imshow(
        mask,
        cmap="gray"
    )

    axes[1].set_title(
        "Binary Wound Mask"
    )

    axes[1].axis("off")

    # =====================================================

    axes[2].imshow(
        importance_map,
        cmap="jet"
    )

    axes[2].set_title(
        "Importance Heatmap"
    )

    axes[2].axis("off")

    # =====================================================
    
    overlay_uint8 = np.uint8(overlay)

    axes[3].imshow(
        cv2.cvtColor(
            overlay_uint8,
            cv2.COLOR_BGR2RGB
        )
    )

    axes[3].set_title(
        "Explainable AI Overlay"
    )

    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

# =========================================================
# TEST PIPELINE
# =========================================================

if __name__ == "__main__":

    print("\n====================================")
    print("EXPLAINABLE AI ANALYSIS")
    print("====================================\n")

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    image_path = (
        "dataset/images/train_images/fusc_0016.png"
    )

    mask_path = (
        "dataset/masks/train_masks/fusc_0016.png"
    )

    image = cv2.imread(image_path)
    image = preprocess_image(image)

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    mask = cv2.resize(
        mask,
        (256, 256)
    )

    mask = binarize_mask(mask)

    # =====================================================
    # GENERATE IMPORTANCE
    # =====================================================

    importance_map = (
        generate_importance_map(
            image,
            mask
        )
    )

    # =====================================================
    # OVERLAY
    # =====================================================

    overlay = overlay_heatmap(
        image,
        importance_map
    )

    # =====================================================
    # VISUALIZATION
    # =====================================================

    visualize_explainability(
        image,
        mask,
        importance_map,
        overlay
    )

    print("Explainability analysis completed.\n")
