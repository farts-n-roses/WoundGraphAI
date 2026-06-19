# =========================================================
# FEATURE EXTRACTOR
# =========================================================

# Converts wound image + mask → patch-based feature vectors
# for graph neural networks and temporal modeling

# =========================================================

import numpy as np
import cv2

# =========================================================
# CORE FEATURE EXTRACTION
# =========================================================

def extract_patch_features(image, mask, patch_size=16):
    """
    Converts image + mask into patch-wise feature vectors.

    Each patch becomes a graph node.

    Features per node:
    - mean intensity
    - standard deviation (texture proxy)
    - wound density ratio
    - normalized x position
    - normalized y position
    """

    # ensure correct format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]

    features = []
    patches = []

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):

            img_patch = image[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]

            if img_patch.size == 0:
                continue

            # =====================================================
            # FEATURE 1: intensity (grayscale mean)
            # =====================================================
            gray_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray_patch)

            # =====================================================
            # FEATURE 2: texture (std dev)
            # =====================================================
            std_intensity = np.std(gray_patch)

            # =====================================================
            # FEATURE 3: wound density
            # =====================================================
            wound_density = np.mean(mask_patch > 0)

            # =====================================================
            # FEATURE 4-5: spatial coordinates
            # =====================================================
            x_norm = x / w
            y_norm = y / h

            # =====================================================
            # NODE FEATURE VECTOR
            # =====================================================
            # if wound_density < 0.01: # filter out empty patches
            #     continue
            
            features.append([
                mean_intensity,
                std_intensity,
                wound_density,
                x_norm,
                y_norm
            ])
            
            patches.append({
                "x": x,
                "y": y,
                "wound_density": wound_density
            })
    
    features_array = np.array(features, dtype=np.float32)
    
    # ensure proper 2D shape even if only one patch
    if features_array.ndim == 1 and len(features_array) > 0:
        features_array = features_array.reshape(1, -1)
    
    # handle empty extraction safely
    if len(features_array) == 0:
        features_array = np.empty((0, 5), dtype=np.float32)

    return features_array, patches


# =========================================================
# OPTIONAL: SIMPLE FEATURE NORMALIZATION
# =========================================================

def normalize_features(features):
    """
    Normalize feature matrix to [0, 1] range per column.
    """

    features = np.array(features, dtype=np.float32)

    min_vals = features.min(axis=0)
    max_vals = features.max(axis=0)

    # avoid divide-by-zero
    denom = (max_vals - min_vals) + 1e-8

    return (features - min_vals) / denom


# =========================================================
# DEBUG UTILITY
# =========================================================

def print_feature_summary(features):
    """
    Prints quick debugging summary of extracted features.
    """

    print("\n================ FEATURE SUMMARY ================")
    print(f"Total Nodes: {len(features)}")
    print(f"Feature Dimension: {features.shape[1]}")

    print("\nMean per feature:")
    print(np.mean(features, axis=0))

    print("=================================================\n")
