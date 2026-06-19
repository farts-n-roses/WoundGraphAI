# =========================================================
# HELPERS MODULE
# =========================================================
# Common utility functions used across:
# - preprocessing
# - segmentation
# - graph building
# - temporal simulation
# - visualization
# =========================================================

import cv2
import numpy as np


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def resize_image(img, size=256):
    """
    Resizes image to fixed square size.
    """
    if img is None:
        raise ValueError("Input image is None")

    return cv2.resize(img, (size, size))


def to_grayscale(img):
    """
    Converts RGB/BGR image to grayscale.
    """
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_image(img):
    """
    Normalizes image to [0,1].
    """
    img = img.astype(np.float32)
    return img / 255.0


# =========================================================
# MASK PROCESSING
# =========================================================

def binarize_mask(mask):
    """
    Converts mask to binary format (0 or 255).
    Ensures wound areas are white (255) and background is black (0).
    Automatically fixes inverted masks.
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # AUTO-INVERSION FIX

    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)

    # if white dominates massively,
    # mask is probably inverted
    white_ratio = white_pixels / (white_pixels + black_pixels)

    # invert ONLY if absurdly white
    if white_ratio > 0.90:
        binary = cv2.bitwise_not(binary)

    return binary


def invert_mask(mask):
    """
    Inverts binary mask (useful for debugging).
    """
    return 255 - mask


# =========================================================
# SAFE TYPE CONVERSIONS (IMPORTANT FOR YOUR ERRORS)
# =========================================================

def to_uint8(img):
    """
    Ensures image is uint8 (fixes OpenCV errors).
    """
    return np.clip(img, 0, 255).astype(np.uint8)


def to_float(img):
    """
    Ensures float32 format (for ML pipelines).
    """
    return img.astype(np.float32)


# =========================================================
# VISUALIZATION HELPERS
# =========================================================

def overlay_images(image, mask, alpha=0.5):
    """
    Overlays mask on image for visualization.
    """

    # STEP 1: ensure same size
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # STEP 2: convert mask → 3 channels
    if len(mask.shape) == 2:
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        mask_colored = mask

    # STEP 3: ensure uint8 format
    image = image.astype(np.uint8)
    mask_colored = mask_colored.astype(np.uint8)

    # STEP 4: blend safely
    return cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)


# =========================================================
# DEBUG HELPERS
# =========================================================

def print_image_info(img, name="Image"):
    """
    Prints shape and type info for debugging.
    """

    print("\n================ IMAGE INFO ================")
    print(f"{name} Shape: {img.shape}")
    print(f"{name} dtype: {img.dtype}")
    print(f"Min: {np.min(img)} | Max: {np.max(img)}")
    print("===========================================\n")


def assert_valid_image(img):
    """
    Ensures image is valid before processing.
    """

    if img is None:
        raise ValueError("Invalid image: None")

    if not isinstance(img, np.ndarray):
        raise TypeError("Image must be numpy array")

    return True
