# =========================================================
# SEGMENTATION MODULE (CLASSICAL CV BASELINE)
# =========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# LOAD FUNCTIONS
# =========================================================

def load_image(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return mask

# =========================================================
# PREPROCESS MASK (ENSURE CLEAN BINARY)
# =========================================================

def binarize_mask(mask):
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary

# =========================================================
# CONTOUR EXTRACTION
# =========================================================

def extract_contours(mask):
    """
    Extract wound boundary contours.
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours

# =========================================================
# AREA CALCULATION
# =========================================================

def compute_wound_area(mask):
    """
    Returns wound area in pixels and percentage.
    """
    wound_pixels = np.sum(mask > 0)
    total_pixels = mask.shape[0] * mask.shape[1]

    area_percent = (wound_pixels / total_pixels) * 100

    return wound_pixels, area_percent

# =========================================================
# PERIMETER CALCULATION
# =========================================================

def compute_perimeter(contours):
    """
    Compute total boundary length.
    """
    perimeter = 0

    for cnt in contours:
        perimeter += cv2.arcLength(cnt, True)

    return perimeter

# =========================================================
# VISUALIZATION
# =========================================================

def show_segmentation(image, mask, contours):
    """
    Overlay contour on image.
    """
    overlay = image.copy()

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (255, 0, 0),  # red boundary
        2
    )

    _, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Mask")
    ax[1].axis("off")

    ax[2].imshow(overlay)
    ax[2].set_title("Contour Overlay")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

# =========================================================
# FULL PIPELINE FUNCTION
# =========================================================

def run_segmentation(image_path, mask_path):
    """
    End-to-end segmentation analysis.
    """

    # Load
    image = load_image(image_path)
    mask = load_mask(mask_path)

    # Ensure binary mask
    mask = binarize_mask(mask)

    # Contours
    contours = extract_contours(mask)

    # Metrics
    area_px, area_percent = compute_wound_area(mask)
    perimeter = compute_perimeter(contours)

    # Output
    print("\n================ SEGMENTATION RESULTS ================\n")
    print(f"Wound Area (pixels): {area_px}")
    print(f"Wound Area (%): {area_percent:.2f}%")
    print(f"Perimeter: {perimeter:.2f}")
    print("\n======================================================\n")

    # Visualization
    show_segmentation(image, mask, contours)

    return {
        "area_px": area_px,
        "area_percent": area_percent,
        "perimeter": perimeter
    }

# =========================================================
# TEST RUN
# =========================================================

if __name__ == "__main__":

    image_path = "dataset/raw_images/train_images/fusc_0002.png"
    mask_path = "dataset/masks/train_masks/fusc_0002.png"

    run_segmentation(image_path, mask_path)
