# =========================================================
# VISUALIZATION MODULE
# =========================================================

import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# BASIC DISPLAY FUNCTIONS
# =========================================================

def show_image(img, title="Image"):
    """
    Display a single image.
    """
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_mask(mask, title="Mask"):
    """
    Display a segmentation mask.
    """
    plt.figure()
    plt.imshow(mask, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

# =========================================================
# OVERLAY FUNCTION (IMPORTANT FOR BIOMEDICAL VALIDATION)
# =========================================================

def overlay_mask(image, mask, alpha=0.4):
    """
    Overlay mask on image for alignment checking.
    """
    plt.figure()

    plt.imshow(image)

    # red overlay highlights wound region
    plt.imshow(mask, cmap="Reds", alpha=alpha)

    plt.title("Image + Mask Overlay")
    plt.axis("off")

    plt.show()

# =========================================================
# SIDE-BY-SIDE COMPARISON
# =========================================================

def compare_pipeline(raw_img, processed_img, mask):
    """
    Compare raw, processed, and mask side by side.
    """
    _, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(raw_img)
    ax[0].set_title("Raw Image")
    ax[0].axis("off")

    ax[1].imshow(processed_img)
    ax[1].set_title("Preprocessed Image")
    ax[1].axis("off")

    ax[2].imshow(mask, cmap="gray")
    ax[2].set_title("Mask")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

# =========================================================
# QUICK DEBUG GRID (OPTIONAL BUT USEFUL)
# =========================================================

def show_grid(images, titles=None, cols=3):
    """
    Display multiple images in grid format.
    """
    rows = int(np.ceil(len(images) / cols))

    plt.figure(figsize=(4 * cols, 4 * rows))

    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)

        if titles:
            plt.title(titles[i])

        plt.axis("off")

    plt.tight_layout()
    plt.show()
