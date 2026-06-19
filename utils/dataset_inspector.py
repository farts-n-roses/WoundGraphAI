# DATASET INSPECTOR

from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to Python path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# =========================================================
# IMPORT CONFIG
# =========================================================

from config import (
    RAW_IMAGES_DIR,
    MASKS_DIR,
    DATASET_DIR,
)

# =========================================================
# DATASET PATHS
# =========================================================

TRAIN_IMAGES_DIR = RAW_IMAGES_DIR / "train_images"
TRAIN_MASKS_DIR = MASKS_DIR / "train_masks"

TEST_IMAGES_DIR = RAW_IMAGES_DIR / "test_images"
TEST_MASKS_DIR = MASKS_DIR / "test_masks"

PRECOMPUTED_DIR = DATASET_DIR / "precomputed"

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_image_files(folder_path):
    """
    Returns sorted list of image files.
    """

    valid_extensions = [".png", ".jpg", ".jpeg"]

    return sorted([
        file for file in folder_path.iterdir()
        if file.suffix.lower() in valid_extensions
    ])


def load_image(image_path):
    """
    Load image using PIL.
    """

    return Image.open(image_path)


def print_dataset_summary():
    """
    Prints dataset statistics.
    """

    train_images = get_image_files(TRAIN_IMAGES_DIR)
    train_masks = get_image_files(TRAIN_MASKS_DIR)

    test_images = get_image_files(TEST_IMAGES_DIR)
    test_masks = get_image_files(TEST_MASKS_DIR)

    print("\n================ DATASET SUMMARY ================\n")

    print(f"Train Images : {len(train_images)}")
    print(f"Train Masks  : {len(train_masks)}")

    print()

    print(f"Test Images  : {len(test_images)}")
    print(f"Test Masks   : {len(test_masks)}")

    print("\n=================================================\n")


def inspect_sample_pair(index=0):
    """
    Displays sample image-mask pair.
    """

    train_images = get_image_files(TRAIN_IMAGES_DIR)
    train_masks = get_image_files(TRAIN_MASKS_DIR)

    image_path = train_images[index]
    mask_path = train_masks[index]

    image = load_image(image_path)
    mask = load_image(mask_path)

    image_np = np.array(image)
    mask_np = np.array(mask)

    print("\n================ SAMPLE INSPECTION ================\n")

    print(f"Image File : {image_path.name}")
    print(f"Mask File  : {mask_path.name}")

    print()

    print(f"Image Shape : {image_np.shape}")
    print(f"Mask Shape  : {mask_np.shape}")

    print()

    print(f"Image Min Pixel : {image_np.min()}")
    print(f"Image Max Pixel : {image_np.max()}")

    print()

    print(f"Mask Unique Values : {np.unique(mask_np)}")

    print("\n===================================================\n")

    # -------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------

    _, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Segmentation Mask")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def inspect_pt_files():
    """
    Inspect .pt files inside precomputed directory.
    """

    if not PRECOMPUTED_DIR.exists():
        print("\nNo precomputed directory found.\n")
        return

    pt_files = sorted(PRECOMPUTED_DIR.glob("*.pt"))

    if len(pt_files) == 0:
        print("\nNo .pt files found.\n")
        return

    print("\n================ PT FILE INSPECTION ================\n")

    print(f"Found {len(pt_files)} .pt files.\n")

    sample_pt = pt_files[0]

    print(f"Loading: {sample_pt.name}\n")

    data = torch.load(sample_pt, weights_only=False)

    print(f"Object Type: {type(data)}\n")

    # -------------------------------------------------
    # IF DICTIONARY
    # -------------------------------------------------

    if isinstance(data, dict):

        print("Dictionary Keys:\n")

        for key in data.keys():
            print(f"- {key}")

    # -------------------------------------------------
    # IF LIST OR TUPLE
    # -------------------------------------------------

    elif isinstance(data, (list, tuple)):

        print(f"Length: {len(data)}")

        if len(data) > 0:
            print(f"First Element Type: {type(data[0])}")

    # -------------------------------------------------
    # OTHERWISE
    # -------------------------------------------------

    else:
        print(data)

    print("\n====================================================\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print_dataset_summary()

    inspect_sample_pair(index=0) # inspect_pt_files()
