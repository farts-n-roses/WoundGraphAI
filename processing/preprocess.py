# =========================================================
# PREPROCESSING PIPELINE
# =========================================================

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from pathlib import Path
from config import IMAGE_SIZE

# =========================================================
# IMAGE LOADING
# =========================================================

def load_image(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return mask

# =========================================================
# RESIZE
# =========================================================

def resize_image(img, size=IMAGE_SIZE):
    return cv2.resize(img, (size, size))


def resize_mask(mask, size=IMAGE_SIZE):
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

# =========================================================
# NORMALIZATION
# =========================================================

def normalize_image(img):
    return img / 255.0

# =========================================================
# MASK BINARY CONVERSION
# =========================================================

def binarize_mask(mask):
    _, binary = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)

# =========================================================
# FULL PIPELINE
# =========================================================

def preprocess_image(img):
    img = resize_image(img)
    img = normalize_image(img)
    return img


def preprocess_mask(mask):
    mask = resize_mask(mask)
    mask = binarize_mask(mask)
    return mask

# =========================================================
# DEBUG VISUALIZATION (VERY IMPORTANT)
# =========================================================

def show_sample(img, mask):
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].imshow(img)
    ax[0].set_title("Preprocessed Image")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Preprocessed Mask")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()
    
# =========================================================
# TESTING THE PIPELINE
# =========================================================

if __name__ == "__main__":
    from utils.dataset_inspector import get_image_files

    img_path = "dataset/images/train_images/fusc_0002.png"
    mask_path = "dataset/masks/train_masks/fusc_0002.png"

    img = load_image(img_path)
    mask = load_mask(mask_path)

    img_p = preprocess_image(img)
    mask_p = preprocess_mask(mask)

    print("Image shape:", img_p.shape)
    print("Mask unique values:", np.unique(mask_p))

    show_sample(img_p, mask_p)
    
# =========================================================
# TESTING VISUALIZATION FUNCTIONS 
# =========================================================

if __name__ == "__main__":
    from utils.visualisation import overlay_mask, compare_pipeline
    from utils.dataset_inspector import get_image_files
    import cv2

    # pick sample files
    img_path = "dataset/images/train_images/fusc_0002.png"
    mask_path = "dataset/masks/train_masks/fusc_0002.png"

    # load raw
    img_raw = cv2.imread(img_path)
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # fake processed version for now (replace with your preprocess later)
    img_proc = img_raw / 255.0

    # visualization tests
    overlay_mask(img_raw, mask)
    compare_pipeline(img_raw, img_proc, mask)
