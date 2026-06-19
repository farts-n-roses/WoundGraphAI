# =========================================================
# DATA LOADER MODULE
# =========================================================
# Loads wound image-mask dataset for:
# - segmentation
# - temporal simulation
# - graph construction
# - ML pipeline training
# =========================================================

import os
import cv2
from pathlib import Path
import random

# =========================================================
# BASIC LOADERS
# =========================================================

def load_image(path):
    """Loads RGB image from disk."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    """Loads grayscale binary mask."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    return mask


# =========================================================
# PAIR MATCHING
# =========================================================

def match_image_mask(image_dir, mask_dir):
    """
    Matches images and masks by filename.
    Assumes: fusc_0001.png exists in both folders.
    """

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    image_files = sorted(os.listdir(image_dir))
    # mask_files = sorted(os.listdir(mask_dir))

    pairs = []

    for img_name in image_files:

        mask_name = img_name  # same filename assumption

        img_path = image_dir / img_name
        mask_path = mask_dir / mask_name

        if not mask_path.exists():
            continue

        pairs.append((img_path, mask_path))

    return pairs


# =========================================================
# FULL DATASET LOADER
# =========================================================

def load_dataset(image_dir, mask_dir, limit=None):
    """Returns: list of (image, mask) tuples"""

    pairs = match_image_mask(image_dir, mask_dir)

    dataset = []

    for i, (img_path, mask_path) in enumerate(pairs):

        if limit and i >= limit:
            break

        image = load_image(img_path)
        mask = load_mask(mask_path)

        dataset.append((image, mask))

    return dataset


# =========================================================
# TRAIN/TEST SPLIT
# =========================================================

def train_test_split(dataset, train_ratio=0.8):
    """Splits dataset into train and test sets."""
    
    random.shuffle(dataset)

    split_idx = int(len(dataset) * train_ratio)

    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]

    return train_data, test_data


# =========================================================
# DEBUG HELPERS
# =========================================================

def dataset_summary(dataset):
    """Prints dataset size and sanity check."""

    print("\n================ DATASET SUMMARY ================")
    print(f"Total Samples: {len(dataset)}")

    if len(dataset) > 0:
        img, mask = dataset[0]
        print(f"Image Shape: {img.shape}")
        print(f"Mask Shape: {mask.shape}")

    print("=================================================\n")
