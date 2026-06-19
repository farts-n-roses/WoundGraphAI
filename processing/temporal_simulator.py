# =========================================================
# TEMPORAL WOUND HEALING SIMULATOR
# =========================================================

# This is the CORE computational biology simulation layer of the project.
# This module creates synthetic wound healing progression from a single wound mask.

# Example:
# Day 0  -> severe wound
# Day 3  -> slightly healed
# Day 7  -> moderate healing
# Day 14 -> mostly healed

# =========================================================
# IMPORTS
# =========================================================

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# LOAD MASK
# =========================================================

# Load grayscale wound mask.
def load_mask(mask_path):

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    return mask

# =========================================================
# BINARIZE MASK
# =========================================================

# Convert mask into binary format.
def binarize_mask(mask):

    _, binary = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return binary

# =========================================================
# HEALING STEP SIMULATION
# =========================================================

# Simulate one healing step using erosion.
# Biological interpretation: wound contracts over time.
def simulate_healing_step(mask, kernel_size=3):

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    healed = cv2.erode(
        mask,
        kernel,
        iterations=1
    )

    return healed

# =========================================================
# EDGE SMOOTHING
# =========================================================

# Smooth wound boundaries to mimic tissue recovery.
def smooth_tissue(mask):

    smoothed = cv2.GaussianBlur(
        mask,
        (5, 5),
        0
    )

    _, smoothed = cv2.threshold(
        smoothed,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return smoothed

# =========================================================
# AREA CALCULATION
# =========================================================

# Calculate wound area percentage.
def compute_wound_area(mask):

    wound_pixels = np.sum(mask > 0)

    total_pixels = mask.shape[0] * mask.shape[1]

    area_percent = (
        wound_pixels / total_pixels
    ) * 100

    return wound_pixels, area_percent

# =========================================================
# TEMPORAL SEQUENCE GENERATION
# =========================================================

def generate_healing_sequence(
    initial_mask,
    healing_rate=0.12
):
    # Generate biologically-inspired wound healing using exponential decay dynamics.
    sequence = []
    current_mask = initial_mask.copy()

    # Initial wound area
    initial_area = np.sum(current_mask > 0)

    # Healing timeline
    days = [0, 3, 5, 7, 10, 14]

    for day in days:
        # TARGET AREA USING EXPONENTIAL DECAY
        # A(t) = A0 * exp(-k*t)

        target_area = initial_area * np.exp(
            -healing_rate * day
        )

        # ITERATIVELY HEAL UNTIL TARGET AREA REACHED

        temp_mask = current_mask.copy()

        while np.sum(temp_mask > 0) > target_area:

            temp_mask = simulate_healing_step(
                temp_mask,
                kernel_size=3
            )

            temp_mask = smooth_tissue(temp_mask)

            # Safety stop
            if np.sum(temp_mask > 0) == 0:
                break

        current_mask = temp_mask.copy()

        sequence.append(current_mask.copy())

        # Debug info
        current_area = np.sum(current_mask > 0)

        healing_percent = (
            current_area / initial_area
        ) * 100

        print(
            f"Day {day} → "
            f"{healing_percent:.2f}% remaining"
        )

    return sequence

# =========================================================
# VISUALIZE HEALING PROGRESSION
# =========================================================

# Display healing progression timeline.
def visualize_sequence(sequence):

    num_images = len(sequence)

    _, ax = plt.subplots(
        1,
        num_images,
        figsize=(3 * num_images, 4)
    )

    if num_images == 1:
        ax = [ax]

    for i, mask in enumerate(sequence):

        _, area_percent = compute_wound_area(mask)

        ax[i].imshow(mask, cmap="gray")

        days = [0, 3, 5, 7, 10, 14]
        
        ax[i].set_title(
            f"Day {days[i]}\n{area_percent:.2f}%"
        )

        ax[i].axis("off")

    plt.tight_layout()
    plt.show()

# =========================================================
# HEALING CURVE VISUALIZATION
# =========================================================

# Plot wound reduction over time.
def plot_healing_curve(sequence):

    healing_percentages = []

    for mask in sequence:

        _, area_percent = compute_wound_area(mask)

        healing_percentages.append(area_percent)

    days = [0, 3, 5, 7, 10, 14]

    plt.figure(figsize=(7, 5))

    plt.plot(
        days,
        healing_percentages,
        marker="o"
    )

    plt.xlabel("Healing Day")

    plt.ylabel("Wound Area (%)")

    plt.title("Temporal Wound Healing Progression")

    plt.grid(True)

    plt.show()

# =========================================================
# SAVE GENERATED SEQUENCES
# =========================================================

def save_sequence(sequence, output_dir):
    # Save simulated healing masks.

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for i, mask in enumerate(sequence):

        save_path = output_dir / f"day_{i}.png"

        cv2.imwrite(
            str(save_path),
            mask
        )

# =========================================================
# FULL SIMULATION PIPELINE
# =========================================================

def run_temporal_simulation(mask_path):

    # Load mask
    mask = load_mask(mask_path)

    # Binarize
    mask = binarize_mask(mask)

    # Generate sequence
    sequence = generate_healing_sequence(
        mask,
        healing_rate=0.12
        # 0.15 = healthy healing
        # 0.05 = diabetic/slow healing
        # 0.02 = infected/very slow healing
    )

    # Visualize
    visualize_sequence(sequence)

    # Healing curve
    plot_healing_curve(sequence)

    # Save sequence
    save_sequence(
        sequence,
        "dataset/simulated_sequences/sample_sequence"
    )

    print("\n================ TEMPORAL SIMULATION ================\n")

    print("Healing sequence generated successfully.")

    print(
        "Saved to: "
        "dataset/simulated_sequences/sample_sequence"
    )

    print("\n=====================================================\n")

# =========================================================
# COMPUTE HEALING PERCENTAGES
# =========================================================

def compute_healing_percentages(masks):

    percentages = []

    # Initial wound area for percentage calculation
    initial_area = (masks[0].sum())

    # Compute Remaining Area %
    for mask in masks:

        current_area = mask.sum()

        remaining = (current_area / initial_area) * 100

        percentages.append(round(remaining, 2))

    return percentages

# =========================================================
# TEST RUN
# =========================================================

if __name__ == "__main__":

    sample_mask = ("dataset/masks/train_masks/fusc_0002.png")

    run_temporal_simulation(sample_mask)
