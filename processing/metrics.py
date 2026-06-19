# =========================================================
# METRICS MODULE
# =========================================================
# Centralized biological + computational metrics for:
# - wound area
# - healing progression
# - perimeter
# - recovery scoring
# =========================================================

import numpy as np
import cv2

# =========================================================
# BASIC WOUND METRICS
# =========================================================

def compute_wound_area(mask):
    """
    Returns number of wound pixels.
    """
    return np.sum(mask > 0)


def compute_wound_percentage(mask):
    """
    Returns wound coverage percentage (0–100).
    """
    
    mask = mask.astype(np.uint8)

    wound_pixels = np.count_nonzero(mask)
    
    total_pixels = mask.shape[0] * mask.shape[1]

    return (wound_pixels / total_pixels) * 100
    

def compute_background_percentage(mask):
    """
    Returns non-wound area percentage.
    """
    return 100 - compute_wound_percentage(mask)


# =========================================================
# HEALING METRICS (CONSISTENT DEFINITION)
# =========================================================

def compute_healing(initial_wound, final_wound):
    """
    Healing = reduction in wound area.

    Example:
    initial = 100%
    final = 20%
    healing = 80%
    """
    return max(0.0, initial_wound - final_wound)


def compute_healing_rate(initial_wound, final_wound, time_steps=1):
    """
    Average healing per step.
    """
    if time_steps == 0:
        return 0.0

    return (initial_wound - final_wound) / time_steps


def normalize_healing_score(healing_percent):
    """
    Converts healing % into 0–1 score for ML models.
    """
    return np.clip(healing_percent / 100.0, 0.0, 1.0)


# =========================================================
# PERIMETER METRICS
# =========================================================

def compute_perimeter(mask):
    """
    Computes wound boundary length using contours.
    """

    mask = mask.astype(np.uint8)
    mask = mask > 0
    
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return sum(cv2.arcLength(c, True) for c in contours)


# =========================================================
# CONSISTENCY HELPERS
# =========================================================

def get_wound_stats(mask):
    """
    Returns a full dictionary of wound metrics.
    Useful for dashboards.
    """

    wound_pct = compute_wound_percentage(mask)

    return {
        "wound_area_pct": wound_pct,
        "background_pct": 100 - wound_pct,
        "perimeter": compute_perimeter(mask)
    }


# =========================================================
# DEBUG TOOL
# =========================================================

def print_metrics(mask):
    """
    Quick debug summary.
    """

    print("\n================ WOUND METRICS ================")
    print(f"Wound Area (%): {compute_wound_percentage(mask):.2f}")
    print(f"Background (%): {compute_background_percentage(mask):.2f}")
    print(f"Perimeter: {compute_perimeter(mask):.2f}")
    print("================================================\n")
