# =========================================================
# HEALING PREDICTOR
# =========================================================

# Converts raw Temporal GNN outputs into:
# - healing percentage
# - recovery interpretation
# - recovery risk
# - estimated healing timeline
# - biomedical summary

# =========================================================
# IMPORTS
# =========================================================

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch

from models.temporal_gnn import (
    TemporalGNN,
    create_graph_sequence
)

from config import DEVICE

# =========================================================
# HEALING INTERPRETATION
# =========================================================

def classify_healing(score):

    if score >= 0.70:

        return (
            "Healthy Healing",
            "Low Risk"
        )

    elif score >= 0.50:

        return (
            "Moderate Recovery",
            "Moderate Risk"
        )

    else:

        return (
            "Delayed Healing Risk",
            "High Risk"
        )

# =========================================================
# ESTIMATED RECOVERY DAYS
# =========================================================

def estimate_recovery_days(score):

    # Higher score = faster recovery

    if score >= 0.90:
        return 7

    elif score >= 0.70:
        return 10

    elif score >= 0.60:
        return 14

    elif score >= 0.50:
        return 21

    else:
        return 30

# =========================================================
# BIOMEDICAL INSIGHT GENERATION
# =========================================================

def generate_clinical_insight(
    healing_label,
    risk_level,
    score
):

    if healing_label == "Healthy Healing":

        return f"""
Patient exhibits strong epithelial wound
contraction with favorable tissue recovery
dynamics.

Predicted healing trajectory indicates
consistent tissue remodeling and low
risk of delayed recovery.

Overall healing confidence:
{score:.2f}
"""

    elif healing_label == "Moderate Recovery":

        return f"""
Patient demonstrates moderate wound
closure progression with partial tissue
recovery.

Healing trajectory suggests stable
recovery behavior but requires continued
monitoring for complete epithelial closure.

Overall healing confidence:
{score:.2f}
"""

    else:

        return f"""
Patient exhibits slow wound contraction
with elevated risk of delayed healing.

Spatial tissue recovery dynamics suggest
possible abnormal healing progression
or impaired epithelial regeneration.

Overall healing confidence:
{score:.2f}
"""

# =========================================================
# MAIN PREDICTION PIPELINE
# =========================================================

def run_healing_prediction():

    SEPARATOR = "======================================"
    print(f"\n{SEPARATOR}")

    print("INITIALIZING TEMPORAL HEALING MODEL")

    print(f"{SEPARATOR}\n")

    # =====================================================
    # CREATE GRAPH SEQUENCE
    # =====================================================

    graph_sequence = create_graph_sequence()

    # =====================================================
    # LOAD MODEL
    # =====================================================

    model = TemporalGNN().to(DEVICE)

    model.eval()

    # =====================================================
    # INFERENCE
    # =====================================================

    with torch.no_grad():
        prediction = model(graph_sequence)

    healing_score = prediction.item()
    healing_score = max(
        0.15,
        min(healing_score, 0.95)
    )

    # =====================================================
    # INTERPRETATION
    # =====================================================

    healing_label, risk_level = (classify_healing(healing_score))

    recovery_days = (estimate_recovery_days(healing_score))

    clinical_summary = (
        generate_clinical_insight(
            healing_label,
            risk_level,
            healing_score
        )
    )

    # =====================================================
    # OUTPUT RESULTS
    # =====================================================

    print(f"\n{SEPARATOR}")

    print("HEALING PREDICTION RESULTS")

    print(f"{SEPARATOR}\n")

    print(
        f"Healing Score: "
        f"{healing_score:.4f}"
    )

    print(
        f"Healing Classification: "
        f"{healing_label}"
    )

    print(
        f"Risk Level: "
        f"{risk_level}"
    )

    print(
        f"Estimated Recovery Time: "
        f"{recovery_days} days"
    )

    print(f"\n{SEPARATOR}")

    print("CLINICAL INSIGHT SUMMARY")

    print(f"{SEPARATOR}\n")

    print(clinical_summary)

# =========================================================
# COMPLETE HEALING ANALYSIS PIPELINE
# =========================================================

def generate_prediction_results():

    # =====================================================
    # CREATE GRAPH SEQUENCE
    # =====================================================

    graph_sequence = create_graph_sequence()

    # =====================================================
    # LOAD MODEL
    # =====================================================

    model = TemporalGNN().to(DEVICE)

    model.eval()

    # =====================================================
    # INFERENCE
    # =====================================================

    with torch.no_grad():

        prediction = model(
            graph_sequence
        )

    healing_score = prediction.item()

    # =====================================================
    # CLAMP OUTPUT
    # =====================================================

    healing_score = max(
        0.15,
        min(healing_score, 0.95)
    )

    # =====================================================
    # INTERPRETATION
    # =====================================================

    healing_label, risk_level = (
        classify_healing(
            healing_score
        )
    )

    recovery_days = (
        estimate_recovery_days(
            healing_score
        )
    )

    clinical_summary = (
        generate_clinical_insight(
            healing_label,
            risk_level,
            healing_score
        )
    )

    # =====================================================
    # RETURN RESULTS
    # =====================================================

    return {

        "healing_score": healing_score,

        "healing_label": healing_label,

        "risk_level": risk_level,

        "recovery_days": recovery_days,

        "clinical_summary": clinical_summary
    }

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    run_healing_prediction()
