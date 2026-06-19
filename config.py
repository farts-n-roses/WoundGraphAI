# ============================================
# CONFIGURATION FILE
# ============================================

from pathlib import Path
import torch

# =========================================================
# PROJECT ROOT
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# =========================================================
# DATASET PATHS
# =========================================================

DATASET_DIR = BASE_DIR / "dataset"

RAW_IMAGES_DIR = DATASET_DIR / "images"

MASKS_DIR = DATASET_DIR / "masks"

SEQUENCES_DIR = DATASET_DIR / "simulated_sequences"

METADATA_FILE = DATASET_DIR / "metadata.csv"

PRECOMPUTED_DIR = DATASET_DIR / "precomputed"

# =========================================================
# OUTPUT / SAVED FILES
# =========================================================

OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR = BASE_DIR / "models"

SAVED_MODELS_DIR = MODEL_DIR / "saved_models"

CHECKPOINT_DIR = SAVED_MODELS_DIR / "checkpoints"

BEST_SEGMENTATION_MODEL = SAVED_MODELS_DIR / "best_segmentation_model.pt"

BEST_TEMPORAL_MODEL = SAVED_MODELS_DIR / "best_temporal_gnn.pt"

# Create folders automatically if missing
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# IMAGE SETTINGS
# =========================================================

IMAGE_SIZE = 256

PATCH_SIZE = 32

CHANNELS = 1

# =========================================================
# GRAPH SETTINGS
# =========================================================

NODE_FEATURES = 4

# mean intensity
# std intensity
# x coordinate
# y coordinate

HIDDEN_CHANNELS = 64

# =========================================================
# TEMPORAL SETTINGS
# =========================================================

SEQUENCE_LENGTH = 7

# Example:
# Day0 → Day3 → Day5 → Day7 → Day10 ...

# =========================================================
# TRAINING SETTINGS
# =========================================================

BATCH_SIZE = 4

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 5e-4

EPOCHS = 50

TRAIN_SPLIT = 0.8

RANDOM_SEED = 42

# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================================
# THRESHOLDS
# =========================================================

HEALING_THRESHOLD = 0.90

# Example:
# if healing >= 90%
# classify as healed

# =========================================================
# VISUALIZATION SETTINGS
# =========================================================

PLOT_DPI = 120

ANIMATION_INTERVAL = 400

# =========================================================
# STREAMLIT SETTINGS
# =========================================================

APP_TITLE = (
    "Spatio-Temporal Graph Neural Network "
    "Framework for Computational Modeling "
    "and Predictive Analysis of "
    "Epithelial Wound Healing Dynamics"
)

PAGE_ICON = "🧬"

LAYOUT = "wide"
