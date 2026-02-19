"""
Central configuration for the Music Genre Classifier.
"""
import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
GTZAN_DIR = os.path.join(DATA_DIR, "genres_original")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# ──────────────────────────────────────────────
# Genre labels (GTZAN)
# ──────────────────────────────────────────────
GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]
NUM_CLASSES = len(GENRES)

# ──────────────────────────────────────────────
# Audio / Mel-spectrogram settings
# ──────────────────────────────────────────────
SAMPLE_RATE = 22050
DURATION = 30            # seconds per GTZAN clip
SEGMENT_DURATION = 3     # seconds per segment window
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# Derived
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_DURATION  # samples per segment

# ──────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
VAL_SPLIT = 0.2          # fraction held out for validation
RANDOM_SEED = 42
DROPOUT = 0.3
