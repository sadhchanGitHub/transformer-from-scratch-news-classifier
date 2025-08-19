# --- CONFIGURATION FILE FOR TRANSFORMER PROJECT ---
import torch
import os
from datetime import datetime

# Base project directory (two levels up from this file)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Data directory (for raw, cache, etc.)
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "ag_news_cache")

# Logs directory
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Models / checkpoints
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure dirs exist
for d in [DATA_DIR, CACHE_DIR, LOGS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# File Paths
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "transformer_news_classifier_best.pth")
VOCAB_SAVE_PATH = os.path.join(MODELS_DIR, "newsclassification_vocab.pth")

# Timestamp for unique log files
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOGS_DIR, f"train_{TIMESTAMP}.log")

# System Parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Model Architecture Hyperparameters
BATCH_SIZE = 64
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 4 * D_MODEL
NUM_CLASSES = 4
MAX_SEQ_LEN = 512  # Max sequence length for positional encoding

# Training Hyperparameters
LEARNING_RATE = 1e-4

# Early Stopping Parameters
PATIENCE = 3  # Number of epochs to wait for improvement
