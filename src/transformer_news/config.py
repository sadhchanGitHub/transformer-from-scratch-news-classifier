"""
Centralized configuration file for the Transformer-based news classification project.

This file consolidates all essential parameters, from file paths and hardware settings
to model architecture and training hyperparameters. Modifying these values allows for
easy experimentation and adaptation of the project without changing the core source code.

Sections:
    - Directory and File Paths: Manages all input/output locations.
    - System Parameters: Defines computational resources (e.g., CPU/GPU).
    - Model Architecture Hyperparameters: Specifies the Transformer model's structure.
    - Training Hyperparameters: Controls the training process.
    - Early Stopping Parameters: Configures the early stopping mechanism.
"""

import torch
import os
from datetime import datetime

# --- DIRECTORY AND FILE PATHS ---

# Base project directory (assumes this config file is in a subdirectory like 'src/config')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Data directory for raw datasets, processed data, and cache
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "ag_news_cache")

# Directory for storing training logs
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Directory for saving trained model checkpoints and vocabularies
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Automatically create necessary directories if they don't exist
for d in [DATA_DIR, CACHE_DIR, LOGS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Path to save the best performing model checkpoint
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "transformer_news_classifier_best.pth")

# Path to save the vocabulary object for consistent tokenization
VOCAB_SAVE_PATH = os.path.join(MODELS_DIR, "news_classification_vocab.pth")

# Unique timestamp for naming log files to prevent overwrites
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOGS_DIR, f"train_{TIMESTAMP}.log")


# --- SYSTEM PARAMETERS ---

# Automatically select CUDA if available, otherwise fall back to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- MODEL ARCHITECTURE HYPERPARAMETERS ---

BATCH_SIZE = 64        # Number of samples per training batch
D_MODEL = 256          # The dimensionality of the model's embeddings and hidden states
NUM_HEADS = 8          # Number of attention heads in the Multi-Head Attention mechanism
NUM_LAYERS = 4         # Number of stacked Encoder layers in the Transformer
D_FF = 4 * D_MODEL     # The inner dimension of the Position-wise Feed-Forward Networks
NUM_CLASSES = 4        # The number of output classes for the classification task (e.g., AG News has 4 categories)
MAX_SEQ_LEN = 512     # The maximum sequence length for positional encoding; longer sequences will be truncated


# --- TRAINING HYPERPARAMETERS ---

LEARNING_RATE = 1e-4   # The initial learning rate for the Adam optimizer


# --- EARLY STOPPING PARAMETERS ---

# Number of consecutive epochs with no improvement on the validation metric before stopping training
PATIENCE = 3
