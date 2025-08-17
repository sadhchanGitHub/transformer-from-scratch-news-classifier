# --- CONFIGURATION FILE FOR TRANSFORMER PROJECT ---
import torch

# System Parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data Parameters
FULL_DATASET = True
SAMPLE_SIZE = 1000 # Only used if FULL_DATASET is False
BATCH_SIZE = 64

# Model Architecture Hyperparameters
# Note: VOCAB_SIZE is data-dependent and will be set in the notebook
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 4 * D_MODEL
NUM_CLASSES = 4
MAX_SEQ_LEN = 512 # Max sequence length for positional encoding

# Training Hyperparameters
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

# --- NEW: Early Stopping Parameters ---
PATIENCE = 3 # Number of epochs to wait for improvement

# File Paths
MODEL_SAVE_PATH = "../models/transformer_news_classifier_best.pth"
VOCAB_SAVE_PATH = "../models/newsclassification_vocab.pth"
