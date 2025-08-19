"""
Prediction script for the Transformer-based news classifier.

This module provides the functionality to load a pre-trained Transformer model
and its associated vocabulary to perform inference on new, unseen news headlines.
It is designed to be the primary entry point for using the trained model in an
application.

The script encapsulates the entire prediction pipeline:
1.  Loading artifacts: Fetches the saved vocabulary and model weights.
2.  Preprocessing: Tokenizes and numericalizes the input text.
3.  Inference: Runs the model to get a prediction.
4.  Post-processing: Maps the model's output index to a human-readable category.

Usage (as a library):
    from src.transformer_news.predict import main as predict_category
    headline = "New rocket launched by SpaceX to explore Mars."
    predicted_class = predict_category(headline)
    print(predicted_class)
    # Expected output: 'Sci/Tech'
"""

# Core PyTorch
import torch
import torch.nn as nn
import logging
from typing import Tuple

# TorchText for NLP
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

# Custom modules
from .model import TransformerClassifier
from . import config


# --- CONSTANTS ---

# Maps the integer output of the model to a human-readable category name.
# This must be consistent with the label processing in the training script.
CATEGORY_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


# --- HELPER FUNCTIONS ---

def get_vocab() -> Tuple[Vocab, int, int]:
    """Loads the saved torchtext vocabulary object from disk.

    This function is responsible for retrieving the vocabulary that was created
    and saved during the training process. Using the same vocabulary ensures
    that tokens in new text are mapped to the same integer indices as they
    were during training.

    Returns:
        Tuple[Vocab, int, int]: A tuple containing:
            - The loaded `torchtext.vocab.Vocab` object.
            - The total size of the vocabulary (int).
            - The integer index for the padding token '<pad>' (int).
    """
    vocab = torch.load(config.VOCAB_SAVE_PATH)
    vocab_size = len(vocab)
    pad_idx = vocab['<pad>']
    logging.info(f"✅ Vocab loaded from {config.VOCAB_SAVE_PATH} | Size: {vocab_size}")
    return vocab, vocab_size, pad_idx

def get_model(vocab_size: int) -> TransformerClassifier:
    """Initializes a Transformer model and loads pre-trained weights.

    This function first instantiates the `TransformerClassifier` with the same
    architecture used during training. It then loads the saved state dictionary
    from the best-performing checkpoint and sets the model to evaluation mode.

    Args:
        vocab_size (int): The size of the vocabulary, required to correctly
                          initialize the model's embedding layer.

    Returns:
        TransformerClassifier: The pre-trained model, ready for inference.
    """
    inference_model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        d_ff=config.D_FF,
        num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)

    # Load the saved weights from the best model checkpoint
    inference_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))

    # Set the model to evaluation mode. This disables layers like Dropout.
    inference_model.eval()

    logging.info(f"✅ Model weights loaded from {config.MODEL_SAVE_PATH}.")
    return inference_model

def exec_predict(
    text: str,
    model: nn.Module,
    vocab: Vocab,
    tokenizer: callable,
    device: str
) -> str:
    """Performs inference on a single piece of text.

    This function encapsulates the core prediction logic. It takes raw text,
    pre-processes it into a tensor of token IDs, feeds it to the model, and
    post-processes the output to return the predicted category name.

    Args:
        text (str): The input text to classify.
        model (nn.Module): The pre-trained PyTorch model.
        vocab (Vocab): The vocabulary for converting tokens to indices.
        tokenizer (callable): The tokenizer function.
        device (str): The device ('cuda' or 'cpu') to run inference on.

    Returns:
        str: The predicted category name (e.g., "Sports").
    """
    model.eval()
    # 1. Pre-process: Tokenize and convert to numerical indices
    token_ids = torch.tensor([vocab[token] for token in tokenizer(text)], dtype=torch.long)

    # 2. Add batch dimension and move to the correct device
    token_ids = token_ids.unsqueeze(0).to(device)

    # 3. Inference: Run the forward pass with no gradient calculation
    with torch.no_grad():
        logits = model(token_ids)

    # 4. Post-process: Get the index of the highest logit and map to category
    predicted_index = torch.argmax(logits, dim=1).item()
    return CATEGORY_MAP[predicted_index]


# -----------------------------
# Main Entry Point
# -----------------------------

def main(news_headline: str) -> str:
    """Orchestrates the prediction pipeline for a single news headline.

    This is the main user-facing function. It loads all necessary artifacts
    (model and vocabulary), processes the input headline, and returns the
    predicted category as a string.

    Args:
        news_headline (str): The raw text of the news headline to classify.

    Returns:
        str: The name of the predicted category, or an error message string
             if the input is invalid.

    Example:
        >>> headline = "The team won the championship game last night."
        >>> prediction = main(headline)
        >>> print(prediction)
        'Sports'
    """
    if not news_headline or not isinstance(news_headline, str):
        error_msg = "❌ Invalid input: Please provide a non-empty string."
        logging.error(error_msg)
        return error_msg

    logging.info("▶ Starting prediction...")
    logging.info(f"   Input Headline: '{news_headline}'")

    # Load necessary artifacts for prediction
    tokenizer = get_tokenizer("basic_english")
    vocab, vocab_size, _ = get_vocab()
    inference_model = get_model(vocab_size=vocab_size)

    # Execute the prediction
    predicted_category = exec_predict(news_headline, inference_model, vocab, tokenizer, config.DEVICE)

    logging.info(f"✅ Prediction Complete: '{predicted_category}'\n")
    return predicted_category