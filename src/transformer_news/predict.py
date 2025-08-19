# src/transformer_news/predict.py

# Core PyTorch
import torch
import torch.nn as nn
import logging

# TorchText for NLP
from torchtext.data.utils import get_tokenizer

# Custom modules
from .model import TransformerClassifier
from . import config

# --- ALL TOP-LEVEL CODE AND UNUSED IMPORTS REMOVED ---

# --- Part 1: Define Artifacts (Stateless Constants) ---
CATEGORY_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


# --- Part 2: Helper Functions (Tools in our Toolbox) ---
def get_vocab():
    vocab = torch.load(config.VOCAB_SAVE_PATH)
    VOCAB_SIZE = len(vocab)
    logging.info(f"✅ Vocab loaded from {config.VOCAB_SAVE_PATH} | Size: {VOCAB_SIZE}")
    PAD_IDX = vocab['<pad>']
    return vocab, VOCAB_SIZE, PAD_IDX

def get_model(vocab_size):
    inference_model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        d_ff=config.D_FF,
        num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)
    inference_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    inference_model.eval()
    logging.info(f"✅ Model weights loaded from {config.MODEL_SAVE_PATH}.")
    return inference_model

def exec_predict(text, model, vocab, tokenizer, device):
    model.eval()
    token_ids = torch.tensor([vocab[token] for token in tokenizer(text)], dtype=torch.long)
    token_ids = token_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(token_ids)
    predicted_index = torch.argmax(logits, dim=1).item()
    return CATEGORY_MAP[predicted_index]


# -----------------------------
# Main entry point (as a callable function)
# -----------------------------
# -----------------------------
# Main entry point (as a callable function)
# -----------------------------
def main(news_headline: str):
    """
    Loads the model and vocabulary, then runs prediction on a single news headline.
    
    Args:
        news_headline (str): The raw text of the news headline to classify.
        
    Returns:
        str: The name of the predicted category.
    """
    if not news_headline:
        logging.error("❌ No news headline provided for prediction.")
        return

    logging.info("▶ Prediction run started with configuration:")
    logging.info(f"   News Article Headline: {news_headline}")
    
    # --- THE FIX IS HERE ---
    # Create the tokenizer here, inside the function, right before you need it.
    tokenizer = get_tokenizer("basic_english")
    
    # Now the rest of the function will work perfectly.
    vocab, vocab_size, pad_idx = get_vocab()
    inference_model = get_model(vocab_size=vocab_size)
    
    result_prediction = exec_predict(news_headline, inference_model, vocab, tokenizer, config.DEVICE)
    
    logging.info(f"Article: '{news_headline}'")
    logging.info(f"Predicted Category: {result_prediction}\n")
    
    return result_prediction
    