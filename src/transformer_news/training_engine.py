# --- IMPORTS ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import portalocker
import os
import argparse
from itertools import islice

# now that we have __init_.py
from .model import TransformerClassifier
from .utils import evaluate
from . import config

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ],
)

logging.info("Starting training run...")
#logging.info(f"Cache dir: {config.CACHE_DIR}")
#logging.info(f"Models dir: {config.MODELS_DIR}")


# Globals (will be set later)
tokenizer = get_tokenizer("basic_english")
vocab = None
PAD_IDX = None
train_dataloader, test_dataloader = None, None
model, optimizer, loss_fn = None, None, None

# -----------------------------
# Part1: Data Preparation
# -----------------------------
# This function is mostly fine, but we'll add the tokenizer as an argument for clarity.
def get_source_data(tokenizer, full_dataset, sample_size, cache_dir=config.CACHE_DIR):
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    # ... (rest of the function is the same, just renamed from getSrcdata)
    # ... but returns the built vocab
    train_iter, test_iter = AG_NEWS(root=cache_dir, split=("train", "test"))
    
    if full_dataset:
        logging.info("Loading FULL dataset...")
        train_sample = list(train_iter)
        test_sample = list(test_iter)
    else:
        logging.info(f"Loading SAMPLE of {sample_size}...")
        train_sample = list(islice(train_iter, sample_size))
        test_sample = list(islice(test_iter, sample_size))

    vocab = build_vocab_from_iterator(yield_tokens(train_sample), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])

    return train_sample, test_sample, vocab

# -----------------------------
# DataLoader creation
# -----------------------------

def create_dataloaders(full_dataset, sample_size, batch_size):
    tokenizer = get_tokenizer("basic_english")
    train_sample, test_sample, vocab = get_source_data(tokenizer, full_dataset, sample_size)
    
    torch.save(vocab, config.VOCAB_SAVE_PATH)
    logging.info(f"✅ Vocab saved to {config.VOCAB_SAVE_PATH}")

    pad_idx = vocab["<pad>"]

    # This collate_fn needs access to vocab, tokenizer, and pad_idx.
    # Defining it inside this function is the cleanest way to give it that access.
    def collate_batch(batch):
        labels_list, text_list = [], []
        for (_label, _text) in batch:
            labels_list.append(_label - 1)
            processed_text = torch.tensor([vocab[token] for token in tokenizer(_text)], dtype=torch.long)
            text_list.append(processed_text)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        padded_text = pad_sequence(text_list, batch_first=True, padding_value=pad_idx)
        return padded_text, labels_tensor

    train_dataloader = DataLoader(train_sample, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_sample, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    logging.info("✅ Train and Test DataLoaders created successfully.")
    
    # Return everything needed by downstream functions
    return train_dataloader, test_dataloader, vocab




# -----------------------------
# Define Model
# -----------------------------
# REFACTORED: Takes vocab as an argument. Returns the created objects.
def define_model(vocab):
    vocab_size = len(vocab)
    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        d_ff=config.D_FF,
        num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    logging.info("✅ Model, optimizer, and loss function defined.")
    return model, optimizer, loss_fn



# -----------------------------
# Inspect Dataloader
# -----------------------------
# REFACTORED: It now accepts the dataloader it needs to inspect
def inspect_dataloader(dataloader_to_inspect):
    logging.info("--- Inspecting the first batch from the DataLoader ---")
    try:
        # Use the argument, not a global variable
        first_batch_tokens, first_batch_labels = next(iter(dataloader_to_inspect))
        
        unique_labels = first_batch_labels.unique()
        min_label = first_batch_labels.min().item()
        max_label = first_batch_labels.max().item()

        logging.info(f"Batch token shape: {first_batch_tokens.shape}")
        logging.info(f"Batch label shape: {first_batch_labels.shape}")
        logging.info(f"Unique labels in this batch: {unique_labels}")
        logging.info(f"Labels range: {min_label}–{max_label}")

        if max_label >= config.NUM_CLASSES or min_label < 0:
            logging.error("❌ ERROR: Labels out of bounds!")
        else:
            logging.info("✅ Labels are correct.")
            
    except Exception as e:
        logging.error(f"❌ Error while inspecting DataLoader: {e}")


# -----------------------------
# Training Loop
# -----------------------------
def execute_training(num_epochs, model, train_dataloader, test_dataloader, optimizer, loss_fn):
    best_test_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_tokens, batch_labels in train_dataloader:
            batch_tokens = batch_tokens.to(config.DEVICE)
            batch_labels = batch_labels.to(config.DEVICE)
            # ... (rest of the training loop is identical)
            logits = model(batch_tokens)
            loss = loss_fn(logits, batch_labels)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_dataloader)
        test_accuracy = evaluate(model, test_dataloader, config.DEVICE)
        logging.info(f"Epoch {epoch+1:02d}/{num_epochs} | Train Loss: {avg_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            logging.info(f"  -> New best model saved with accuracy: {best_test_accuracy:.4f}")
        else:
            epochs_without_improvement += 1
            logging.info(f"  -> No improvement. Patience: {epochs_without_improvement}/{config.PATIENCE}")

        if epochs_without_improvement >= config.PATIENCE:
            logging.info(f"⏹ Early stopping after {config.PATIENCE} epochs without improvement.")
            break


# -----------------------------
# Main entry point (as a callable function)
# -----------------------------
def main(full_dataset: bool, num_epochs: int, sample_size: int):
    """Orchestrates the model training process."""
    
    # 1. Create data loaders and get the vocab
    train_dl, test_dl, vocab = create_dataloaders(
        full_dataset=full_dataset, 
        sample_size=sample_size,
        batch_size=config.BATCH_SIZE
    )
    
    # 2. Inspect the newly created training dataloader
    #    Pass the 'train_dl' variable as an argument
    inspect_dataloader(train_dl)
    
    # 3. Define the model, passing the vocab to it
    model, optimizer, loss_fn = define_model(vocab)
    
    # 4. Execute the training loop, passing all required objects
    execute_training(
        num_epochs=num_epochs, 
        model=model,
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        optimizer=optimizer,
        loss_fn=loss_fn
    )
