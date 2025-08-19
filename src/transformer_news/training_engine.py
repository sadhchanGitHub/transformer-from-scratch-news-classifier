"""
Main training script for the Transformer-based news classifier.

This script orchestrates the entire model training pipeline, including:
1.  Data Loading and Preprocessing: Fetches the AG_NEWS dataset, tokenizes text,
    builds a vocabulary, and creates PyTorch DataLoaders.
2.  Model Definition: Initializes the TransformerClassifier model, optimizer,
    and loss function using parameters from the `config` module.
3.  Sanity Checks: Inspects a sample batch to ensure data integrity before
    training begins.
4.  Training Loop: Executes the main training and validation loop, computes
    loss, performs backpropagation, and updates model weights.
5.  Evaluation and Early Stopping: Evaluates the model on a test set after each
    epoch and implements early stopping to prevent overfitting.
6.  Model Checkpointing: Saves the best-performing model based on test accuracy.

This script is designed to be run from the command line, with arguments to
control whether to use a full dataset or a smaller sample for quick tests.

Usage:
    # For a quick run on a small sample of the data:
    python -m src.train --epochs 5 --sample-size 1000

    # For a full training run on the entire dataset:
    python -m src.train --full-dataset --epochs 20
"""

# --- IMPORTS ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.nn.utils.rnn import pad_sequence
import argparse
from itertools import islice
from typing import List, Tuple

# Relative imports from the project structure
from .model import TransformerClassifier
from .utils import evaluate
from . import config

import logging

# --- LOGGING SETUP ---
# Configure logging to write to both a file and the console.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ],
)

logging.info("Starting new training run...")
logging.info(f"Using device: {config.DEVICE}")
logging.info(f"Log file located at: {config.LOG_FILE}")


# -----------------------------
# Part 1: Data Preparation
# -----------------------------

def get_source_data(
    tokenizer: callable,
    full_dataset: bool,
    sample_size: int,
    cache_dir: str = config.CACHE_DIR
) -> Tuple[list, list, Vocab]:
    """Loads the AG_NEWS dataset, builds a vocabulary, and returns data samples.

    This function handles the initial data fetching. It downloads the dataset
    (if not already cached), allows for using either the full dataset or a
    smaller subset for faster iteration, and constructs a vocabulary based on the
    training data.

    Args:
        tokenizer (callable): The tokenizer function to apply to the raw text.
        full_dataset (bool): If True, use the entire dataset. Otherwise, use a
            sample of `sample_size`.
        sample_size (int): The number of samples to use if `full_dataset` is False.
        cache_dir (str): The directory to cache the downloaded dataset.

    Returns:
        Tuple[list, list, Vocab]: A tuple containing:
            - The training data sample (list of tuples).
            - The test data sample (list of tuples).
            - The constructed vocabulary (torchtext.vocab.Vocab).
    """
    def yield_tokens(data_iter: iter) -> iter:
        """Helper generator to yield tokens from a dataset iterator."""
        for _, text in data_iter:
            yield tokenizer(text)

    train_iter, test_iter = AG_NEWS(root=cache_dir, split=("train", "test"))

    if full_dataset:
        logging.info("Loading the FULL AG_NEWS dataset...")
        train_sample = list(train_iter)
        test_sample = list(test_iter)
    else:
        logging.info(f"Loading a SAMPLE of {sample_size} records...")
        train_sample = list(islice(train_iter, sample_size))
        test_sample = list(islice(test_iter, sample_size))

    # Build vocabulary from the training sample and add special tokens
    vocab = build_vocab_from_iterator(yield_tokens(train_sample), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])

    return train_sample, test_sample, vocab


def create_dataloaders(
    full_dataset: bool,
    sample_size: int,
    batch_size: int
) -> Tuple[DataLoader, DataLoader, Vocab]:
    """Creates training and testing DataLoaders from the source data.

    This function orchestrates the data preparation pipeline. It calls
    `get_source_data` to get the raw data and vocabulary, saves the vocabulary
    for later use (e.g., in inference), and then constructs and returns the
    final DataLoader objects ready for training.

    Args:
        full_dataset (bool): Flag to determine if the full dataset should be used.
        sample_size (int): The size of the sample to use if not using the full dataset.
        batch_size (int): The number of samples per batch.

    Returns:
        Tuple[DataLoader, DataLoader, Vocab]: A tuple containing:
            - The training DataLoader.
            - The testing DataLoader.
            - The vocabulary object.
    """
    tokenizer = get_tokenizer("basic_english")
    train_sample, test_sample, vocab = get_source_data(tokenizer, full_dataset, sample_size)

    # Save the vocabulary for consistency during inference or future runs
    torch.save(vocab, config.VOCAB_SAVE_PATH)
    logging.info(f"✅ Vocabulary saved to {config.VOCAB_SAVE_PATH}")

    pad_idx = vocab["<pad>"]

    def collate_batch(batch: List[Tuple[int, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collates a batch of text and labels into padded tensors.

        This function is passed to the DataLoader. It processes a list of
        (label, text) pairs by:
        1. Tokenizing and converting text to numerical IDs using the vocabulary.
        2. Adjusting labels to be zero-indexed (as AG_NEWS labels are 1-based).
        3. Padding sequences to the length of the longest sequence in the batch.
        4. Returning tensors ready for model input.

        Args:
            batch (List[Tuple[int, str]]): A list of data samples from the dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of padded text tensors and
            their corresponding label tensors.
        """
        labels_list, text_list = [], []
        for (_label, _text) in batch:
            labels_list.append(_label - 1) # AG_NEWS labels are 1-4, so adjust to 0-3
            processed_text = torch.tensor([vocab[token] for token in tokenizer(_text)], dtype=torch.long)
            text_list.append(processed_text)

        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        padded_text = pad_sequence(text_list, batch_first=True, padding_value=pad_idx)
        return padded_text, labels_tensor

    train_dataloader = DataLoader(train_sample, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_sample, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    logging.info("✅ Train and Test DataLoaders created successfully.")
    return train_dataloader, test_dataloader, vocab


# -----------------------------
# Part 2: Model Definition
# -----------------------------

def define_model(vocab: Vocab) -> Tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
    """Initializes the model, optimizer, and loss function.

    Based on the vocabulary size and hyperparameters from the config file,
    this function instantiates the TransformerClassifier, the Adam optimizer,
    and the CrossEntropyLoss criterion.

    Args:
        vocab (Vocab): The vocabulary object, used to determine the embedding layer size.

    Returns:
        Tuple[nn.Module, torch.optim.Optimizer, nn.Module]: A tuple containing:
            - The initialized Transformer model.
            - The Adam optimizer.
            - The CrossEntropyLoss function.
    """
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
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {num_params:,} trainable parameters.")
    return model, optimizer, loss_fn


# -----------------------------
# Part 3: Sanity Checks
# -----------------------------

def inspect_dataloader(dataloader_to_inspect: DataLoader):
    """Logs the shape and label range of the first batch from a DataLoader.

    This is a crucial sanity check to run before training. It verifies that the
    data processing and batching work as expected and that the labels are within
    the expected range for the classification task.

    Args:
        dataloader_to_inspect (DataLoader): The DataLoader to inspect.
    """
    logging.info("--- Inspecting the first batch from the DataLoader ---")
    try:
        first_batch_tokens, first_batch_labels = next(iter(dataloader_to_inspect))

        logging.info(f"Batch token shape: {first_batch_tokens.shape}")
        logging.info(f"Batch label shape: {first_batch_labels.shape}")

        min_label, max_label = first_batch_labels.min().item(), first_batch_labels.max().item()
        logging.info(f"Labels in batch range from {min_label} to {max_label}.")

        if max_label >= config.NUM_CLASSES or min_label < 0:
            logging.error(f"❌ LABEL RANGE ERROR: Expected labels between 0 and {config.NUM_CLASSES-1}.")
        else:
            logging.info("✅ Label range is valid.")
    except Exception as e:
        logging.error(f"❌ Error during DataLoader inspection: {e}", exc_info=True)


# -----------------------------
# Part 4: Training Loop
# -----------------------------

def execute_training(
    num_epochs: int,
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module
):
    """Runs the main training and evaluation loop.

    This function iterates for a specified number of epochs. In each epoch, it
    trains the model on the training data and evaluates its performance on the
    test data. It implements an early stopping mechanism to halt training if the
    test accuracy does not improve for a set number of epochs (`PATIENCE`). The
    best model state is saved to disk.

    Args:
        num_epochs (int): The maximum number of epochs to train for.
        model (nn.Module): The model to be trained.
        train_dataloader (DataLoader): DataLoader for the training data.
        test_dataloader (DataLoader): DataLoader for the test/validation data.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        loss_fn (nn.Module): The loss function.
    """
    best_test_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0

        for batch_tokens, batch_labels in train_dataloader:
            batch_tokens = batch_tokens.to(config.DEVICE)
            batch_labels = batch_labels.to(config.DEVICE)

            # Standard training steps: forward pass, loss, backward pass, optimizer step
            logits = model(batch_tokens)
            loss = loss_fn(logits, batch_labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_dataloader)
        test_accuracy = evaluate(model, test_dataloader, config.DEVICE)
        logging.info(f"Epoch {epoch+1:02d}/{num_epochs} | Train Loss: {avg_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

        # Check for improvement and apply early stopping logic
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            logging.info(f"  -> New best model saved with accuracy: {best_test_accuracy:.4f} to {config.MODEL_SAVE_PATH}")
        else:
            epochs_without_improvement += 1
            logging.info(f"  -> No improvement. Patience: {epochs_without_improvement}/{config.PATIENCE}")

        if epochs_without_improvement >= config.PATIENCE:
            logging.info(f"⏹ Early stopping triggered after {config.PATIENCE} epochs without improvement.")
            break
    logging.info(f"🏁 Training finished. Best test accuracy: {best_test_accuracy:.4f}")


# -----------------------------
# Part 5: Main Entry Point
# -----------------------------

def main(full_dataset: bool, num_epochs: int, sample_size: int):
    """Orchestrates the entire model training and evaluation process.

    This main function serves as the primary entry point. It calls the necessary
    helper functions in sequence to prepare data, define the model, and execute
    the training loop.

    Args:
        full_dataset (bool): Whether to use the full dataset for training.
        num_epochs (int): The maximum number of epochs for training.
        sample_size (int): The number of data points to use if not using the full dataset.
    """
    logging.info("--- Step 1: Creating DataLoaders ---")
    train_dl, test_dl, vocab = create_dataloaders(
        full_dataset=full_dataset,
        sample_size=sample_size,
        batch_size=config.BATCH_SIZE
    )

    logging.info("--- Step 2: Inspecting DataLoader ---")
    inspect_dataloader(train_dl)

    logging.info("--- Step 3: Defining Model ---")
    model, optimizer, loss_fn = define_model(vocab)

    logging.info("--- Step 4: Starting Training ---")
    execute_training(
        num_epochs=num_epochs,
        model=model,
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        optimizer=optimizer,
        loss_fn=loss_fn
    )
    logging.info("✅ Training script finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model for news classification.")

    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Use the full AG_NEWS dataset. If not set, a small sample is used."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The maximum number of training epochs."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="The number of samples to use for training and testing if --full-dataset is not set."
    )

    args = parser.parse_args()

    main(
        full_dataset=args.full_dataset,
        num_epochs=args.epochs,
        sample_size=args.sample_size
    )