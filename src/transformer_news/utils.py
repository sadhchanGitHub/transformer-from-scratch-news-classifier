"""
Utility functions for the Transformer model training and evaluation pipeline.

This module provides helper functions that are used across different stages of
the machine learning workflow, such as model evaluation. Consolidating these
routines here keeps the main training script cleaner and more focused on the
core logic.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# It's good practice to add type hints to function signatures.
# This improves code readability, and allows for static analysis.
def evaluate(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    """Evaluates the performance of a classification model on a given dataset.

    This function iterates through the provided `DataLoader`, performs a forward
    pass to get model predictions, and computes the overall accuracy by comparing
    these predictions to the ground truth labels.

    The function automatically handles:
    - Setting the model to evaluation mode (`model.eval()`).
    - Disabling gradient calculations (`torch.no_grad()`) for efficiency.
    - Moving data tensors to the specified computation device.

    Args:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        dataloader (torch.utils.data.DataLoader): A DataLoader providing batches
            of tokenized inputs and corresponding integer labels.
        device (str): The device identifier ('cuda' or 'cpu') on which to
            perform the computation.

    Returns:
        float: The overall accuracy of the model on the provided dataset,
               represented as a float between 0.0 and 1.0.

    Example:
        >>> # Assuming `my_model`, `val_loader`, and `device` are defined
        >>> validation_accuracy = evaluate(my_model, val_loader, device)
        >>> print(f"Validation Accuracy: {validation_accuracy:.4f}")
    """
    # Set the model to evaluation mode. This is crucial as it disables layers
    # like Dropout and uses the learned statistics for BatchNorm, ensuring
    # deterministic and representative outputs.
    model.eval()

    total_correct = 0
    total_samples = 0

    # The `torch.no_grad()` context manager is a critical optimization. It
    # tells PyTorch not to compute or store gradients, which significantly
    # reduces memory consumption and speeds up the forward pass.
    with torch.no_grad():
        for batch_tokens, batch_labels in dataloader:
            # Move data to the same device as the model to prevent runtime errors.
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)

            # 1. Forward pass: Get the raw, unnormalized scores (logits) from the model.
            logits = model(batch_tokens)

            # 2. Get predictions: The predicted class is the index of the highest
            #    logit score for each sample in the batch.
            predictions = torch.argmax(logits, dim=1)

            # 3. Tally correct predictions by comparing with true labels.
            #    `.sum()` counts the number of `True` values, and `.item()`
            #    extracts the scalar value from the resulting tensor.
            total_correct += (predictions == batch_labels).sum().item()

            # 4. Keep track of the total number of samples processed.
            total_samples += batch_labels.size(0)

    # Calculate the final accuracy.
    accuracy = total_correct / total_samples

    return accuracy