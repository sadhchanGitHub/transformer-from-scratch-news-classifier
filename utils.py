import torch

def evaluate(model, dataloader, device):
    """
    Calculates the accuracy of the model on a given dataset.

    Args:
        model (nn.Module): The trained Transformer model.
        dataloader (DataLoader): DataLoader for the evaluation data.
        device (str): The device to run the evaluation on ('cuda' or 'cpu').

    Returns:
        float: The accuracy of the model on the dataset.
    """
    
    # Set the model to evaluation mode
    # This turns off layers like Dropout for deterministic evaluation
    model.eval()
    
    total_correct = 0
    total_samples = 0
    
    # Disable gradient calculations to save memory and speed up computation
    with torch.no_grad():
        
        # Iterate through the evaluation data
        for batch_tokens, batch_labels in dataloader:
            
            # Move data to the same device as the model
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            
            # 1. Forward pass to get model predictions (logits)
            logits = model(batch_tokens)
            
            # 2. Get the predicted class by finding the index of the max logit
            # torch.argmax finds the index with the highest score along dimension 1
            predictions = torch.argmax(logits, dim=1)
            
            # 3. Compare predictions to the true labels and count correct ones
            total_correct += (predictions == batch_labels).sum().item()
            
            # 4. Count the total number of samples processed
            total_samples += batch_labels.size(0)
            
    # Calculate the final accuracy
    accuracy = total_correct / total_samples
    
    return accuracy       