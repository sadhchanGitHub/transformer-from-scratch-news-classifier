# tests/test_model_shape.py

import torch
import pytest
from transformer_news.model import TransformerClassifier
from transformer_news import config

# We can define the dummy vocab size here to be used by both the fixture and the test
DUMMY_VOCAB_SIZE = 10000

@pytest.fixture(scope="session")
def initialized_model():
    """Initializes and returns the TransformerClassifier model."""
    model = TransformerClassifier(
        vocab_size=DUMMY_VOCAB_SIZE,  # Use the constant here
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        d_ff=config.D_FF,
        num_classes=config.NUM_CLASSES
    )
    model.eval()
    return model

def test_output_shape(initialized_model):
    """
    Tests if the model output has the correct shape (batch_size, num_classes).
    """
    # 1. Define dummy input parameters
    batch_size = 4
    seq_length = 50

    # 2. Create a dummy input tensor
    #    Use the same constant we defined earlier
    dummy_input = torch.randint(0, DUMMY_VOCAB_SIZE, (batch_size, seq_length))

    # 3. Pass the dummy input through the model
    with torch.no_grad():
        output = initialized_model(dummy_input)

    # 4. Assert that the output shape is what we expect
    expected_shape = (batch_size, config.NUM_CLASSES)
    assert output.shape == expected_shape, \
        f"Model output shape is {output.shape}, but expected {expected_shape}"