# \# Transformer-Based News Classifier

# 

# This project provides a complete implementation of a Transformer-based text classifier for the AG News dataset. It is built from scratch using PyTorch and structured as a professional, reusable Python package.

# 

# The repository includes scripts for training the model, running inference on new headlines, and a full suite of unit tests. The goal is to provide a clean, well-documented, and maintainable example of a modern NLP project.

# 

# \## Features

# \- \*\*Transformer Architecture:\*\* A from-scratch implementation of a Transformer encoder for classification.

# \- \*\*Packaged Code:\*\* All logic is contained within the installable `transformer\_news` package in the `src/` directory.

# \- \*\*Clear Entry Points:\*\* Separate, user-friendly scripts (`run\_train.py`, `run\_prediction.py`) for training and inference.

# \- \*\*Reproducible Environment:\*\* A `requirements.txt` file ensures a consistent setup.

# \- \*\*Unit \& Integration Tests:\*\* A `tests/` directory with `pytest` tests to validate model functionality and the prediction pipeline.

# \- \*\*Experiment-Friendly:\*\* Easily switch between using a full dataset or a smaller sample for quick training runs.

# 

# \## Project Structure

# ```

# ├── models/                # Saved model weights and vocabulary artifacts

# ├── src/

# │   └── transformer\_news/    # The core, installable Python package

# ├── tests/                 # Unit and integration tests

# ├── .gitignore             # Specifies files for Git to ignore

# ├── pyproject.toml         # Modern Python project configuration

# ├── README.md              # You are here!

# ├── requirements.txt       # Project dependencies

# ├── run\_prediction.py      # Entry-point script for inference

# └── run\_train.py           # Entry-point script for training

# ```

# 

# \## Quick Start

# 

# Follow these steps to set up and run the project.

# 

# \### 1. Prerequisites

# \- Python 3.9+

# \- An NVIDIA GPU is recommended for training.

# 

# \### 2. Installation

# First, clone the repository and navigate into the project directory:

# ```bash

# git clone https://github.com/YOUR\_USERNAME/transformer-from-scratch-news-classifier.git

# cd transformer-from-scratch-news-classifier

# ```

# 

# Next, create and activate a virtual environment:

# ```bash

# python -m venv venv

# source venv/bin/activate  # On Windows, use `venv\\Scripts\\activate`

# ```

# 

# Finally, install the project dependencies and the `transformer\_news` package in editable mode:

# ```bash

# pip install -r requirements.txt

# pip install -e .

# ```

# \*(`pip install -e .` makes your `src/transformer\_news` package importable everywhere in your environment.)\*

# 

# \## Usage

# 

# \### Training the Model

# You can train the model using the `run\_train.py` script.

# 

# \*\*To train on a small sample (for a quick test):\*\*

# ```bash

# python run\_train.py --use-sample --num-epochs 5 --sample-size 2000

# ```

# 

# \*\*To train on the full AG News dataset:\*\*

# ```bash

# python run\_train.py --num-epochs 10

# ```

# The best performing model will be saved to `models/transformer\_news\_classifier\_best.pth`.

# 

# \### Running Inference

# Once a model is trained, you can classify new headlines using `run\_prediction.py`.

# 

# ```bash

# python run\_prediction.py --news\_article\_headline "NASA discovers new planet in a distant galaxy"

# ```

# \*\*Output:\*\*

# ```

# Predicted Category: Sci/Tech

# ```

# 

# \## Running Tests

# This project uses `pytest` for testing. To run the full test suite, navigate to the project root and run:

# ```bash

# pytest

# ```

# 

# \## Acknowledgments

# The core concepts and architectural patterns implemented here were learned from and inspired by several excellent educational resources, including Jay Alammar's "The Illustrated Transformer" and Andrej Karpathy's "Let's build GPT".

