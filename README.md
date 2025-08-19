<div align="center">
Transformer-Based News Classifier
A from-scratch implementation of a Transformer text classifier, structured as a professional, installable Python package with a full testing suite.
![alt text](https://img.shields.io/badge/Python-3.9%2B-blue.svg)

![alt text](https://img.shields.io/badge/Built%20with-PyTorch-FF69B4.svg)

![alt text](https://img.shields.io/badge/Tests-Passing-green.svg)
</div>
This project provides a complete implementation of a Transformer-based text classifier for the AG News dataset. It is built from scratch using PyTorch and structured as a professional, reusable Python package.
The repository includes scripts for training the model, running inference on new headlines, and a full suite of unit tests. The goal is to provide a clean, well-documented, and maintainable example of a modern NLP project.
Features
Transformer Architecture: A from-scratch implementation of a Transformer encoder for classification.
Packaged Code: All logic is contained within the installable transformer_news package in the src/ directory.
Clear Entry Points: Separate, user-friendly scripts (train.py, predict.py) for training and inference.
Reproducible Environment: A requirements.txt file ensures a consistent setup.
Unit & Integration Tests: A tests/ directory with pytest tests to validate model functionality and the prediction pipeline.
Experiment-Friendly: Easily switch between using a full dataset or a smaller sample for quick training runs.
Project Structure
code
Code
├── models/                # Saved model weights and vocabulary artifacts
├── src/
│   └── transformer_news/    # The core, installable Python package
├── tests/                 # Unit and integration tests
├── .gitignore             # Specifies files for Git to ignore
├── pyproject.toml         # Modern Python project configuration
├── README.md              # You are here!
├── requirements.txt       # Project dependencies
├── predict.py             # Entry-point script for inference
└── train.py               # Entry-point script for training
Quick Start
Follow these steps to set up and run the project.
1. Prerequisites
Python 3.9+
An NVIDIA GPU is recommended for training.
2. Installation
First, clone the repository and navigate into the project directory:```bash
git clone https://github.com/sadhchanGitHub/transformer-from-scratch-news-classifier.git
cd transformer-from-scratch-news-classifier
code
Code
Next, create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Finally, install the project dependencies and the transformer_news package in editable mode:
code
Bash
pip install -r requirements.txt
pip install -e .
```*(Note: `pip install -e .` makes your `src/transformer_news` package importable everywhere in your environment.)*

## Usage

### Training the Model
You can train the model using the `train.py` script.

**To train on a small sample (for a quick test):**
```bash
python train.py --use-sample --num-epochs 5 --sample-size 2000
To train on the full AG News dataset:
code
Bash
python train.py --num-epochs 10
The best performing model will be saved to models/transformer_news_classifier_best.pth.
Running Inference
Once a model is trained, you can classify new headlines using predict.py.
code
Bash
python predict.py --news_article_headline "NASA discovers new planet in a distant galaxy"
Expected Output:
code
Code
Predicted Category: Sci/Tech
Running Tests
This project uses pytest for testing. To run the full test suite, navigate to the project root and run:
code
Bash
pytest
Acknowledgments
The core concepts and architectural patterns implemented here were learned from and inspired by several excellent educational resources, including Jay Alammar's "The Illustrated Transformer" and Andrej Karpathy's "Let's build GPT".
Start typing a prompt
