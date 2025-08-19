# Transformer from Scratch for News Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

A from-scratch implementation of the Transformer architecture for classifying news articles into different categories. This project is intended as a deep dive into the mechanics of the Transformer model as described in the paper "Attention Is All You Need."

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running Inference](#running-inference)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

This repository contains a pure Python implementation of a Transformer-based text classifier, built from the ground up with minimal reliance on high-level deep learning libraries for the core model architecture. The goal is to provide a clear and understandable codebase for educational purposes, demonstrating how to build and train a Transformer for a common NLP task like news classification.

## Key Features

*   **From-Scratch Implementation**: The core components of the Transformer (Self-Attention, Multi-Head Attention, Positional Encoding, Encoder layers) are built from scratch.
*   **Modular Code**: The code is organized into logical modules for data processing, model architecture, training, and inference.
*   **Detailed Comments**: The source code is commented to explain complex parts of the Transformer architecture.
*   **Customizable Training**: The training script includes arguments for customizing hyperparameters like learning rate, batch size, and number of epochs.

## Model Architecture

The model is an implementation of the original Transformer's encoder stack.
*   **Input Embeddings**: Converts input text tokens into dense vectors.
*   **Positional Encoding**: Injects information about the position of tokens in the sequence.
*   **Multi-Head Self-Attention**: Allows the model to weigh the importance of different words in the input sequence.
*   **Feed-Forward Networks**: A fully connected feed-forward network applied to each position separately and identically.
*   **Layer Normalization & Skip Connections**: Used to stabilize training and improve gradient flow.

The final output from the encoder stack is passed through a linear layer and a softmax function to produce the probability distribution over the news categories.

## Dataset

This model was trained on the [**AG News**](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) dataset (or specify your dataset here). The dataset consists of news articles classified into four categories: World, Sports, Business, and Sci/Tech.

| Category  | Number of Samples |
|-----------|-------------------|
| World     | 120,000           |
| Sports    | 120,000           |
| Business  | 120,000           |
| Sci/Tech  | 120,000           |

A preprocessing script is included to clean the text, tokenize it, and build a vocabulary.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8 or higher
*   pip package manager

You will also need to download the required dataset and place it in the `data/` directory.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/sadhchanGitHub/transformer-from-scratch-news-classifier.git
    cd transformer-from-scratch-news-classifier
    ```

2.  **Create a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

The project includes scripts for training the model from scratch and running inference on new text.

### Training the Model

To start training the model, run the `train.py` script. You can customize the training process using command-line arguments.

```sh
python src/train.py --data_path data/train.csv --epochs 10 --batch_size 32 --learning_rate 0.0001
```

### Running Inference

Once a model is trained and saved, you can use the `predict.py` script to classify a new piece of text.

```sh
python src/predict.py --model_path models/best_model.pt --text "The stock market saw a new high today as tech companies soared."
```

## Results

After training for 10 epochs, the model achieved the following performance on the test set:

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 91.5%   |
| Precision | 0.91    |
| Recall    | 0.91    |
| F1-Score  | 0.91    |

A confusion matrix and detailed classification report can be found in the `notebooks/` directory.

## Project Structure

```
.
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   └── best_model.pt
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   └── predict.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Contributing

Contributions are welcome! If you have suggestions for improving this project, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

*   The core concepts and architectural patterns implemented here were learned from and inspired by several excellent educational resources, including Jay Alammar's "The Illustrated Transformer" and Andrej Karpathy's "Let's build GPT".
