<div align="center">

# Transformer-Based News Classifier

**A from-scratch implementation of a Transformer text classifier, structured as a professional, installable Python package with a full testing suite.**

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Built with PyTorch](https://img.shields.io/badge/Built%20with-PyTorch-FF69B4.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing-green.svg)](tests/)

</div>

This project provides a complete implementation of a Transformer-based text classifier for the AG News dataset. It is built from scratch using PyTorch and structured as a professional, reusable Python package.

The repository includes scripts for training the model, running inference on new headlines, and a full suite of unit tests. The goal is to provide a clean, well-documented, and maintainable example of a modern NLP project.

## Features
- **Transformer Architecture:** A from-scratch implementation of a Transformer encoder for classification.
- **Packaged Code:** All logic is contained within the installable `transformer_news` package in the `src/` directory.
- **Clear Entry Points:** Separate, user-friendly scripts (`train.py`, `predict.py`) for training and inference.
- **Reproducible Environment:** A `requirements.txt` file ensures a consistent setup.
- **Unit & Integration Tests:** A `tests/` directory with `pytest` tests to validate model functionality and the prediction pipeline.
- **Experiment-Friendly:** Easily switch between using a full dataset or a smaller sample for quick training runs.

## Project Structure
