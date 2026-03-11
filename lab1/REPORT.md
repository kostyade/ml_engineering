# Lab 1 Report: Basic ML Training Pipeline

## Introduction

This lab implements an image classification pipeline using the CIFAR-10 dataset — 60,000 32x32 color images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The goal is to build a complete training pipeline with a CNN model following MLOps best practices.

## Pipeline Description

The pipeline consists of the following stages:

1. **Configuration** — all hyperparameters (learning rate, batch size, epochs, etc.) and paths are loaded from `config/config.yaml`. No values are hardcoded.
2. **Data Download** — CIFAR-10 is downloaded automatically via `torchvision.datasets.CIFAR10`. Training data is augmented with `RandomHorizontalFlip` and `RandomCrop` to improve generalization.
3. **Data Ingestion & Splitting** — the 50k training set is split into 40k train and 10k validation subsets. The 10k test set is kept separate.
4. **Model** — `SimpleCNN` with 3 convolutional layers (32 -> 64 -> 128 filters), max pooling, and a classifier with dropout (0.3). Replaces the template's `SimpleNN` which was too small for meaningful results.
5. **Training Loop** — trains for 15 epochs using Adam optimizer and CrossEntropyLoss. Validates after each epoch. Saves the best model checkpoint based on validation loss.
6. **Evaluation** — loads the best model and computes test loss, accuracy, precision, recall, and F1 score using scikit-learn.
7. **Artifact Collection** — saves `best_model.pth`, `loss_plot.png`, and `metrics.yaml` to `artifacts/`.

## Model Evaluation

Results after 15 epochs of training on CPU (~18 min total):

| Metric            | Value  |
| ----------------- | ------ |
| Test Loss         | 0.6358 |
| Accuracy          | 78.27% |
| Precision (macro) | 78.28% |
| Recall (macro)    | 78.27% |
| F1 Score (macro)  | 78.02% |

Training loss decreased steadily from 1.62 (epoch 1) to 0.64 (epoch 15). Validation loss converged to 0.68 by epoch 12, with the best checkpoint saved at that point. The small gap between train loss (0.64) and val loss (0.68) indicates mild overfitting, effectively mitigated by dropout (0.3) and data augmentation (RandomHorizontalFlip + RandomCrop).

## Best Practices

- **Configuration Management** — `config/config.yaml` controls all parameters; changes require no code edits.
- **Logging** — Python's `logging` module is used throughout all modules. No `print()` statements.
- **Code Quality** — `pyproject.toml` configures `black`, `isort`, `ruff`, and `mypy`. All functions have type hints.
- **Dependency Management** — Poetry manages all dependencies via `pyproject.toml` and `poetry.lock`.
- **Version Control** — GitHub repository with meaningful commit history.
- **Project Structure** — modular code split into `data_download.py`, `data_ingestion.py`, `model.py`, `train.py`, `evaluate.py`, and `main.py`.

## Reflection

The SimpleCNN provides a solid baseline for CIFAR-10. Possible improvements with more time:

- Use a pretrained ResNet-18 with fine-tuning for better accuracy
- Train longer (50-100 epochs) with early stopping
