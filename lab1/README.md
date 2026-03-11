# Lab 1: Basic ML Training Pipeline

CIFAR-10 image classification using a CNN built with PyTorch.

## Setup

```bash
poetry install
```

## Usage

```bash
poetry run python src/main.py
```

Runs the full pipeline: download data, train the model (15 epochs), evaluate, and save artifacts to `artifacts/`.

## Results

78.27% test accuracy. See [REPORT.md](REPORT.md) for details.
