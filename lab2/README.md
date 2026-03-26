# Lab 2: Automating Dataset Extension

CIFAR-10 image classification with config-driven batch selection to study how dataset size affects model performance.

## Setup

```bash
poetry install
```

## Usage

Run all experiments (1-batch, 3-batch, 5-batch configs):

```bash
poetry run python -m src.main
```

This trains a fresh CNN for each config, evaluates on the static test set, and saves per-config metrics to `artifacts/<config>/metrics.yaml` plus a comparison to `artifacts/comparison.yaml`.

## Results

| Config            | Accuracy | Precision | Recall | F1 Score | Test Loss |
|-------------------|----------|-----------|--------|----------|-----------|
| 1 batch (8k)      | 63.57%   | 64.60%    | 63.58% | 63.62%   | 1.0468    |
| 3 batches (24k)   | 72.30%   | 72.33%    | 72.35% | 72.25%   | 0.8362    |
| 4 batches (32k)   | 74.30%   | 74.68%    | 74.38% | 74.35%   | 0.7763    |

More training data consistently improves all metrics. See [REPORT.md](REPORT.md) for details.
