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

| Config       | Accuracy | Precision | Recall | F1 Score | Test Loss |
|--------------|----------|-----------|--------|----------|-----------|
| 1 batch (10k)  | 64.14%   | 64.72%    | 64.14% | 64.13%   | 1.0138    |
| 3 batches (30k) | 73.22%   | 73.06%    | 73.22% | 72.99%   | 0.7913    |
| 5 batches (40k) | 75.41%   | 75.56%    | 75.41% | 75.26%   | 0.7406    |

More training data consistently improves all metrics. See [REPORT.md](REPORT.md) for details.
