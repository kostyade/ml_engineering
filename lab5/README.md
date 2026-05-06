# Lab 5: Weights & Biases Experiment Tracking

CIFAR-10 classification with W&B experiment tracking and artifact management.

## Setup

```bash
cd lab5
poetry install
poetry run wandb login    # paste API key from https://wandb.ai/authorize
```

## Usage

Run all experiments (1-batch, 3-batch, 5-batch):

```bash
deactivate  # if .venv is active
poetry env use python3.11
poetry run python -m src.main
```

Open the W&B dashboard at `https://wandb.ai/<your-username>/cifar10-classification` to view results.

## What's tracked in W&B

Each config produces 2 grouped runs (train + evaluate):

- **Config (`wandb.config`):** all hyperparameters from the YAML
- **Per-epoch metrics:** `train_loss`, `val_loss` (visualized as charts)
- **Test metrics:** `accuracy`, `precision`, `recall`, `f1_score`, `test_loss`
- **Artifacts:**
  - `model-<run>` — best model weights (`best_model.pth`)
  - `metrics-<run>` — final test metrics JSON
  - `config-<config>` — original YAML config
- **Plots:** loss curve image, confusion matrix

## Run organization

Runs are **grouped by config**:

```
config_1batch (group)
├── config_1batch - training
└── config_1batch - evaluation
config_3batch (group)
├── config_3batch - training
└── config_3batch - evaluation
config_5batch (group)
├── config_5batch - training
└── config_5batch - evaluation
```

See [REPORT.md](REPORT.md) for full details and screenshots.
