# Lab 4: MLflow Experiment Tracking

CIFAR-10 classification with MLflow experiment tracking and artifact management.

## Setup

```bash
cd lab4
poetry install
```

## Usage

Run all experiments (1-batch, 3-batch, 5-batch):

```bash
deactivate  # if .venv is active
poetry run python -m src.main
```

View results in MLflow UI:

```bash
poetry run mlflow ui
# Open http://localhost:5000
```

## What's tracked in MLflow

Each config run logs:
- **Parameters**: all config values (learning rate, batch size, train/val batches, etc.)
- **Metrics per epoch**: train_loss, val_loss (viewable as charts)
- **Test metrics**: accuracy, precision, recall, f1_score, test_loss
- **Artifacts**: best_model.pth, loss_plot.png, config YAML, metrics.json

## Results

See [REPORT.md](REPORT.md) for details.
