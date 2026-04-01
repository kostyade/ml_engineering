# Lab 3: DVC Pipeline Automation

CIFAR-10 classification pipeline managed by DVC for reproducible ML workflows.

## Setup

```bash
poetry install
poetry run dvc init --subdir
```

## Usage

Run the full pipeline (download → train → evaluate):

```bash
poetry run dvc repro
```

View metrics:

```bash
poetry run dvc metrics show
```

Re-run after changing parameters (e.g. `learning_rate` in `params.yaml`):

```bash
poetry run dvc repro   # only re-runs affected stages
```

## Pipeline Stages

```
download → train → evaluate
```

- **download**: fetches CIFAR-10 dataset via torchvision
- **train**: loads selected batches, trains SimpleCNN, saves best model
- **evaluate**: loads trained model, evaluates on static test set, outputs `metrics.json`

## Results

```json
{
  "test_loss": 0.7674,
  "accuracy": 0.7419,
  "precision": 0.7459,
  "recall": 0.7423,
  "f1_score": 0.7427
}
```

See [REPORT.md](REPORT.md) for details.
