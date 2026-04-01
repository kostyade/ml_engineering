# Lab 3 Report: DVC Pipeline Automation

## 1. Introduction

In machine learning projects, reproducibility is a major challenge. Models depend on specific versions of data, code, and hyperparameters — changing any of these can produce different results. **Data Version Control (DVC)** solves this by tracking data files, defining pipeline stages with explicit dependencies, and caching outputs so that only affected stages re-run when something changes.

In this lab, we applied DVC to our CIFAR-10 classification pipeline from Lab 2, converting it into a 3-stage reproducible pipeline managed by `dvc.yaml` and `params.yaml`.

## 2. Pipeline Description

### Stages

The pipeline is defined in `dvc.yaml` with three stages:

```
download → train → evaluate
```

**Stage 1 — download:**

- Command: `python -m src.download`
- Params: `data.save_dir`
- Outputs: `data/cifar-10-batches-py/` (DVC-tracked)
- Downloads CIFAR-10 dataset via torchvision

**Stage 2 — train:**

- Command: `python -m src.train`
- Depends on: downloaded data, `src/train.py`, `src/model.py`, `src/data_loader.py`
- Params: `data`, `training`, `model` sections
- Outputs: `artifacts/best_model.pth`, `artifacts/loss_plot.png`
- Loads selected batches from config, splits off static test set, trains SimpleCNN for 15 epochs

**Stage 3 — evaluate:**

- Command: `python -m src.evaluate`
- Depends on: `artifacts/best_model.pth`, `src/evaluate.py`, `src/data_loader.py`
- Params: `data`, `model` sections
- Metrics: `artifacts/metrics.json` (committed to git, not cached)
- Loads best model, evaluates on the static test set (10k samples)

### Reproducibility with `dvc repro`

DVC tracks the hash of every dependency and output. When running `dvc repro`:

- If nothing changed → all stages are skipped ("cached")
- If `params.yaml` training parameters change → only `train` + `evaluate` re-run, `download` is skipped
- If `src/model.py` changes → `train` + `evaluate` re-run
- If data download URL changes → all 3 stages re-run

This is recorded in `dvc.lock`, which is committed to git alongside the code.

## 3. Parameterization

All pipeline parameters are stored in `params.yaml`:

```yaml
data:
  save_dir: "./data"
  test_size: 0.2 # fraction of data reserved for static test set
  random_state: 42 # seed for reproducible test split
  n_batches: 5 # divide remaining data into N chunks
  train_batches: [0, 1, 2, 3] # batch indices for training
  val_batches: [4] # batch indices for validation

training:
  batch_size: 64
  num_workers: 2
  num_epochs: 15
  learning_rate: 0.001

model:
  n_classes: 10

artifacts:
  save_dir: "./artifacts"
  best_model_name: "best_model.pth"
```

Each stage declares which params it depends on in `dvc.yaml`. For example, the `download` stage only watches `data.save_dir`, so changing `learning_rate` won't trigger a re-download.

### Results

| Metric    | Value  |
| --------- | ------ |
| Test Loss | 0.7674 |
| Accuracy  | 74.19% |
| Precision | 74.59% |
| Recall    | 74.23% |
| F1 Score  | 74.27% |

Results are consistent with Lab 2's 4-batch configuration (32k train samples), confirming the pipeline produces equivalent output.

## 4. Best Practices

- **Configuration Management** — all parameters in `params.yaml`, DVC watches for changes
- **Logging** — Python's `logging` module in every stage script, tracking start/end of each stage and key events
- **Code Quality** — `pyproject.toml` configures `black`, `isort`, `ruff`, `mypy`
- **Dependency Management** — Poetry manages all dependencies including DVC
- **Version Control** — `dvc.lock` committed to git alongside code; data tracked by DVC

## 5. Reflection

**Benefits of DVC:**

- **Selective re-runs:** Changing only `learning_rate` skips the download stage entirely, saving significant time
- **Reproducibility:** Any collaborator can run `dvc repro` and get the same results from the same `params.yaml` + `dvc.lock`
- **Clear dependency graph:** `dvc.yaml` makes it explicit what each stage needs and produces
- **Metrics tracking:** `dvc metrics show` and `dvc metrics diff` make it easy to compare experiments

**Challenges:**

- **Large data:** CIFAR-10 is relatively small; for larger datasets, a proper DVC remote (S3, GCS) would be needed instead of local storage
- **Overhead for simple projects:** For a single-config pipeline, DVC adds files (`dvc.yaml`, `dvc.lock`, `.dvc/`) that may feel excessive

**Possible improvements:**

- Set up a cloud DVC remote (e.g. S3) for team collaboration
- Use `dvc experiments` for systematic hyperparameter exploration
- Add a `process_data` stage to separate data splitting from training
- Integrate with CI/CD to automatically run `dvc repro` on pull requests
