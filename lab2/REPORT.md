# Lab 2 Report: Automating Dataset Extension

## 1. Introduction

This lab investigates how the size and composition of training data affects model performance. Using the CIFAR-10 dataset (50,000 32x32 color images across 10 classes), we build a configuration-driven pipeline that splits the data into a static test set and dynamically selected train/val batches. By varying the number of training batches (1, 3, or 4 out of 5 available), we demonstrate the relationship between dataset size and classification accuracy.

## 2. Pipeline Description

### Configuration Management

Each experiment is defined by a YAML config file in `config/`. The key parameters are:

```yaml
data:
  test_size: 0.2         # fraction split off as static test set
  random_state: 42       # seed for reproducible test split
  n_batches: 5           # divide remaining data into N equal chunks
  train_batches: [0, 1, 2]  # which batch indices to use for training
  val_batches: [3]           # which batch indices to use for validation
```

Three configs were created:

- **config_1batch.yaml** — train on batch 0 (8k samples), validate on batch 1
- **config_3batch.yaml** — train on batches 0-2 (24k samples), validate on batch 3
- **config_5batch.yaml** — train on batches 0-3 (32k samples), validate on batch 4

### Data Loading

The `data_loader.py` module implements the following pipeline:

1. Downloads all 5 CIFAR-10 training batches (50k samples total) via torchvision
2. Splits off a **static test set** (20% = 10k samples, `random_state=42`) — this set never changes across experiments
3. Divides the remaining 40k samples into 5 equal batches (8k each)
4. Selects which batches go to train and which to val based on the config
5. Normalizes pixel values and applies CIFAR-10 channel normalization
6. Wraps data in PyTorch `TensorDataset` and `DataLoader`

The test split uses a fixed random seed, ensuring the **same 10k test samples** are used across all experiments for fair comparison. Train and val sets are created **dynamically** based on the configuration file.

### Experiment Runner

`main.py` discovers all `config_*.yaml` files in the config directory and runs each experiment sequentially:

1. Load config and create DataLoaders (static test + dynamic train/val)
2. Initialize a fresh `SimpleCNN` model
3. Train for 15 epochs with Adam optimizer
4. Evaluate on the static test set
5. Save metrics and loss plot per config
6. Print a comparison table and save `artifacts/comparison.yaml`

## 3. Model Evaluation

### Results

| Config           | Train Samples | Val Samples | Accuracy | Precision | Recall | F1 Score | Test Loss |
| ---------------- | ------------- | ----------- | -------- | --------- | ------ | -------- | --------- |
| 1 batch          | 8,000         | 8,000       | 63.57%   | 64.60%    | 63.58% | 63.62%   | 1.0468    |
| 3 batches        | 24,000        | 8,000       | 72.30%   | 72.33%    | 72.35% | 72.25%   | 0.8362    |
| 4 batches (full) | 32,000        | 8,000       | 74.30%   | 74.68%    | 74.38% | 74.35%   | 0.7763    |

All metrics are evaluated on the same static test set of 10,000 samples.

### Analysis

- **1 batch to 3 batches (8k → 24k):** The largest jump in performance — accuracy improves by ~9 percentage points. Tripling the training data has a dramatic effect when starting from a small dataset.
- **3 batches to 4 batches (24k → 32k):** A more modest gain of ~2 percentage points. Diminishing returns are visible as the dataset grows.
- **Overfitting:** The 1-batch experiment shows clear overfitting — train loss drops to 0.20 while val loss rises to 1.48 by epoch 15. With more data (4 batches), the gap narrows (train loss 0.18 vs val loss 1.12), confirming that more data acts as a regularizer.
- **All metrics move together:** Accuracy, precision, recall, and F1 score are closely aligned across all experiments, indicating balanced performance across classes.

## 4. Best Practices

- **Configuration Management** — all data splitting parameters (`test_size`, `n_batches`, `train_batches`, `val_batches`), hyperparameters, and paths are controlled via YAML config files. Adding a new experiment requires only creating a new config file.
- **Logging** — Python's `logging` module tracks data loading (all 50k samples loaded, test split sizes, batch assignment, batch selection), training progress (per-epoch losses), evaluation results, and the final comparison table.
- **Code Quality** — `pyproject.toml` configures `black`, `isort`, `ruff`, and `mypy`. All functions have type hints and docstrings.
- **Dependency Management** — Poetry manages all dependencies via `pyproject.toml` and `poetry.lock`.
- **Version Control** — GitHub repository with meaningful commit history.
- **Modularity** — code is split into `data_loader.py`, `model.py`, `train.py`, `evaluate.py`, and `main.py`. The model, training loop, and evaluation are reused from Lab 1 unchanged.

## 5. Reflection

The experiment clearly demonstrates that dataset size is a critical factor in model performance. Key takeaways:

- **Diminishing returns:** Going from 8k to 24k training samples yields a much larger improvement than going from 24k to 32k. This suggests a logarithmic relationship between data size and accuracy for this model/task.
- **Static test set matters:** Using the same 10k test samples (split with a fixed seed) across all experiments ensures a fair comparison. Without this, differences in test set composition could confound the results.
- **Dynamic train/val via config:** The batch-based approach makes it easy to run new experiments — just change `train_batches` and `val_batches` in the YAML file, no code changes needed.
- **No augmentation in this experiment:** Unlike Lab 1, data augmentation (RandomHorizontalFlip, RandomCrop) was deliberately omitted to isolate the effect of raw data quantity. Adding augmentation would likely improve all configs but especially benefit the smaller datasets.

Possible improvements:

- Add data augmentation and measure whether it reduces the gap between small and large datasets
- Test with more granular batch counts (2, 4 batches) for a smoother curve
- Use early stopping to prevent overfitting, especially on smaller datasets
- Try a larger model (e.g., ResNet-18) to see if more data helps more with higher-capacity models
