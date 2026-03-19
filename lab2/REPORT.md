# Lab 2 Report: Automating Dataset Extension

## 1. Introduction

This lab investigates how the size and composition of training data affects model performance. Using the CIFAR-10 dataset (60,000 32x32 color images across 10 classes), we build a configuration-driven pipeline that selects specific data batches for training and validation while keeping the test set static. By varying the number of training batches (1, 3, or 4 out of 5 available), we demonstrate the relationship between dataset size and classification accuracy.

## 2. Pipeline Description

### Configuration Management

Each experiment is defined by a YAML config file in `config/`. The key parameters are:

```yaml
data:
  train_batches: [1, 2, 3] # which CIFAR-10 batch files to use for training
  val_batches: [4] # which batch files to use for validation
```

Three configs were created:

- **config_1batch.yaml** — train on batch 1 (10k samples), validate on batch 2
- **config_3batch.yaml** — train on batches 1-3 (30k samples), validate on batch 4
- **config_5batch.yaml** — train on batches 1-4 (40k samples), validate on batch 5

### Data Loading

CIFAR-10 is distributed as 5 training batches (10k samples each) plus a `test_batch` (10k samples). The `data_loader.py` module:

1. Downloads CIFAR-10 via torchvision if not present
2. Loads specific batch pickle files by number based on the config
3. Concatenates selected batches into numpy arrays
4. Normalizes pixel values and applies CIFAR-10 channel normalization
5. Wraps data in PyTorch `TensorDataset` and `DataLoader`

The **test set is always the static `test_batch`** (10k samples), providing a consistent evaluation metric across all experiments.

### Experiment Runner

`main.py` discovers all `config_*.yaml` files in the config directory and runs each experiment sequentially:

1. Load config and create DataLoaders from selected batches
2. Initialize a fresh `SimpleCNN` model
3. Train for 15 epochs with Adam optimizer
4. Evaluate on the static test set
5. Save metrics and loss plot per config
6. Print a comparison table and save `artifacts/comparison.yaml`

## 3. Model Evaluation

### Results

| Config           | Train Samples | Accuracy | Precision | Recall | F1 Score | Test Loss |
| ---------------- | ------------- | -------- | --------- | ------ | -------- | --------- |
| 1 batch          | 10,000        | 64.14%   | 64.72%    | 64.14% | 64.13%   | 1.0138    |
| 3 batches        | 30,000        | 73.22%   | 73.06%    | 73.22% | 72.99%   | 0.7913    |
| 4 batches (full) | 40,000        | 75.41%   | 75.56%    | 75.41% | 75.26%   | 0.7406    |

### Analysis

- **1 batch to 3 batches (10k → 30k):** The largest jump in performance — accuracy improves by ~9 percentage points (+14%). Tripling the data has a dramatic effect when starting from a small dataset.
- **3 batches to 4 batches (30k → 40k):** A more modest gain of ~2 percentage points. Diminishing returns are visible as the dataset grows.
- **Overfitting:** The 1-batch experiment shows clear overfitting — train loss drops to 0.15 while val loss rises to 1.58 by epoch 15. With more data (4 batches), the gap narrows significantly, confirming that more data acts as a regularizer.
- **All metrics move together:** Accuracy, precision, recall, and F1 score are closely aligned across all experiments, indicating balanced performance across classes.

## 4. Best Practices

- **Configuration Management** — all batch selection, hyperparameters, and paths are controlled via YAML config files. Adding a new experiment requires only creating a new config file.
- **Logging** — Python's `logging` module tracks data loading (which batches, sample counts), training progress (per-epoch losses), evaluation results, and the final comparison.
- **Code Quality** — `pyproject.toml` configures `black`, `isort`, `ruff`, and `mypy`. All functions have type hints and docstrings.
- **Dependency Management** — Poetry manages all dependencies via `pyproject.toml` and `poetry.lock`.
- **Version Control** — GitHub repository with meaningful commit history.
- **Modularity** — code is split into `data_loader.py`, `model.py`, `train.py`, `evaluate.py`, and `main.py`. The model, training loop, and evaluation are reused from Lab 1 unchanged.

## 5. Reflection

The experiment clearly demonstrates that dataset size is a critical factor in model performance. Key takeaways:

- **Diminishing returns:** Going from 10k to 30k training samples yields a much larger improvement than going from 30k to 40k. This suggests a logarithmic relationship between data size and accuracy for this model/task.
- **Static test set matters:** Using the same test set across all experiments ensures a fair comparison. Without this, differences in test set difficulty could confound the results.
- **No augmentation in this experiment:** Unlike Lab 1, data augmentation (RandomHorizontalFlip, RandomCrop) was deliberately omitted to isolate the effect of raw data quantity. Adding augmentation would likely improve all configs but especially benefit the smaller datasets.

Possible improvements:

- Add data augmentation and measure whether it reduces the gap between small and large datasets
- Test with more granular batch counts (2, 4 batches) for a smoother curve
- Use early stopping to prevent overfitting, especially on smaller datasets
- Try a larger model (e.g., ResNet-18) to see if more data helps more with higher-capacity models
