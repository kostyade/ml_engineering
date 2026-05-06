"""Batch-aware data loading for CIFAR-10 dataset extension experiments."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

logger = logging.getLogger(__name__)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def ensure_cifar10_downloaded(save_dir: str) -> Path:
    """Download CIFAR-10 if not already present and return the batch directory path."""
    CIFAR10(root=save_dir, train=True, download=True)
    batch_dir = Path(save_dir) / "cifar-10-batches-py"
    logger.info("CIFAR-10 data available at %s", batch_dir)
    return batch_dir


def load_cifar_batch(batch_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single CIFAR-10 batch pickle file."""
    with open(batch_path, "rb") as f:
        batch_dict = pickle.load(f, encoding="bytes")
    data = batch_dict[b"data"]
    labels = batch_dict[b"labels"]
    # Reshape from (N, 3072) to (N, 3, 32, 32)
    data = data.reshape(-1, 3, 32, 32)
    return data, np.array(labels)


def load_all_train_batches(batch_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load all 5 CIFAR-10 training batch files and concatenate them."""
    all_data = []
    all_labels = []
    for i in range(1, 6):
        batch_path = batch_dir / f"data_batch_{i}"
        data, labels = load_cifar_batch(batch_path)
        all_data.append(data)
        all_labels.append(labels)
        logger.info("Loaded data_batch_%d: %d samples", i, len(labels))

    combined_data = np.concatenate(all_data, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    logger.info("All training data loaded: %d total samples", len(combined_labels))
    return combined_data, combined_labels


def split_test_set(
    data: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split off a static test set from the full dataset.

    Args:
    - data: Full dataset images.
    - labels: Full dataset labels.
    - test_size: Fraction of data to use for test (e.g. 0.2).
    - random_state: Seed for reproducible split.

    Returns:
    - Tuple of (remaining_data, remaining_labels, test_data, test_labels).
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(data)
    indices = rng.permutation(n_samples)
    n_test = int(n_samples * test_size)

    test_indices = indices[:n_test]
    remaining_indices = indices[n_test:]

    logger.info(
        "Test split: %d test samples, %d remaining samples (seed=%d)",
        len(test_indices), len(remaining_indices), random_state,
    )
    return data[remaining_indices], labels[remaining_indices], data[test_indices], labels[test_indices]


def assign_batches(
    data: np.ndarray,
    labels: np.ndarray,
    n_batches: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Divide data into n_batches equal-sized chunks.

    Args:
    - data: Images array.
    - labels: Labels array.
    - n_batches: Number of batches to divide into.

    Returns:
    - List of (data, labels) tuples, one per batch.
    """
    batch_size = len(data) // n_batches
    batches = []
    for i in range(n_batches):
        if i == n_batches - 1:
            # Last batch gets remaining samples
            batch_data = data[i * batch_size:]
            batch_labels = labels[i * batch_size:]
        else:
            batch_data = data[i * batch_size: (i + 1) * batch_size]
            batch_labels = labels[i * batch_size: (i + 1) * batch_size]
        batches.append((batch_data, batch_labels))
        logger.info("Batch %d: %d samples", i, len(batch_labels))

    return batches


def select_and_combine_batches(
    batches: List[Tuple[np.ndarray, np.ndarray]],
    batch_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Select specific batches by index and concatenate them."""
    selected_data = [batches[i][0] for i in batch_indices]
    selected_labels = [batches[i][1] for i in batch_indices]
    combined_data = np.concatenate(selected_data, axis=0)
    combined_labels = np.concatenate(selected_labels, axis=0)
    logger.info("Selected batches %s: %d total samples", batch_indices, len(combined_labels))
    return combined_data, combined_labels


def numpy_to_dataloader(
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Convert numpy arrays to a DataLoader with normalization applied."""
    # Normalize to [0, 1] float first
    tensor_data = torch.tensor(data, dtype=torch.float32) / 255.0
    # Apply channel normalization
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    tensor_data = normalize(tensor_data)
    tensor_labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(tensor_data, tensor_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


def create_data_loaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Flow:
    1. Load all 5 CIFAR-10 training batches (50k samples)
    2. Split off a static test set (test_size fraction, fixed seed)
    3. Divide remaining data into n_batches chunks
    4. Select train batches and val batches from config
    5. Return DataLoaders

    Args:
    - config: Configuration dictionary.

    Returns:
    - Tuple of (train_loader, val_loader, test_loader).
    """
    data_cfg = config["data"]
    train_cfg = config["training"]

    batch_dir = ensure_cifar10_downloaded(data_cfg["save_dir"])

    # Step 1: Load all data
    all_data, all_labels = load_all_train_batches(batch_dir)

    # Step 2: Split off static test set
    remaining_data, remaining_labels, test_data, test_labels = split_test_set(
        all_data, all_labels,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
    )

    # Step 3: Divide remaining data into batches
    n_batches = data_cfg["n_batches"]
    batches = assign_batches(remaining_data, remaining_labels, n_batches)

    # Step 4: Select train and val batches from config
    train_batch_indices: List[int] = data_cfg["train_batches"]
    val_batch_indices: List[int] = data_cfg["val_batches"]

    logger.info("Train batch indices: %s, Val batch indices: %s", train_batch_indices, val_batch_indices)

    train_data, train_labels = select_and_combine_batches(batches, train_batch_indices)
    val_data, val_labels = select_and_combine_batches(batches, val_batch_indices)

    # Step 5: Create DataLoaders
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg["num_workers"]

    train_loader = numpy_to_dataloader(train_data, train_labels, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = numpy_to_dataloader(val_data, val_labels, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = numpy_to_dataloader(test_data, test_labels, batch_size, shuffle=False, num_workers=num_workers)

    logger.info(
        "DataLoaders created: train=%d, val=%d, test=%d samples",
        len(train_data), len(val_data), len(test_data),
    )
    return train_loader, val_loader, test_loader
