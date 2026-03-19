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
    CIFAR10(root=save_dir, train=False, download=True)
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


def load_selected_batches(batch_dir: Path, batch_numbers: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Load and concatenate multiple CIFAR-10 batch files by number."""
    all_data = []
    all_labels = []
    for num in batch_numbers:
        batch_path = batch_dir / f"data_batch_{num}"
        if not batch_path.exists():
            raise FileNotFoundError(f"Batch file not found: {batch_path}")
        data, labels = load_cifar_batch(batch_path)
        all_data.append(data)
        all_labels.append(labels)
        logger.info("Loaded data_batch_%d: %d samples", num, len(labels))

    combined_data = np.concatenate(all_data, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    logger.info("Combined %d batches: %d total samples", len(batch_numbers), len(combined_labels))
    return combined_data, combined_labels


def load_test_batch(batch_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load the static CIFAR-10 test batch."""
    test_path = batch_dir / "test_batch"
    data, labels = load_cifar_batch(test_path)
    logger.info("Loaded test_batch: %d samples", len(labels))
    return data, labels


def numpy_to_dataloader(
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    transform: transforms.Compose,
) -> DataLoader:
    """Convert numpy arrays to a DataLoader with transforms applied."""
    # Normalize to [0, 1] float first
    tensor_data = torch.tensor(data, dtype=torch.float32) / 255.0
    # Apply normalization (transforms.Normalize expects (N, C, H, W) tensor)
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
    Create train, validation, and test DataLoaders based on config batch selection.

    Args:
    - config: Configuration dictionary with data.train_batches, data.val_batches, etc.

    Returns:
    - Tuple of (train_loader, val_loader, test_loader).
    """
    data_cfg = config["data"]
    train_cfg = config["training"]

    batch_dir = ensure_cifar10_downloaded(data_cfg["save_dir"])

    train_batches: List[int] = data_cfg["train_batches"]
    val_batches: List[int] = data_cfg["val_batches"]

    logger.info("Train batches: %s, Val batches: %s", train_batches, val_batches)

    # Load selected batches
    train_data, train_labels = load_selected_batches(batch_dir, train_batches)
    val_data, val_labels = load_selected_batches(batch_dir, val_batches)
    test_data, test_labels = load_test_batch(batch_dir)

    # Transforms (no augmentation for simplicity — raw batch comparison)
    transform = transforms.Compose([
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg["num_workers"]

    train_loader = numpy_to_dataloader(train_data, train_labels, batch_size, shuffle=True, num_workers=num_workers, transform=transform)
    val_loader = numpy_to_dataloader(val_data, val_labels, batch_size, shuffle=False, num_workers=num_workers, transform=transform)
    test_loader = numpy_to_dataloader(test_data, test_labels, batch_size, shuffle=False, num_workers=num_workers, transform=transform)

    logger.info(
        "DataLoaders created: train=%d, val=%d, test=%d samples",
        len(train_data), len(val_data), len(test_data),
    )
    return train_loader, val_loader, test_loader
