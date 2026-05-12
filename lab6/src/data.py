"""CIFAR-10 data loading, splitting, and image utilities."""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

CIFAR10_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def _load_batch(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one CIFAR-10 pickle batch file."""
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    data = batch[b"data"].reshape(-1, 3, 32, 32).astype(np.float32)
    labels = np.array(batch[b"labels"], dtype=np.int64)
    return data, labels


def load_all_train_data(cifar_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load all 5 CIFAR-10 training batches as a single (50000, 3, 32, 32) array."""
    all_data, all_labels = [], []
    for i in range(1, 6):
        data, labels = _load_batch(cifar_dir / f"data_batch_{i}")
        all_data.append(data)
        all_labels.append(labels)
    full_data = np.concatenate(all_data, axis=0)
    full_labels = np.concatenate(all_labels, axis=0)
    logger.info("Loaded %d training samples from %s", len(full_data), cifar_dir)
    return full_data, full_labels


def split_test_set(
    data: np.ndarray, labels: np.ndarray, test_size: float, random_state: int
) -> Dict[str, np.ndarray]:
    """Split off a deterministic static test set; return train/test arrays."""
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    logger.info(
        "Split: train=%d, test=%d (seed=%d)",
        len(train_data), len(test_data), random_state,
    )
    return {
        "train_data": train_data,
        "train_labels": train_labels,
        "test_data": test_data,
        "test_labels": test_labels,
    }


def normalize_batch(arr: np.ndarray) -> torch.Tensor:
    """Convert raw uint8/float [0, 255] images to normalized torch tensor."""
    x = torch.tensor(arr, dtype=torch.float32) / 255.0
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Reverse CIFAR normalization on a single (3, 32, 32) tensor → (32, 32, 3) numpy in [0, 1]."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.detach().cpu().numpy().copy()
    for i in range(3):
        img[i] = img[i] * CIFAR10_STD[i] + CIFAR10_MEAN[i]
    img = np.clip(img.transpose(1, 2, 0), 0, 1)
    return img


def raw_image_to_display(arr: np.ndarray) -> np.ndarray:
    """Convert a (3, 32, 32) raw [0, 255] image to (32, 32, 3) [0, 1] for display."""
    return np.clip(arr.transpose(1, 2, 0) / 255.0, 0, 1)
