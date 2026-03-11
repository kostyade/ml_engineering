"""Data ingestion: split CIFAR-10 into train/val/test data loaders."""

import logging
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

logger = logging.getLogger(__name__)


def create_data_loaders(
    train_dataset: CIFAR10,
    test_dataset: CIFAR10,
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split train into train+val, return three DataLoaders."""
    batch_size: int = config["training"]["batch_size"]
    num_workers: int = config["training"]["num_workers"]
    val_size: float = config["data"]["val_size"]
    random_state: int = config["data"]["random_state"]

    n_total = len(train_dataset)
    n_val = int(n_total * val_size)
    n_train = n_total - n_val

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(random_state),
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logger.info("Data loaders: train=%d, val=%d, test=%d", len(train_subset), len(val_subset), len(test_dataset))
    return train_loader, val_loader, test_loader
