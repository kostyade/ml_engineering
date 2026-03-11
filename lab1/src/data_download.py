"""Download CIFAR-10 dataset using torchvision."""

import logging

from torchvision import transforms
from torchvision.datasets import CIFAR10

logger = logging.getLogger(__name__)


def download_and_extract(save_dir: str) -> tuple[CIFAR10, CIFAR10]:
    """Download CIFAR-10 and return train/test datasets with transforms."""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = CIFAR10(root=save_dir, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=save_dir, train=False, download=True, transform=transform_test)

    logger.info("CIFAR-10 downloaded: %d train, %d test samples", len(train_dataset), len(test_dataset))
    return train_dataset, test_dataset
