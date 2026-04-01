"""DVC Stage 1: Download CIFAR-10 dataset."""

import logging

import yaml
from torchvision.datasets import CIFAR10

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Download CIFAR-10 dataset to the configured directory."""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    save_dir = params["data"]["save_dir"]

    logger.info("Starting CIFAR-10 download to %s", save_dir)
    CIFAR10(root=save_dir, train=True, download=True)
    logger.info("CIFAR-10 training data downloaded.")

    CIFAR10(root=save_dir, train=False, download=True)
    logger.info("CIFAR-10 test data downloaded.")

    logger.info("Download stage complete.")


if __name__ == "__main__":
    main()
