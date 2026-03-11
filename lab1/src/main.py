"""Main pipeline: orchestrates download, training, and evaluation."""

import logging
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.data_download import download_and_extract
from src.data_ingestion import create_data_loaders
from src.evaluate import test_model
from src.model import SimpleCNN
from src.train import train_model

# Logging setup
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    logger.info("Configuration loaded from %s", config_path)
    return config


def main() -> None:
    config = load_config()

    artifacts_dir = Path(config["artifacts"]["save_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download data
    train_dataset, test_dataset = download_and_extract(config["data"]["save_dir"])

    # Step 2: Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, test_dataset, config)

    # Step 3: Define model, loss, optimizer
    model = SimpleCNN(n_classes=config["model"]["n_classes"])
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    logger.info("Model: %s", model.__class__.__name__)
    logger.info("Device: %s", device)

    # Step 4: Train
    save_path = artifacts_dir / config["artifacts"]["best_model_name"]
    best_model_path = train_model(
        model, train_loader, val_loader, loss_function, optimizer,
        num_epochs=config["training"]["num_epochs"],
        device=device,
        save_path=save_path,
    )

    # Step 5: Load best model & test
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    metrics = test_model(model, test_loader, loss_function, device)

    # Save metrics
    metrics_path = artifacts_dir / "metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f)
    logger.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
