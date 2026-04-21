"""Train the model on selected data batches with MLflow tracking."""

import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import SimpleCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    save_path: Path,
) -> Path:
    """Train the model, log per-epoch metrics to MLflow, save best checkpoint and loss plot."""
    model.to(device)
    best_val_loss: float = float("inf")
    history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        running_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            outputs = model(batch_inputs)
            loss = loss_function(outputs, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model(val_inputs)
                val_loss += loss_function(val_outputs, val_targets).item()

        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)

        # Log per-epoch metrics to MLflow
        mlflow.log_metrics(
            {"train_loss": avg_train_loss, "val_loss": avg_val_loss},
            step=epoch + 1,
        )

        logger.info(
            "Epoch %d/%d — Train Loss: %.4f, Val Loss: %.4f",
            epoch + 1, num_epochs, avg_train_loss, avg_val_loss,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            logger.info("Best model saved (val_loss=%.4f)", best_val_loss)

    # Save loss plot
    plot_path = save_path.parent / "loss_plot.png"
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.savefig(plot_path)
    plt.close()
    logger.info("Loss plot saved to %s", plot_path)

    # Log artifacts to MLflow
    mlflow.log_artifact(str(save_path))
    mlflow.log_artifact(str(plot_path))
    mlflow.log_metric("best_val_loss", best_val_loss)

    logger.info("Training complete.")
    return save_path


def run_training(config: Dict[str, Any], run_name: str) -> None:
    """Run training for a single config inside an MLflow run."""
    artifacts_dir = Path(config["artifacts"]["save_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Train batches: %s, Val batches: %s",
                config["data"]["train_batches"], config["data"]["val_batches"])

    from src.data_loader import create_data_loaders
    train_loader, val_loader, _ = create_data_loaders(config)

    model = SimpleCNN(n_classes=config["model"]["n_classes"])
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    logger.info("Model: %s, Device: %s", model.__class__.__name__, device)

    save_path = artifacts_dir / config["artifacts"]["best_model_name"]
    train_model(
        model, train_loader, val_loader, loss_function, optimizer,
        num_epochs=config["training"]["num_epochs"],
        save_path=save_path,
    )
