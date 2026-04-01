"""DVC Stage 2: Train the model on selected data batches."""

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from src.data_loader import create_data_loaders
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
    """Train the model, save best checkpoint and loss plot."""
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

    logger.info("Training complete.")
    return save_path


def main() -> None:
    """Run the training stage."""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    artifacts_dir = Path(params["artifacts"]["save_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training stage.")
    logger.info("Train batches: %s, Val batches: %s", params["data"]["train_batches"], params["data"]["val_batches"])

    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(params)

    # Define model, loss, optimizer
    model = SimpleCNN(n_classes=params["model"]["n_classes"])
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["training"]["learning_rate"])

    logger.info("Model: %s, Device: %s", model.__class__.__name__, device)

    # Train
    save_path = artifacts_dir / params["artifacts"]["best_model_name"]
    train_model(
        model, train_loader, val_loader, loss_function, optimizer,
        num_epochs=params["training"]["num_epochs"],
        save_path=save_path,
    )

    logger.info("Training stage complete.")


if __name__ == "__main__":
    main()
