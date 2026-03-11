"""Training loop with validation and artifact saving."""

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
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
