"""Model evaluation with accuracy, precision, recall, F1 score."""

import logging
from typing import Dict

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model and return all metrics."""
    model.to(device)
    model.eval()

    test_loss = 0.0
    all_preds: list = []
    all_targets: list = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += loss_function(outputs, targets).item()

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    metrics = {
        "test_loss": test_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    logger.info("=== Test Results ===")
    for name, value in metrics.items():
        logger.info("  %s: %.4f", name, value)

    return metrics
