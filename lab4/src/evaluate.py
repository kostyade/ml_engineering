"""Evaluate the trained model on the static test set with MLflow tracking."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import mlflow
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data_loader import create_data_loaders
from src.model import SimpleCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_evaluation(config: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate model and log metrics to MLflow. Returns metrics dict."""
    artifacts_dir = Path(config["artifacts"]["save_dir"])
    model_path = artifacts_dir / config["artifacts"]["best_model_name"]

    logger.info("Loading model from %s", model_path)

    model = SimpleCNN(n_classes=config["model"]["n_classes"])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    _, _, test_loader = create_data_loaders(config)

    loss_function = nn.CrossEntropyLoss()
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
        "test_loss": round(test_loss, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }

    logger.info("=== Test Results ===")
    for name, value in metrics.items():
        logger.info("  %s: %.4f", name, value)

    # Log test metrics to MLflow
    mlflow.log_metrics(metrics)

    # Save as JSON and log as artifact
    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    mlflow.log_artifact(str(metrics_path))
    logger.info("Metrics saved to %s", metrics_path)

    return metrics
