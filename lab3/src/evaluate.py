"""DVC Stage 3: Evaluate the trained model on the static test set."""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data_loader import create_data_loaders
from src.model import SimpleCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    """Run the evaluation stage."""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    artifacts_dir = Path(params["artifacts"]["save_dir"])
    model_path = artifacts_dir / params["artifacts"]["best_model_name"]

    logger.info("Starting evaluation stage.")
    logger.info("Loading model from %s", model_path)

    # Load model
    model = SimpleCNN(n_classes=params["model"]["n_classes"])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    # Load test data
    _, _, test_loader = create_data_loaders(params)

    # Evaluate
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

    # Save as JSON for DVC metrics
    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    logger.info("Evaluation stage complete.")


if __name__ == "__main__":
    main()
