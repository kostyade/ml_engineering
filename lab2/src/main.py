"""Main pipeline: single-config and multi-config experiment runner for dataset extension."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.data_loader import create_data_loaders
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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    logger.info("Configuration loaded from %s", config_path)
    return config


def run_single_experiment(config: Dict[str, Any], config_name: str) -> Dict[str, float]:
    """
    Run a single training + evaluation experiment with the given config.

    Args:
    - config: Parsed YAML configuration dictionary.
    - config_name: Name identifier for this experiment.

    Returns:
    - Dict with evaluation metrics.
    """
    logger.info("=" * 60)
    logger.info("Starting experiment: %s", config_name)
    logger.info("Train batches: %s, Val batches: %s", config["data"]["train_batches"], config["data"]["val_batches"])

    artifacts_dir = Path(config["artifacts"]["save_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create data loaders from selected batches
    train_loader, val_loader, test_loader = create_data_loaders(config)

    # Step 2: Define model, loss, optimizer (fresh for each experiment)
    model = SimpleCNN(n_classes=config["model"]["n_classes"])
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    logger.info("Model: %s, Device: %s", model.__class__.__name__, device)

    # Step 3: Train
    save_path = artifacts_dir / config["artifacts"]["best_model_name"]
    best_model_path = train_model(
        model, train_loader, val_loader, loss_function, optimizer,
        num_epochs=config["training"]["num_epochs"],
        device=device,
        save_path=save_path,
    )

    # Step 4: Load best model & evaluate on static test set
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    metrics = test_model(model, test_loader, loss_function, device)

    # Save metrics
    metrics_path = artifacts_dir / "metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f)
    logger.info("Metrics saved to %s", metrics_path)

    logger.info("Experiment '%s' complete.", config_name)
    return metrics


def run_all_experiments(config_dir: str = "config") -> None:
    """
    Run experiments for all config files in the given directory and produce a comparison.

    Args:
    - config_dir: Path to directory containing YAML config files.
    """
    config_paths = sorted(Path(config_dir).glob("config_*.yaml"))
    if not config_paths:
        logger.error("No config files found in %s", config_dir)
        return

    logger.info("Found %d config files: %s", len(config_paths), [p.name for p in config_paths])

    all_results: Dict[str, Dict[str, float]] = {}

    for config_path in config_paths:
        config_name = config_path.stem  # e.g. "config_1batch"
        config = load_config(str(config_path))
        metrics = run_single_experiment(config, config_name)
        all_results[config_name] = metrics

    # Print comparison table
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPARISON")
    logger.info("=" * 60)
    header = f"{'Config':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Loss':>10}"
    logger.info(header)
    logger.info("-" * 70)
    for name, m in all_results.items():
        row = (
            f"{name:<20} "
            f"{m['accuracy']:>10.4f} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1_score']:>10.4f} "
            f"{m['test_loss']:>10.4f}"
        )
        logger.info(row)

    # Save comparison
    comparison_path = Path("artifacts") / "comparison.yaml"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comparison_path, "w") as f:
        yaml.dump(all_results, f)
    logger.info("Comparison saved to %s", comparison_path)


def main() -> None:
    """Entry point: run all experiments."""
    run_all_experiments()


if __name__ == "__main__":
    main()
