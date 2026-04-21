"""MLflow experiment runner: runs multiple configs as separate MLflow runs."""

import logging
import os
from pathlib import Path

import mlflow
import yaml

from src.evaluate import run_evaluation
from src.train import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "cifar10-classification"


def flatten_params(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict for MLflow log_params (which only supports flat keys)."""
    items: dict = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_params(v, key))
        else:
            items[key] = v
    return items


def main() -> None:
    config_dir = Path("config")
    config_files = sorted(config_dir.glob("config_*.yaml"))
    logger.info("Found %d config files: %s", len(config_files), [f.name for f in config_files])

    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow experiment: %s", EXPERIMENT_NAME)

    all_results = []

    for config_file in config_files:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        experiment_label = config_file.stem  # e.g. "config_1batch"

        logger.info("=" * 60)
        logger.info("Starting experiment: %s", experiment_label)

        # Each config gets its own MLflow run with hierarchical naming
        with mlflow.start_run(run_name=experiment_label):
            # Log all params
            flat_params = flatten_params(config)
            # Convert lists to strings for MLflow
            for k, v in flat_params.items():
                if isinstance(v, list):
                    flat_params[k] = str(v)
            mlflow.log_params(flat_params)

            # Log the config file itself as artifact
            mlflow.log_artifact(str(config_file))

            # Stage 1: Train (nested child run)
            with mlflow.start_run(
                run_name=f"{experiment_label} - Stage 1: Training",
                nested=True,
            ):
                logger.info("Stage 1: Training")
                mlflow.log_params(flat_params)
                run_training(config, run_name=experiment_label)

            # Stage 2: Evaluate (nested child run)
            with mlflow.start_run(
                run_name=f"{experiment_label} - Stage 2: Evaluation",
                nested=True,
            ):
                logger.info("Stage 2: Evaluation")
                mlflow.log_params(flat_params)
                metrics = run_evaluation(config)

            all_results.append((experiment_label, metrics))

            logger.info("Experiment '%s' complete.", experiment_label)

    # Print comparison table
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPARISON")
    logger.info("=" * 60)
    logger.info(
        "%-22s %9s %9s %9s %9s %9s",
        "Config", "Accuracy", "Precision", "Recall", "F1", "Loss",
    )
    logger.info("-" * 70)
    for label, m in all_results:
        logger.info(
            "%-22s %9.4f %9.4f %9.4f %9.4f %9.4f",
            label, m["accuracy"], m["precision"], m["recall"], m["f1_score"], m["test_loss"],
        )

    logger.info("All experiments logged to MLflow. Run 'poetry run mlflow ui' to view.")


if __name__ == "__main__":
    main()
