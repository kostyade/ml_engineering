"""W&B experiment runner: runs multiple configs as separate W&B runs grouped by config."""

import logging
from pathlib import Path

import wandb
import yaml

from src.evaluate import run_evaluation
from src.train import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_NAME = "cifar10-classification"
ENTITY = "kostiantyn-dehtiarenko-kharkiv-polytechnic-institute"


def main() -> None:
    config_dir = Path("config")
    config_files = sorted(config_dir.glob("config_*.yaml"))
    logger.info("Found %d config files: %s", len(config_files), [f.name for f in config_files])

    all_results = []

    for config_file in config_files:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        experiment_label = config_file.stem  # e.g. "config_1batch"

        logger.info("=" * 60)
        logger.info("Starting experiment: %s", experiment_label)

        # --- Stage 1: Training (one W&B run, grouped by config) ---
        wandb.init(
            entity=ENTITY,
            project=PROJECT_NAME,
            name=f"{experiment_label} - training",
            group=experiment_label,
            job_type="train",
            config=config,
            tags=[experiment_label, "train"],
            reinit=True,
        )
        # Also log the config file itself as an artifact
        cfg_artifact = wandb.Artifact(name=f"config-{experiment_label}", type="config")
        cfg_artifact.add_file(str(config_file))
        wandb.log_artifact(cfg_artifact)

        logger.info("Stage 1: Training")
        run_training(config)
        wandb.finish()

        # --- Stage 2: Evaluation (separate W&B run, same group) ---
        wandb.init(
            entity=ENTITY,
            project=PROJECT_NAME,
            name=f"{experiment_label} - evaluation",
            group=experiment_label,
            job_type="evaluate",
            config=config,
            tags=[experiment_label, "evaluate"],
            reinit=True,
        )
        logger.info("Stage 2: Evaluation")
        metrics = run_evaluation(config)
        wandb.finish()

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

    logger.info("All experiments logged to W&B project: %s", PROJECT_NAME)


if __name__ == "__main__":
    main()
