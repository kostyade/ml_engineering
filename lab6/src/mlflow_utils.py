"""MLflow query helpers — list experiments, runs, and download model artifacts."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd

logger = logging.getLogger(__name__)


def set_tracking_uri(uri: str) -> None:
    mlflow.set_tracking_uri(uri)
    logger.info("MLflow tracking URI set to %s", uri)


def list_experiments() -> List[Dict[str, Any]]:
    """Return all MLflow experiments with id, name, and lifecycle stage."""
    experiments = mlflow.search_experiments()
    return [
        {"experiment_id": e.experiment_id, "name": e.name, "lifecycle_stage": e.lifecycle_stage}
        for e in experiments
    ]


def list_runs_with_model(experiment_name: str) -> pd.DataFrame:
    """Return runs that have a `best_model.pth` artifact (i.e. training runs)."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.warning("Experiment '%s' not found", experiment_name)
        return pd.DataFrame()

    all_runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if all_runs.empty:
        logger.warning("No runs in experiment '%s'", experiment_name)
        return all_runs

    # Filter to runs that have a model artifact
    keep = []
    for _, row in all_runs.iterrows():
        artifact_uri = row["artifact_uri"]
        artifact_path = Path(artifact_uri.replace("file://", "")) / "best_model.pth"
        if artifact_path.exists():
            keep.append(row)
    if not keep:
        return pd.DataFrame()
    return pd.DataFrame(keep)


def download_model_artifact(run_id: str, artifact_name: str = "best_model.pth") -> Path:
    """Download the model artifact from a run and return the local path."""
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_name)
    logger.info("Downloaded artifact %s from run %s to %s", artifact_name, run_id, local_path)
    return Path(local_path)


def get_run_metrics(run_id: str) -> Dict[str, float]:
    """Return the final logged metrics for a run."""
    run = mlflow.get_run(run_id)
    return dict(run.data.metrics)


def get_run_params(run_id: str) -> Dict[str, str]:
    run = mlflow.get_run(run_id)
    return dict(run.data.params)


def get_run_display_name(run_row: pd.Series) -> str:
    """Build a human-readable label for a run row from search_runs()."""
    name = run_row.get("tags.mlflow.runName") or run_row["run_id"][:8]
    return f"{name} ({run_row['run_id'][:8]})"


def find_evaluation_run_for_training(
    experiment_name: str, training_run_id: str
) -> Optional[str]:
    """Given a training run id, locate the sibling evaluation run id (same parent)."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    training_run = mlflow.get_run(training_run_id)
    parent_id = training_run.data.tags.get("mlflow.parentRunId")
    if parent_id is None:
        return None
    siblings = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
    )
    for _, sib in siblings.iterrows():
        name = sib.get("tags.mlflow.runName", "")
        if "Evaluation" in name and sib["run_id"] != training_run_id:
            return sib["run_id"]
    return None
