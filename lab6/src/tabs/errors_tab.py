"""Tab 2: Model Error Analysis (MLflow integration)."""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.data import (
    CIFAR10_CLASSES,
    load_all_train_data,
    normalize_batch,
    raw_image_to_display,
    split_test_set,
)
from src.inference import load_model_state, run_inference
from src.mlflow_utils import (
    download_model_artifact,
    get_run_display_name,
    get_run_metrics,
    list_runs_with_model,
    set_tracking_uri,
)
from src.model import SimpleCNN
from src.viz import confusion_matrix_fig, per_class_error_fig

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner="Loading test set...")
def _cached_test_split(cifar_dir: str, test_size: float, random_state: int):
    data, labels = load_all_train_data(Path(cifar_dir))
    split = split_test_set(data, labels, test_size=test_size, random_state=random_state)
    return split["test_data"], split["test_labels"]


@st.cache_resource(show_spinner="Loading model...")
def _cached_load_model(run_id: str, n_classes: int):
    ckpt_path = download_model_artifact(run_id)
    model = SimpleCNN(n_classes=n_classes)
    load_model_state(model, ckpt_path)
    return model


@st.cache_data(show_spinner="Running inference on test set...")
def _cached_inference(run_id: str, cifar_dir: str, test_size: float, random_state: int, n_classes: int):
    test_data, test_labels = _cached_test_split(cifar_dir, test_size, random_state)
    model = _cached_load_model(run_id, n_classes)
    tensor = normalize_batch(test_data)
    preds, probs = run_inference(model, tensor)
    return preds, probs, test_data, test_labels


def render(config: Dict[str, Any]) -> None:
    st.header("Model Error Analysis")
    st.write(
        "Select an MLflow run with a logged model, then inspect its misclassifications "
        "on the static test set."
    )

    set_tracking_uri(config["mlflow"]["tracking_uri"])
    experiment_name = config["mlflow"]["experiment_name"]

    try:
        runs_df = list_runs_with_model(experiment_name)
    except Exception as e:
        logger.exception("Failed to list MLflow runs")
        st.error(f"MLflow query failed: {e}")
        return

    if runs_df.empty:
        st.warning(
            f"No runs with a `best_model.pth` artifact found in experiment "
            f"`{experiment_name}`. Train a model first via Lab 4."
        )
        return

    # --- Run selector ---
    run_options = {get_run_display_name(row): row["run_id"] for _, row in runs_df.iterrows()}
    chosen_label = st.selectbox("MLflow run", list(run_options.keys()), key="errors_run")
    run_id = run_options[chosen_label]

    metrics = get_run_metrics(run_id)
    if metrics:
        m_cols = st.columns(min(5, len(metrics)))
        for col, (k, v) in zip(m_cols, sorted(metrics.items())[:5]):
            col.metric(k, f"{v:.4f}")

    # --- Inference ---
    try:
        preds, probs, test_data, test_labels = _cached_inference(
            run_id,
            config["data"]["cifar_dir"],
            config["data"]["test_size"],
            config["data"]["random_state"],
            config["model"]["n_classes"],
        )
    except FileNotFoundError as e:
        st.error(f"Artifact or data missing: {e}")
        return
    except RuntimeError as e:
        logger.exception("Model load / inference failure")
        st.error(f"Could not load or run the model: {e}")
        return
    except Exception as e:
        logger.exception("Unexpected inference failure")
        st.error(f"Unexpected error: {e}")
        return

    acc = float((preds == test_labels).mean())
    st.success(f"Test accuracy (recomputed): **{acc:.4f}** ({(preds == test_labels).sum()} / {len(test_labels)})")

    # --- Confusion matrix + per-class errors ---
    st.subheader("Confusion matrix")
    cm_col, err_col = st.columns(2)
    with cm_col:
        st.plotly_chart(
            confusion_matrix_fig(test_labels, preds, CIFAR10_CLASSES),
            use_container_width=True,
        )
    with err_col:
        st.plotly_chart(
            per_class_error_fig(test_labels, preds, CIFAR10_CLASSES),
            use_container_width=True,
        )

    # --- Misclassified examples ---
    st.subheader("Misclassified examples")
    mis_mask = preds != test_labels
    n_errors = int(mis_mask.sum())
    if n_errors == 0:
        st.info("No misclassifications on the test set.")
        return

    confidence = probs[np.arange(len(probs)), preds]
    err_df = pd.DataFrame({
        "test_idx": np.where(mis_mask)[0],
        "true": [CIFAR10_CLASSES[i] for i in test_labels[mis_mask]],
        "predicted": [CIFAR10_CLASSES[i] for i in preds[mis_mask]],
        "confidence": confidence[mis_mask],
    })

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    sort_choice = ctrl_col1.selectbox(
        "Sort by",
        ["highest confidence (most overconfident)", "lowest confidence", "predicted class"],
        key="errors_sort",
    )
    max_show = ctrl_col2.slider(
        "Max examples to display", 6, 96, config["ui"]["default_max_errors"], step=6,
        key="errors_max",
    )
    class_filter = ctrl_col3.selectbox(
        "Filter by true class", ["(all)"] + CIFAR10_CLASSES, key="errors_class_filter",
    )

    if class_filter != "(all)":
        err_df = err_df[err_df["true"] == class_filter]

    if sort_choice == "highest confidence (most overconfident)":
        err_df = err_df.sort_values("confidence", ascending=False)
    elif sort_choice == "lowest confidence":
        err_df = err_df.sort_values("confidence", ascending=True)
    else:
        err_df = err_df.sort_values("predicted")

    err_df = err_df.head(max_show)
    st.caption(f"Showing {len(err_df)} of {n_errors} total errors.")

    cols_per_row = 6
    for row_start in range(0, len(err_df), cols_per_row):
        row = err_df.iloc[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (_, e) in zip(cols, row.iterrows()):
            img = raw_image_to_display(test_data[int(e["test_idx"])])
            col.image(img, width=96)
            col.caption(
                f"#{e['test_idx']}\ntrue: **{e['true']}**\npred: **{e['predicted']}** ({e['confidence']:.2f})"
            )
