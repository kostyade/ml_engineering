"""Tab 1: Dataset Exploration."""

import logging
from typing import Any, Dict

import numpy as np
import streamlit as st

from src.data import (
    CIFAR10_CLASSES,
    load_all_train_data,
    raw_image_to_display,
    split_test_set,
)
from src.viz import class_distribution_fig

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner="Loading CIFAR-10 dataset...")
def _cached_load_split(cifar_dir: str, test_size: float, random_state: int) -> Dict[str, Any]:
    from pathlib import Path

    data, labels = load_all_train_data(Path(cifar_dir))
    split = split_test_set(data, labels, test_size=test_size, random_state=random_state)
    return split


def render(config: Dict[str, Any]) -> None:
    st.header("Dataset Exploration")
    st.write("Inspect CIFAR-10 dataset statistics, class distributions, and individual samples.")

    cfg_data = config["data"]
    try:
        split = _cached_load_split(
            cfg_data["cifar_dir"],
            cfg_data["test_size"],
            cfg_data["random_state"],
        )
    except FileNotFoundError as e:
        st.error(f"Could not load CIFAR-10 from {cfg_data['cifar_dir']}: {e}")
        return
    except Exception as e:
        logger.exception("Dataset load failed")
        st.error(f"Unexpected error while loading dataset: {e}")
        return

    train_data, train_labels = split["train_data"], split["train_labels"]
    test_data, test_labels = split["test_data"], split["test_labels"]

    # --- Overview ---
    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total samples", len(train_data) + len(test_data))
    col2.metric("Classes", len(CIFAR10_CLASSES))
    col3.metric("Train set", len(train_data))
    col4.metric("Test set", len(test_data))

    # --- Class distribution ---
    st.subheader("Class distribution")
    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        st.plotly_chart(
            class_distribution_fig(train_labels, CIFAR10_CLASSES, "Train set"),
            use_container_width=True,
        )
    with dist_col2:
        st.plotly_chart(
            class_distribution_fig(test_labels, CIFAR10_CLASSES, "Test set"),
            use_container_width=True,
        )

    # --- Sample browser ---
    st.subheader("Sample inspection")
    inspect_col1, inspect_col2, inspect_col3 = st.columns([1, 1, 1])
    split_choice = inspect_col1.selectbox("Split", ["train", "test"], key="ds_split")
    class_filter = inspect_col2.selectbox(
        "Filter by class", ["(all classes)"] + CIFAR10_CLASSES, key="ds_class_filter",
    )

    data = train_data if split_choice == "train" else test_data
    labels = train_labels if split_choice == "train" else test_labels

    if class_filter == "(all classes)":
        indices = np.arange(len(labels))
    else:
        target_label = CIFAR10_CLASSES.index(class_filter)
        indices = np.where(labels == target_label)[0]

    if len(indices) == 0:
        st.warning("No samples match the current filter.")
        return

    sample_pos = inspect_col3.slider(
        "Sample position",
        min_value=0,
        max_value=len(indices) - 1,
        value=0,
        key="ds_sample_pos",
    )
    sample_idx = int(indices[sample_pos])
    img = raw_image_to_display(data[sample_idx])
    true_label = CIFAR10_CLASSES[int(labels[sample_idx])]

    img_col, info_col = st.columns([1, 2])
    with img_col:
        st.image(img, caption=f"Index {sample_idx}", width=192)
    with info_col:
        st.markdown(f"**True label:** `{true_label}`")
        st.markdown(f"**Dataset index:** `{sample_idx}` (within {split_choice} split)")
        st.markdown(f"**Filter:** `{class_filter}`")
        st.markdown(f"**Matches:** `{len(indices)}` of `{len(labels)}`")
