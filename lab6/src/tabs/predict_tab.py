"""Tab 3: Model Prediction & Explainability (Grad-CAM)."""

import io
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.data import (
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    denormalize_image,
    load_all_train_data,
    normalize_batch,
    raw_image_to_display,
    split_test_set,
)
from src.gradcam import GradCAM, get_target_layer
from src.inference import device, load_model_state
from src.mlflow_utils import (
    download_model_artifact,
    get_run_display_name,
    list_runs_with_model,
    set_tracking_uri,
)
from src.model import SimpleCNN
from src.viz import overlay_heatmap, probability_bar_fig

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


def _preprocess_uploaded(image: Image.Image) -> np.ndarray:
    """Resize PIL image to 32x32, return as (3, 32, 32) uint8-style float array."""
    image = image.convert("RGB").resize((32, 32), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
    return arr  # values in [0, 255]


def _predict_single(model: torch.nn.Module, raw_image: np.ndarray):
    """Run inference on a single (3, 32, 32) raw [0, 255] image."""
    tensor = normalize_batch(raw_image[None, ...]).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return tensor, probs


def render(config: Dict[str, Any]) -> None:
    st.header("Prediction & Explainability")
    st.write(
        "Run inference on a test sample or uploaded image, then visualize a Grad-CAM "
        "heatmap showing which input regions drove the prediction."
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
        st.warning(f"No runs with a model artifact in experiment `{experiment_name}`.")
        return

    # --- Selectors ---
    run_options = {get_run_display_name(row): row["run_id"] for _, row in runs_df.iterrows()}
    chosen_label = st.selectbox("MLflow run", list(run_options.keys()), key="predict_run")
    run_id = run_options[chosen_label]

    source_choice = st.radio(
        "Image source",
        ["Test set sample", "Upload an image"],
        horizontal=True,
        key="predict_source",
    )

    try:
        model = _cached_load_model(run_id, config["model"]["n_classes"])
    except Exception as e:
        logger.exception("Model load failed")
        st.error(f"Could not load model from run {run_id}: {e}")
        return

    raw_image: np.ndarray
    true_label_text: str

    if source_choice == "Test set sample":
        try:
            test_data, test_labels = _cached_test_split(
                config["data"]["cifar_dir"],
                config["data"]["test_size"],
                config["data"]["random_state"],
            )
        except FileNotFoundError as e:
            st.error(f"CIFAR-10 data missing: {e}")
            return

        class_filter = st.selectbox(
            "Filter by class", ["(all classes)"] + CIFAR10_CLASSES, key="predict_class_filter",
        )
        if class_filter == "(all classes)":
            candidate_indices = np.arange(len(test_labels))
        else:
            target_label = CIFAR10_CLASSES.index(class_filter)
            candidate_indices = np.where(test_labels == target_label)[0]

        if len(candidate_indices) == 0:
            st.warning("No samples match the current filter.")
            return

        sample_pos = st.slider(
            "Sample position", 0, len(candidate_indices) - 1, 0, key="predict_sample_pos",
        )
        sample_idx = int(candidate_indices[sample_pos])
        raw_image = test_data[sample_idx]
        true_label_text = CIFAR10_CLASSES[int(test_labels[sample_idx])]
    else:
        uploaded = st.file_uploader(
            "Upload an image (any size — will be resized to 32x32)",
            type=["png", "jpg", "jpeg", "bmp"],
        )
        if uploaded is None:
            st.info("Upload an image to run inference.")
            return
        try:
            pil_image = Image.open(io.BytesIO(uploaded.read()))
        except Exception as e:
            st.error(f"Could not open uploaded file as image: {e}")
            return
        raw_image = _preprocess_uploaded(pil_image)
        true_label_text = "(uploaded — unknown)"

    # --- Inference ---
    try:
        tensor, probs = _predict_single(model, raw_image)
    except Exception as e:
        logger.exception("Inference failed")
        st.error(f"Inference failed: {e}")
        return

    pred_idx = int(probs.argmax())
    pred_label = CIFAR10_CLASSES[pred_idx]

    explain_class_label = st.selectbox(
        "Explain class",
        ["(top prediction)"] + CIFAR10_CLASSES,
        help="Pick a different class to see what evidence supports it.",
        key="predict_explain_class",
    )
    explain_idx = (
        pred_idx
        if explain_class_label == "(top prediction)"
        else CIFAR10_CLASSES.index(explain_class_label)
    )

    # --- Grad-CAM ---
    target_layer = get_target_layer(model, config["model"]["last_conv_layer"])
    cam = GradCAM(model, target_layer)
    try:
        heatmap, used_class = cam(tensor.clone().requires_grad_(True), target_class=explain_idx)
    finally:
        cam.close()

    display_image = raw_image_to_display(raw_image)
    overlaid = overlay_heatmap(display_image, heatmap, alpha=0.5)

    # --- Display ---
    st.markdown(
        f"**True label:** `{true_label_text}` &nbsp;&nbsp; **Predicted:** `{pred_label}` "
        f"({probs[pred_idx]:.2%}) &nbsp;&nbsp; **Explained class:** `{CIFAR10_CLASSES[used_class]}`"
    )

    img_col, heat_col, overlay_col = st.columns(3)
    img_col.image(display_image, caption="Input image", width=200)
    heat_col.image(heatmap, caption="Grad-CAM heatmap", width=200, clamp=True)
    overlay_col.image(overlaid, caption="Overlay", width=200)

    st.plotly_chart(probability_bar_fig(probs, CIFAR10_CLASSES), use_container_width=True)
