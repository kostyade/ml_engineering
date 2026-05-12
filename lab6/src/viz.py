"""Visualization helpers — Plotly figures and matplotlib utilities."""

import logging
from typing import List

import matplotlib.cm as cm
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def class_distribution_fig(labels: np.ndarray, class_names: List[str], title: str) -> go.Figure:
    """Bar chart of class counts."""
    counts = np.bincount(labels, minlength=len(class_names))
    fig = px.bar(
        x=class_names,
        y=counts,
        labels={"x": "Class", "y": "Count"},
        title=title,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def confusion_matrix_fig(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
) -> go.Figure:
    """Plotly heatmap of the confusion matrix."""
    cm_arr = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig = px.imshow(
        cm_arr,
        x=class_names,
        y=class_names,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "True", "color": "Count"},
        aspect="equal",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def per_class_error_fig(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
) -> go.Figure:
    """Bar chart of misclassification count per true class."""
    errors = np.zeros(len(class_names), dtype=int)
    for true_label in range(len(class_names)):
        mask = y_true == true_label
        errors[true_label] = int(((y_pred[mask] != true_label)).sum())
    fig = px.bar(
        x=class_names,
        y=errors,
        labels={"x": "True class", "y": "# misclassified"},
        title="Misclassifications per true class",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def probability_bar_fig(probs: np.ndarray, class_names: List[str]) -> go.Figure:
    """Horizontal bar chart of class probabilities for one sample."""
    fig = px.bar(
        x=probs,
        y=class_names,
        orientation="h",
        labels={"x": "Probability", "y": "Class"},
        title="Prediction probabilities",
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1]),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis={"categoryorder": "array", "categoryarray": class_names[::-1]},
    )
    return fig


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a heatmap [0, 1] onto an RGB image [0, 1] using jet colormap."""
    colored = cm.jet(heatmap)[..., :3]
    overlaid = (1 - alpha) * image + alpha * colored
    return np.clip(overlaid, 0, 1)
