"""Streamlit entry point for the CIFAR-10 Model Analysis Dashboard."""

import logging

import numpy as np
import streamlit as st
import torch

from src.config import load_config
from src.tabs import dataset_tab, errors_tab, predict_tab

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Deterministic UI
torch.manual_seed(0)
np.random.seed(0)


def main() -> None:
    config = load_config()
    st.set_page_config(
        page_title=config["ui"]["page_title"],
        layout="wide",
    )
    st.title(config["ui"]["page_title"])

    with st.sidebar:
        st.header("Configuration")
        st.text_input(
            "MLflow tracking URI",
            value=config["mlflow"]["tracking_uri"],
            disabled=True,
        )
        st.text_input(
            "Experiment name",
            value=config["mlflow"]["experiment_name"],
            disabled=True,
        )
        st.caption("Edit `config.yaml` and restart the app to change these.")

    tab1, tab2, tab3 = st.tabs(
        ["Dataset Exploration", "Error Analysis", "Prediction & Explainability"]
    )

    with tab1:
        dataset_tab.render(config)
    with tab2:
        errors_tab.render(config)
    with tab3:
        predict_tab.render(config)


if __name__ == "__main__":
    main()
