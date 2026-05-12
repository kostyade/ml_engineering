# Lab 6: Streamlit Model Analysis Dashboard

Interactive Streamlit dashboard for CIFAR-10 classification with MLflow integration and Grad-CAM explainability.

## Prerequisites

Lab 4 must already have been executed at least once (so `lab4/mlruns/` and `lab4/data/cifar-10-batches-py/` exist with trained models).

## Setup

```bash
cd lab6
poetry env use python3.11
poetry install
```

## Run

```bash
poetry run streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Architecture

```
app.py                    # Streamlit entry, sidebar + 3 tabs
config.yaml               # MLflow URI, data paths, model config
src/
├── config.py             # Config loader
├── model.py              # SimpleCNN (shared with Labs 4/5)
├── data.py               # CIFAR-10 loading, splits, normalization, class names
├── mlflow_utils.py       # Experiment/run discovery + artifact download
├── inference.py          # Batch inference helper
├── viz.py                # Plotly figures + heatmap overlay
├── gradcam.py            # Grad-CAM implementation (hooks-based)
└── tabs/
    ├── dataset_tab.py    # Tab 1: Dataset Exploration
    ├── errors_tab.py     # Tab 2: Model Error Analysis (MLflow)
    └── predict_tab.py    # Tab 3: Prediction + Grad-CAM
```

## Tabs

1. **Dataset Exploration** — Dataset stats, class distributions, sample browser with class filtering.
2. **Error Analysis** — Pick an MLflow run, view its test-set confusion matrix, per-class error counts, and inspect misclassified examples (sortable by confidence).
3. **Prediction & Explainability** — Pick an MLflow run, select a test sample or upload an image, see the predicted class probabilities and a Grad-CAM heatmap highlighting which input regions drove the prediction. Supports explaining non-top classes.

## Reports

See [REPORT.md](REPORT.md) for a full discussion of the architecture, findings, and engineering reflection.
