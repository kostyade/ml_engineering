"""Batch inference helpers."""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_inference(
    model: nn.Module, data_tensor: torch.Tensor, batch_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a batch of preprocessed images.

    Returns (preds, probs) where:
      preds: (N,) int64 array of predicted class indices
      probs: (N, n_classes) float32 softmax probabilities
    """
    model.eval()
    all_preds, all_probs = [], []
    for i in range(0, len(data_tensor), batch_size):
        batch = data_tensor[i : i + batch_size].to(device)
        logits = model(batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_preds.append(preds)
        all_probs.append(probs)
    preds_arr = np.concatenate(all_preds)
    probs_arr = np.concatenate(all_probs)
    logger.info("Inference complete on %d samples", len(preds_arr))
    return preds_arr, probs_arr


def load_model_state(model: nn.Module, ckpt_path) -> nn.Module:
    """Load weights from a .pth checkpoint into a model instance."""
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
