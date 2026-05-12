"""Grad-CAM implementation for SimpleCNN."""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GradCAM:
    """Compute Grad-CAM activation maps for a target layer in a CNN."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activations)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inp, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out) -> None:
        self.gradients = grad_out[0].detach()

    def __call__(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> tuple[np.ndarray, int]:
        """Compute Grad-CAM heatmap (32x32, [0, 1]) for the input image.

        Returns (heatmap, target_class_used).
        """
        self.model.zero_grad()
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[0, target_class]
        score.backward()

        assert self.activations is not None and self.gradients is not None
        # activations: (1, C, h, w), gradients: (1, C, h, w)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam, target_class

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def get_target_layer(model: nn.Module, layer_name: str) -> nn.Module:
    """Resolve a dotted layer name like 'features.6' to the actual nn.Module."""
    module: nn.Module = model
    for part in layer_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module
