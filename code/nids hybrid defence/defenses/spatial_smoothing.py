"""
spatial_smoothing.py
Spatial and Temporal Smoothing Defenses for NIDS.

Enhancements:
  - Temporal Stream Smoothing (TSS): Filters each feature along the time axis (axis 0)
    across consecutive packets (t-1, t, t+1) without mixing independent tabular features.
  - Gated Temporal Smoothing (Adaptive): Preserves confident clean flows in their crisp
    representation and activates temporal median filtering on boundary/perturbed flows.
  - Legacy Feature Smoothing: Retained for baseline replication and ablation comparisons.
"""

import numpy as np
from scipy.ndimage import median_filter
import torch


def spatial_smoothing(
    X: np.ndarray,
    window_size: int = 3,
    mode: str = "temporal",
    model=None,
    device: str = "cpu",
    margin_thresh: float = 0.20,
) -> np.ndarray:
    """
    Apply smoothing to input flows.

    Parameters:
      X: (n_samples, n_features) input matrix
      window_size: median filter window (default: 3)
      mode:
        - "temporal": Denoises each feature over time (axis 0). Each feature is smoothed
          only with its own past/future values. Tabular feature semantics are preserved.
        - "gated_temporal": Evaluates model confidence margin; passes high-confidence flows
          untouched and applies temporal median filtering only on ambiguous/perturbed flows.
        - "feature": Legacy baseline that filters across columns (axis 1).
    """
    if X.ndim != 2:
        raise ValueError("Expected X of shape (n_samples, n_features)")

    if mode == "temporal":
        # Smooth along axis 0 (time) with window_size, keeping features independent (size 1 on axis 1)
        return median_filter(X, size=(window_size, 1), mode="nearest")

    elif mode == "gated_temporal" and model is not None:
        model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32).to(device)
            probs = torch.softmax(model(Xt), dim=1).cpu().numpy()
        margin = np.abs(probs[:, 1] - probs[:, 0])

        smoothed = median_filter(X, size=(window_size, 1), mode="nearest")
        # Where margin is low (perturbed / boundary), use smoothed; otherwise keep crisp raw input
        mask = (margin < margin_thresh)[:, np.newaxis]
        return np.where(mask, smoothed, X)

    elif mode == "feature":
        # Legacy mode: filters across feature dimension
        return np.array([
            median_filter(row, size=window_size, mode="nearest") for row in X
        ])

    else:
        # Default fallback to temporal smoothing
        return median_filter(X, size=(window_size, 1), mode="nearest")


class SpatialSmoothingDefense:
    """Wraps a trained model so predictions run smoothing first (inference-only)."""

    def __init__(self, model, window_size: int = 3, mode: str = "temporal", device: str = "cpu"):
        self.model = model
        self.window_size = window_size
        self.mode = mode
        self.device = device

    def predict(self, X: np.ndarray):
        X_smooth = spatial_smoothing(
            X, window_size=self.window_size, mode=self.mode, model=self.model, device=self.device
        )
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X_smooth, dtype=torch.float32).to(self.device)
            logits = self.model(Xt)
            return logits.argmax(1).cpu().numpy()
