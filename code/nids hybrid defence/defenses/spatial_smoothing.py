"""
Spatial Smoothing (SS) - the testing-phase defense in Barik & Misra (2025).
Cheap local-neighbor averaging/median filter applied to inputs right before
inference, to denoise adversarial perturbations. Applied only at test time
(never during training), and is lightweight enough for real-time inline use.
"""

import numpy as np
from scipy.ndimage import median_filter


def spatial_smoothing(X: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Apply a 1D median filter across the feature dimension of each sample.
    X: (n_samples, n_features)
    """
    if X.ndim != 2:
        raise ValueError("Expected X of shape (n_samples, n_features)")
    smoothed = np.array([
        median_filter(row, size=window_size, mode="nearest") for row in X
    ])
    return smoothed


class SpatialSmoothingDefense:
    """Wraps a trained model so predictions run smoothing first (inference-only, ~O(n) cost)."""

    def __init__(self, model, window_size: int = 3, device: str = "cpu"):
        self.model = model
        self.window_size = window_size
        self.device = device

    def predict(self, X: np.ndarray):
        import torch
        X_smooth = spatial_smoothing(X, window_size=self.window_size)
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X_smooth, dtype=torch.float32).to(self.device)
            logits = self.model(Xt)
            return logits.argmax(1).cpu().numpy()
