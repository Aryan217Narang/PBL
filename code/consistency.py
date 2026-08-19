"""
consistency.py
Adaptive Input Consistency Check (AICC) + Temporal Consistency Check (TCC)
Vectorized anomaly detector for adversarial network intrusions.
"""

import numpy as np


def detect_adversarial(
    model,
    X: np.ndarray,
    window_size: int = 3,
    n_samples: int = 5,
    noise_std: float = 0.015,
    aicc_thresh: float = 0.70,
    final_thresh: float = 0.70,
    **kwargs
) -> np.ndarray:
    """
    Vectorized Consistency Detector (AICC + TCC).
    Evaluates prediction stability under low-amplitude Gaussian noise and temporal sliding windows.
    Adversarial perturbations exhibit high output volatility near decision boundaries.
    """
    N = len(X)
    if N == 0:
        return np.array([])

    # ── 1. Vectorized AICC ──────────────────────────────────────
    X_expanded = np.repeat(X[:, np.newaxis, ...], n_samples, axis=1)
    noises = np.random.normal(0, noise_std, size=X_expanded.shape)
    X_noisy = (X_expanded + noises).reshape(N * n_samples, *X.shape[1:])

    preds = model.predict(X_noisy, batch_size=512, verbose=0)
    pred_classes = np.argmax(preds, axis=1).reshape(N, n_samples)

    # Compute agreement ratio across noisy perturbations
    aicc_scores = np.zeros(N, dtype=np.float32)
    for i in range(N):
        counts = np.bincount(pred_classes[i], minlength=preds.shape[1])
        aicc_scores[i] = counts.max() / n_samples

    # ── 2. Vectorized Base Predictions for TCC ──────────────────
    base_preds = model.predict(X, batch_size=512, verbose=0)
    base_classes = np.argmax(base_preds, axis=1)

    tcc_scores = np.zeros(N, dtype=np.float32)
    for i in range(N):
        start = max(0, i - window_size + 1)
        window = base_classes[start:i+1]
        counts = np.bincount(window, minlength=base_preds.shape[1])
        tcc_scores[i] = counts.max() / len(window)

    # ── 3. Combined Score & Decision ────────────────────────────
    c_final = 0.5 * aicc_scores + 0.5 * tcc_scores
    adv_flags = (c_final < final_thresh).astype(int)

    return adv_flags