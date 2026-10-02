"""
consistency.py
PyTorch implementation of Adaptive Input Consistency Check (AICC) + Temporal Consistency Check (TCC).
Evaluates prediction stability under low-amplitude Gaussian jitter and temporal windows.
Intercepts evasive adversarial samples near decision boundaries.
"""

import numpy as np
import torch
import torch.nn.functional as F


def detect_adversarial_pytorch(
    model,
    X: np.ndarray,
    device: str = "cpu",
    window_size: int = 3,
    n_samples: int = 5,
    noise_std: float = 0.02,
    final_thresh: float = 0.65,
    adaptive: bool = True,
) -> np.ndarray:
    """
    Evaluates prediction stability under noise and time.
    Returns adv_flags: 1 = adversarial intrusion detected, 0 = normal flow.
    """
    model.eval()
    N = len(X)
    if N == 0:
        return np.array([])

    # 1. Vectorized AICC
    X_expanded = np.repeat(X[:, np.newaxis, ...], n_samples, axis=1)
    noises = np.random.normal(0, noise_std, size=X_expanded.shape).astype(np.float32)
    X_noisy = (X_expanded + noises).reshape(N * n_samples, *X.shape[1:]).astype(np.float32)

    with torch.no_grad():
        Xt = torch.tensor(X_noisy, dtype=torch.float32).to(device)
        logits = model(Xt)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        pred_classes = probs.argmax(axis=1).reshape(N, n_samples)

    aicc_scores = np.zeros(N, dtype=np.float32)
    for i in range(N):
        counts = np.bincount(pred_classes[i], minlength=probs.shape[1])
        aicc_scores[i] = counts.max() / n_samples

    # 2. Base predictions for TCC & Margin
    with torch.no_grad():
        Xt_base = torch.tensor(X.astype(np.float32), dtype=torch.float32).to(device)
        base_logits = model(Xt_base)
        base_probs = F.softmax(base_logits, dim=1).cpu().numpy()
        base_classes = base_probs.argmax(axis=1)

    tcc_scores = np.zeros(N, dtype=np.float32)
    for i in range(N):
        start = max(0, i - window_size + 1)
        window = base_classes[start:i+1]
        counts = np.bincount(window, minlength=base_probs.shape[1])
        tcc_scores[i] = counts.max() / len(window)

    c_final = 0.5 * aicc_scores + 0.5 * tcc_scores

    if adaptive:
        mean_stab = float(np.mean(c_final))
        eff_thresh = max(0.55, min(final_thresh, mean_stab - 0.05))
    else:
        eff_thresh = final_thresh

    if base_probs.shape[1] >= 2:
        margin = np.abs(base_probs[:, 1] - base_probs[:, 0])
        adv_flags = ((c_final < eff_thresh) | ((margin < 0.15) & (c_final < 0.80))).astype(int)
    else:
        adv_flags = (c_final < eff_thresh).astype(int)

    return adv_flags
