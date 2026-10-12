"""
consistency.py
PyTorch implementation of:
  - AICC (Adaptive Input Consistency Check)
  - TCC  (Temporal Consistency Check)
  - Combined AICC + TCC

Evaluates prediction stability under low-amplitude Gaussian jitter (AICC)
and temporal sliding window modal agreement (TCC).
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
    mode: str = "both",
) -> np.ndarray:
    """
    Evaluates prediction stability under input jitter (AICC), temporal sequence (TCC), or both.
    
    Parameters:
      model: PyTorch NIDS_CNN model
      X: (N, n_features) input array
      device: 'cpu' or 'cuda'
      window_size: temporal sliding window size for TCC (e.g. 3 or 5)
      n_samples: number of Gaussian jitter samples for AICC
      noise_std: standard deviation of Gaussian jitter for AICC
      final_thresh: baseline decision threshold for stability score
      adaptive: dynamically adjust threshold based on batch stability mean
      mode: 'both', 'aicc', or 'tcc'
      
    Returns:
      adv_flags: (N,) binary array where 1 = flagged as anomalous / adversarial, 0 = normal
    """
    model.eval()
    N = len(X)
    if N == 0:
        return np.array([], dtype=int)

    # 1. Base predictions (needed for TCC, confidence margin, and baseline logits)
    with torch.no_grad():
        Xt_base = torch.tensor(X.astype(np.float32), dtype=torch.float32).to(device)
        base_logits = model(Xt_base)
        base_probs = F.softmax(base_logits, dim=1).cpu().numpy()
        base_classes = base_probs.argmax(axis=1)

    # 2. Compute AICC (Adaptive Input Consistency Check) if requested
    aicc_scores = np.ones(N, dtype=np.float32)
    if mode in ("aicc", "both"):
        X_expanded = np.repeat(X[:, np.newaxis, ...], n_samples, axis=1)
        noises = np.random.normal(0, noise_std, size=X_expanded.shape).astype(np.float32)
        X_noisy = (X_expanded + noises).reshape(N * n_samples, *X.shape[1:]).astype(np.float32)

        with torch.no_grad():
            Xt = torch.tensor(X_noisy, dtype=torch.float32).to(device)
            logits = model(Xt)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            pred_classes = probs.argmax(axis=1).reshape(N, n_samples)

        for i in range(N):
            counts = np.bincount(pred_classes[i], minlength=probs.shape[1])
            aicc_scores[i] = counts.max() / n_samples

    # 3. Compute TCC (Temporal Consistency Check) if requested
    tcc_scores = np.ones(N, dtype=np.float32)
    if mode in ("tcc", "both"):
        for i in range(N):
            start = max(0, i - window_size + 1)
            window = base_classes[start:i+1]
            counts = np.bincount(window, minlength=base_probs.shape[1])
            tcc_scores[i] = counts.max() / len(window)

    # 4. Combine based on selected mode
    if mode == "aicc":
        c_final = aicc_scores
    elif mode == "tcc":
        c_final = tcc_scores
    else:  # "both"
        c_final = 0.5 * aicc_scores + 0.5 * tcc_scores

    # 5. Threshold calibration
    if adaptive:
        mean_stab = float(np.mean(c_final))
        eff_thresh = max(0.55, min(final_thresh, mean_stab - 0.05))
    else:
        eff_thresh = final_thresh

    # 6. Flag decision: low stability score OR boundary margin anomaly
    if base_probs.shape[1] >= 2:
        margin = np.abs(base_probs[:, 1] - base_probs[:, 0])
        # A sample is flagged if its stability score drops below threshold,
        # or if it has low confidence margin AND reduced stability.
        adv_flags = ((c_final < eff_thresh) | ((margin < 0.15) & (c_final < 0.80))).astype(int)
    else:
        adv_flags = (c_final < eff_thresh).astype(int)

    return adv_flags
