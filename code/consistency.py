import numpy as np

# ─────────────────────────────────────────────
# Vectorized AICC & TCC Detection
# ─────────────────────────────────────────────
def detect_adversarial(model, X, window_size=3, n_samples=5, noise_std=0.01, aicc_thresh=0.7, final_thresh=0.75, **kwargs):
    """
    Fully vectorized Consistency Detector (AICC + TCC).
    Processes all samples in batched matrix operations for ultra-fast speed.
    """
    N = len(X)
    if N == 0:
        return np.array([])

    # ── 1. Vectorized AICC ──
    # Create (N, n_samples, ...) noisy variants
    X_expanded = np.repeat(X[:, np.newaxis, ...], n_samples, axis=1)
    noises = np.random.normal(0, noise_std, size=X_expanded.shape)
    X_noisy = (X_expanded + noises).reshape(N * n_samples, *X.shape[1:])

    preds = model.predict(X_noisy, batch_size=512, verbose=0)
    pred_classes = np.argmax(preds, axis=1).reshape(N, n_samples)

    # Calculate majority fraction per sample
    aicc_scores = np.zeros(N, dtype=np.float32)
    for i in range(N):
        counts = np.bincount(pred_classes[i])
        aicc_scores[i] = counts.max() / n_samples

    # ── 2. Vectorized Base Predictions for TCC ──
    base_preds = model.predict(X, batch_size=512, verbose=0)
    base_classes = np.argmax(base_preds, axis=1)

    tcc_scores = np.zeros(N, dtype=np.float32)
    for i in range(N):
        start = max(0, i - window_size + 1)
        window = base_classes[start:i+1]
        counts = np.bincount(window)
        tcc_scores[i] = counts.max() / len(window)

    # ── 3. Combined Score & Decision ──
    c_final = 0.5 * aicc_scores + 0.5 * tcc_scores
    adv_flags = (c_final < final_thresh).astype(int)

    return adv_flags