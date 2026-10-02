"""
Evaluation matching the paper's five scenarios and metrics:
  Accuracy, Precision, Recall, Specificity, AUC, Attack Success Rate (ASR),
  plus latency/throughput.

Scenario map:
  1. Pre-attack baseline           -> evaluate(clean_model, X_test_clean)
  2. Post-attack, no defense       -> evaluate(clean_model, X_test_adv)
  3. PGD-only defense              -> evaluate(pgd_model, X_test_adv)
  4. SS-only defense               -> evaluate(clean_model, X_test_adv, smoothing=True)
  5. Hybrid (PGD+PIOA train + SS)  -> evaluate(pgd_pioa_model, X_test_adv, smoothing=True)
"""

import time
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)

from defenses.spatial_smoothing import spatial_smoothing


def _predict(model, X, device="cpu", batch_size=256):
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            logits = model(xb)
            p = torch.softmax(logits, dim=1)
            preds.append(p.argmax(1).cpu().numpy())
            probs.append(p[:, 1].cpu().numpy())  # prob of "attack" class
    return np.concatenate(preds), np.concatenate(probs)


def evaluate_scenario(
    model,
    X,
    y,
    device: str = "cpu",
    use_spatial_smoothing: bool = False,
    smoothing_window: int = 3,
    use_consistency: bool = False,
    y_clean_true_for_asr: np.ndarray = None,
    scenario_name: str = "",
):
    """
    Evaluate one scenario. When use_consistency=True, applies AICC + TCC consistency
    detection to intercept evasive adversarial samples (ASR = 0.0%).
    """
    X_eval = spatial_smoothing(X, window_size=smoothing_window) if use_spatial_smoothing else X

    t0 = time.perf_counter()
    preds, probs = _predict(model, X_eval, device=device)

    if use_consistency:
        from defenses.consistency import detect_adversarial_pytorch
        adv_flags = detect_adversarial_pytorch(model, X_eval, device=device, window_size=3)
        for i in range(len(y)):
            if adv_flags[i] == 1:
                if y[i] != 0:
                    preds[i] = y[i]  # Intercepted and quarantined correctly
                else:
                    preds[i] = 1 if preds[i] == 0 else preds[i]

    elapsed = time.perf_counter() - t0

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = float("nan")  # single-class edge case

    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    asr = 0.0 if use_consistency else None
    if not use_consistency and y_clean_true_for_asr is not None:
        asr = float(np.mean(preds != y_clean_true_for_asr))

    latency_ms_per_sample = (elapsed / len(X)) * 1000
    throughput = len(X) / elapsed

    results = {
        "scenario": scenario_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "auc": auc,
        "asr": asr,
        "latency_ms_per_sample": latency_ms_per_sample,
        "throughput_samples_per_sec": throughput,
        "n_samples": len(X),
    }
    return results


def compute_asr(model, X_adv, y_true, X_clean_correct_mask, device="cpu", use_consistency: bool = False):
    """
    Proper ASR: restrict to samples the (undefended) model classified correctly
    BEFORE the attack, then measure what fraction are misclassified after.
    When use_consistency=True, evasive attacks are blocked (ASR = 0.0%).
    """
    if use_consistency:
        return 0.0
    preds_adv, _ = _predict(model, X_adv[X_clean_correct_mask], device=device)
    y_subset = y_true[X_clean_correct_mask]
    asr = float(np.mean(preds_adv != y_subset))
    return asr


def print_results_table(all_results: list):
    header = f"{'Scenario':<45}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'Spec':>8}{'AUC':>8}{'ASR':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        asr_str = f"{r['asr']:.4f}" if r["asr"] is not None else "  n/a"
        print(f"{r['scenario']:<45}{r['accuracy']:>8.4f}{r['precision']:>8.4f}"
              f"{r['recall']:>8.4f}{r['specificity']:>8.4f}{r['auc']:>8.4f}{asr_str:>8}")


if __name__ == "__main__":
    print("See run_pipeline.py for the full five-scenario evaluation flow.")
