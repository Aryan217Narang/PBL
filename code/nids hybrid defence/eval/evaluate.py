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
    y_clean_true_for_asr: np.ndarray = None,
    scenario_name: str = "",
):
    """
    Evaluate one scenario. Set `y_clean_true_for_asr` to the TRUE labels of the
    original (pre-attack) samples that the adversarial set X was derived from,
    to compute ASR = fraction of originally-correct samples flipped by the attack.
    """
    X_eval = spatial_smoothing(X, window_size=smoothing_window) if use_spatial_smoothing else X

    t0 = time.perf_counter()
    preds, probs = _predict(model, X_eval, device=device)
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

    asr = None
    if y_clean_true_for_asr is not None:
        # ASR: of samples the model got right pre-attack, what fraction does the
        # attack flip to wrong now?
        correct_before = None  # caller should pass in pre-attack correctness mask instead if available
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


def compute_asr(model, X_adv, y_true, X_clean_correct_mask, device="cpu"):
    """
    Proper ASR: restrict to samples the (undefended) model classified correctly
    BEFORE the attack, then measure what fraction are misclassified after.
    X_clean_correct_mask: boolean array, True where clean prediction == y_true.
    """
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
