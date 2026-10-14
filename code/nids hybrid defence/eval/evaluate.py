"""
Evaluation matching the paper's scenarios and metrics:
  Accuracy, Precision, Recall, Specificity, AUC, Attack Success Rate (ASR),
  plus latency/throughput.

Fixed in this version:
  - Removed ground-truth label peek (preds[i] = y[i])
  - Removed hardcoded ASR = 0.0
  - Honest active NIDS defense policy: flagged samples are intercepted as Attack (1)
  - Modular consistency modes: "none", "aicc", "tcc", "both"
  - Proper ASR computation measuring true evasions against the defense
"""

import time
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)

from defenses.spatial_smoothing import spatial_smoothing
from defenses.consistency import detect_adversarial_pytorch


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
    smoothing_mode: str = "temporal",
    use_consistency: bool = False,
    consistency_mode: str = "both",
    y_clean_true_for_asr: np.ndarray = None,
    scenario_name: str = "",
):
    """
    Evaluate one scenario honestly.
    
    When use_consistency=True:
      - Detects adversarial/anomalous samples via AICC, TCC, or combined.
      - Intercepted samples are quarantined/alerted as Attack (preds = 1).
      - NEVER accesses or cheats using ground-truth y.
      - Any false alarms on benign flows penalize false positive rate / specificity.
    """
    X_eval = (
        spatial_smoothing(
            X,
            window_size=smoothing_window,
            mode=smoothing_mode,
            model=model,
            device=device,
        )
        if use_spatial_smoothing
        else X
    )

    t0 = time.perf_counter()
    preds, probs = _predict(model, X_eval, device=device)

    adv_flags = None
    if use_consistency:
        adv_flags = detect_adversarial_pytorch(
            model, X_eval, device=device, window_size=3, mode=consistency_mode
        )
        # Active NIDS policy: anomalous/adversarial detections are flagged as Attack (1).
        # Samples not flagged retain their model predictions. No label oracle used.
        preds = np.where(adv_flags == 1, 1, preds)

    elapsed = time.perf_counter() - t0

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    # Honest ASR: fraction of attack samples that successfully evaded detection
    asr = None
    if y_clean_true_for_asr is not None:
        attack_mask = (y_clean_true_for_asr == 1)
        if np.sum(attack_mask) > 0:
            # An attack evades if the final system output is Benign (0)
            asr = float(np.mean(preds[attack_mask] == 0))

    # Consistency detection statistics (if applicable)
    det_rate = float(np.mean(adv_flags == 1)) if adv_flags is not None else None
    far = None
    if adv_flags is not None and np.sum(y == 0) > 0:
        far = float(np.mean(adv_flags[y == 0] == 1))

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
        "detection_rate": det_rate,
        "false_alarm_rate": far,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "latency_ms_per_sample": latency_ms_per_sample,
        "throughput_samples_per_sec": throughput,
        "n_samples": len(X),
    }
    return results


def compute_asr(
    model,
    X_adv,
    y_true,
    X_clean_correct_mask,
    device: str = "cpu",
    use_spatial_smoothing: bool = False,
    smoothing_window: int = 3,
    smoothing_mode: str = "temporal",
    use_consistency: bool = False,
    consistency_mode: str = "both",
):
    """
    Proper ASR computation without cheating:
      - Evaluates on samples that the clean model originally got right.
      - Denoises the continuous stream sequentially to preserve temporal continuity.
      - If consistency is enabled, flags are intercepted as Attack (1).
      - Returns fraction of attack samples that successfully evade (final pred == 0).
    """
    if len(X_adv) == 0:
        return 0.0

    X_eval = (
        spatial_smoothing(
            X_adv,
            window_size=smoothing_window,
            mode=smoothing_mode,
            model=model,
            device=device,
        )
        if use_spatial_smoothing
        else X_adv
    )
    preds, _ = _predict(model, X_eval, device=device)

    if use_consistency:
        adv_flags = detect_adversarial_pytorch(
            model, X_eval, device=device, window_size=3, mode=consistency_mode
        )
        preds = np.where(adv_flags == 1, 1, preds)

    # Attacks succeed when an actual attack (y=1) is classified as Benign (0)
    preds_subset = preds[X_clean_correct_mask]
    y_subset = y_true[X_clean_correct_mask]

    attack_mask = (y_subset == 1)
    if np.sum(attack_mask) == 0:
        return 0.0

    asr = float(np.mean(preds_subset[attack_mask] == 0))
    return asr


def print_results_table(all_results: list):
    header = f"{'Scenario':<48}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'Spec':>8}{'AUC':>8}{'ASR':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        asr_str = f"{r['asr']:.4f}" if r.get("asr") is not None else "     n/a"
        print(f"{r['scenario']:<48}{r['accuracy']:>8.4f}{r['precision']:>8.4f}"
              f"{r['recall']:>8.4f}{r['specificity']:>8.4f}{r['auc']:>8.4f}{asr_str:>8}")
