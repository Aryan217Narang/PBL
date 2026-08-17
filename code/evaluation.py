"""
evaluation.py
All evaluation metrics from paper Section 4 (Eq. 29-37):
  Accuracy, Precision, Recall, F1-score, AUC, Specificity, ASR
  + Confusion Matrix, Classification Report, ROC curve plots
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize


# ─────────────────────────────────────────────────────────────────────────────
# Metric computations  (Equations 29-37)
# ─────────────────────────────────────────────────────────────────────────────

def compute_auc(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """
    Robust AUC computation for both binary and multiclass (Eq. 35).
    Uses macro-average OvR for multiclass, handles missing classes gracefully.
    """
    n_classes = y_pred_prob.shape[1]
    classes = np.unique(y_true)

    try:
        if n_classes == 2:
            return roc_auc_score_binary(y_true, y_pred_prob[:, 1])

        # Binarize only the classes that actually appear in y_true
        y_bin = label_binarize(y_true, classes=np.arange(n_classes))

        auc_scores = []
        for i in range(n_classes):
            # Skip classes not present in y_true
            if len(np.unique(y_bin[:, i])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
            auc_scores.append(auc(fpr, tpr))

        return float(np.mean(auc_scores)) if auc_scores else float('nan')

    except Exception as e:
        return float('nan')


def roc_auc_score_binary(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


def compute_asr(model, X_clean, X_adv, y_true) -> float:
    """ASR (Eq. 37): ratio of successful attacks on previously correct samples."""
    y_clean_pred = np.argmax(model.predict(X_clean, verbose=0), axis=1)
    y_adv_pred   = np.argmax(model.predict(X_adv,   verbose=0), axis=1)
    correctly_classified = (y_clean_pred == y_true)
    n_correct = np.sum(correctly_classified)
    if n_correct == 0:
        return 1.0
    n_attacked = np.sum(correctly_classified & (y_adv_pred != y_true))
    return float(n_attacked / n_correct)


def compute_specificity(y_true, y_pred) -> float:
    """Specificity (Eq. 36): macro-average TN / (TN + FP) across all classes."""
    classes = np.unique(y_true)
    specs = []
    for cls in classes:
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
        fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
        specs.append(tn / (tn + fp + 1e-12))
    return float(np.mean(specs))


def evaluate_model(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    label: str = "",
    X_clean: np.ndarray = None,
) -> dict:
    """Run full evaluation and return a results dict."""
    y_pred_prob = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    n_classes = y_pred_prob.shape[1]
    avg = "binary" if n_classes == 2 else "weighted"

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=avg, zero_division=0)
    auc_score = compute_auc(y_true, y_pred_prob)
    spec = compute_specificity(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    tp = int(np.sum(np.diag(cm)))
    fp = int(np.sum(cm.sum(axis=0) - np.diag(cm)))
    fn = int(np.sum(cm.sum(axis=1) - np.diag(cm)))
    tn = int(cm.sum() - tp - fp - fn)

    asr = compute_asr(model, X_clean, X, y_true) if X_clean is not None else None

    return {
        "label":       label,
        "accuracy":    round(acc  * 100, 2),
        "precision":   round(prec * 100, 2),
        "recall":      round(rec  * 100, 2),
        "f1_score":    round(f1   * 100, 2),
        "auc":         round(auc_score, 4) if not np.isnan(auc_score) else "N/A",
        "specificity": round(spec * 100, 2),
        "asr":         round(asr  * 100, 2) if asr is not None else "N/A",
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "cm":          cm,
        "y_pred":      y_pred,
        "y_pred_prob": y_pred_prob,
        "y_true":      y_true,
        "n_classes":   n_classes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Console printing
# ─────────────────────────────────────────────────────────────────────────────

def print_results(res: dict):
    print(f"\n  +-- {res['label']} {'-'*(45 - len(res['label']))}")
    print(f"  |  Accuracy   : {res['accuracy']:>7.2f}%")
    print(f"  |  Precision  : {res['precision']:>7.2f}%")
    print(f"  |  Recall     : {res['recall']:>7.2f}%")
    print(f"  |  F1-Score   : {res['f1_score']:>7.2f}%")
    print(f"  |  AUC        : {str(res['auc']):>8}")
    print(f"  |  Specificity: {res['specificity']:>7.2f}%")
    if res["asr"] != "N/A":
        print(f"  |  ASR        : {res['asr']:>7.2f}%")
    print(f"  |  TP={res['tp']}  TN={res['tn']}  FP={res['fp']}  FN={res['fn']}")
    print(f"  +{'-'*50}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_training_history(history, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_history.png"), dpi=150)
    plt.close()


def _plot_confusion_matrix(cm, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_roc_curves(results_list: list, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, res in enumerate(results_list):
        y_true = res["y_true"]
        y_prob = res["y_pred_prob"]
        n_cls  = res["n_classes"]

        try:
            y_bin = label_binarize(y_true, classes=np.arange(n_cls))
            auc_scores, fprs, tprs = [], [], []

            for j in range(n_cls):
                if len(np.unique(y_bin[:, j])) < 2:
                    continue
                fpr, tpr, _ = roc_curve(y_bin[:, j], y_prob[:, j])
                auc_scores.append(auc(fpr, tpr))
                fprs.append(fpr)
                tprs.append(tpr)

            if not auc_scores:
                continue

            # Interpolate to common FPR grid for mean curve
            mean_fpr = np.linspace(0, 1, 200)
            mean_tpr = np.mean([np.interp(mean_fpr, f, t) for f, t in zip(fprs, tprs)], axis=0)
            mean_auc = np.mean(auc_scores)

            ax.plot(mean_fpr, mean_tpr, color=colors[i % 4],
                    label=f"{res['label']} (AUC = {mean_auc:.2f})")
        except Exception:
            continue

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_comparison_bar(results: list, metrics: list, title: str, out_path: str):
    labels = [r["label"] for r in results]
    x = np.arange(len(labels))
    width = 0.2
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2), 5))
    for i, metric in enumerate(metrics):
        vals = [r.get(metric, 0) for r in results]
        ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=colors[i])

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_summary_csv(all_results: dict, out_path: str):
    import pandas as pd
    rows = []
    for key, res in all_results.items():
        rows.append({
            "Experiment":      res["label"],
            "Accuracy (%)":    res["accuracy"],
            "Precision (%)":   res["precision"],
            "Recall (%)":      res["recall"],
            "F1-Score (%)":    res["f1_score"],
            "AUC":             res["auc"],
            "Specificity (%)": res["specificity"],
            "ASR (%)":         res["asr"],
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"      Summary saved: {out_path}")


def plot_results(
    pre_attack_res: dict,
    all_attack_results: dict,
    defense_results: dict,
    history,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    _plot_training_history(history, out_dir)

    _plot_confusion_matrix(
        pre_attack_res["cm"],
        "Confusion Matrix: Pre-Attack",
        os.path.join(out_dir, "cm_pre_attack.png")
    )

    for part_label, attack_data in all_attack_results.items():
        part_slug = part_label.replace(" ", "_").lower()
        for attack_name in ["jsma", "fgsm", "cw"]:
            res = attack_data[attack_name]
            _plot_confusion_matrix(
                res["cm"],
                f"CM: Post-{attack_name.upper()} {part_label}",
                os.path.join(out_dir, f"cm_post_{attack_name}_{part_slug}.png")
            )
        def_data = defense_results.get(part_label, {})
        if def_data:
            _plot_confusion_matrix(
                def_data["hybrid"]["cm"],
                f"CM: Hybrid Defense {part_label}",
                os.path.join(out_dir, f"cm_hybrid_{part_slug}.png")
            )

    _plot_roc_curves(
        [all_attack_results[p]["jsma"] for p in all_attack_results],
        "AUC-ROC: Post-JSMA Attack",
        os.path.join(out_dir, "roc_post_jsma.png")
    )

    _plot_roc_curves(
        [defense_results[p]["hybrid"] for p in defense_results],
        "AUC-ROC: Hybrid Defense",
        os.path.join(out_dir, "roc_hybrid_defense.png")
    )

    best_part = list(all_attack_results.keys())[-1]
    _plot_comparison_bar(
        [
            pre_attack_res,
            all_attack_results[best_part]["jsma"],
            all_attack_results[best_part]["fgsm"],
            all_attack_results[best_part]["cw"],
            defense_results[best_part]["pgd"],
            defense_results[best_part]["ss"],
            defense_results[best_part]["hybrid"],
        ],
        metrics=["accuracy", "precision", "recall", "f1_score"],
        title="Performance Comparison Across Experiments",
        out_path=os.path.join(out_dir, "comparison_bar.png")
    )

    all_flat = {"pre_attack": pre_attack_res}
    for part, data in all_attack_results.items():
        all_flat[f"{part}_jsma"] = data["jsma"]
        all_flat[f"{part}_fgsm"] = data["fgsm"]
        all_flat[f"{part}_cw"]   = data["cw"]
    for part, data in defense_results.items():
        all_flat[f"{part}_pgd"]    = data["pgd"]
        all_flat[f"{part}_ss"]     = data["ss"]
        all_flat[f"{part}_hybrid"] = data["hybrid"]

    _save_summary_csv(all_flat, os.path.join(out_dir, "summary_results.csv"))
   
# ─────────────────────────────────────────────────────────────
# AICC + TCC Detection Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_detection(adv_flags: np.ndarray, y_true: np.ndarray):
    """
    Evaluate adversarial detection performance.

    adv_flags: 1 = detected as adversarial, 0 = normal
    y_true: ground truth (you must define adversarial labels externally)
    """

    # Assume:
    # y_true_adv = 1 for adversarial samples, 0 for clean
    y_true_adv = y_true

    tp = np.sum((adv_flags == 1) & (y_true_adv == 1))
    tn = np.sum((adv_flags == 0) & (y_true_adv == 0))
    fp = np.sum((adv_flags == 1) & (y_true_adv == 0))
    fn = np.sum((adv_flags == 0) & (y_true_adv == 1))

    detection_rate = tp / (tp + fn + 1e-12)   # True Positive Rate
    false_alarm_rate = fp / (fp + tn + 1e-12)

    return {
        "detection_rate": round(detection_rate * 100, 2),
        "false_alarm_rate": round(false_alarm_rate * 100, 2),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }