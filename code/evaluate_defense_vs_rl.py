"""
Evaluate the trained RL attacker against the defense pipeline and compare with static baselines (FGSM, PGD).

Generates:
- Attack Success Rate (ASR) for RL policy on a held-out malicious test set
- ASR for FGSM and PGD baselines
- Saves a summary CSV with ASR and average perturbation magnitudes
- Produces ROC curve and confusion matrix plots (optional dependencies: matplotlib, scikit-learn)

Usage
- Ensure the RL policy file `models/rl_attacker_policy.zip` exists (created by train_rl_attacker.py)
- Ensure the TF model (defended classifier) and preprocessing cache `data/processed_cicddos2019.npz` exist
- Run: python code/evaluate_defense_vs_rl.py

"""
from typing import Tuple, Dict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

import gymnasium as gym
from stable_baselines3 import PPO

from rl_env import AdaptiveNIDSEnv
from defense import spatial_smoothing
from model import build_cnn_model
from attacks import generate_fgsm


def load_policy(policy_path: str) -> PPO:
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy not found: {policy_path}")
    model = PPO.load(policy_path)
    return model


def evaluate_rl_policy(
    policy: PPO,
    tf_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_epsilon: float = 0.05,
    smoothing_window: float = 1.0,
    n_eval: int = 500,
) -> Dict[str, float]:
    """
    Evaluate RL policy on n_eval malicious samples sampled from the provided X_test,y_test.
    Returns metrics dict with ASR and avg perturbation magnitude.
    """
    env = AdaptiveNIDSEnv(model=tf_model, X=X_test, y=y_test, max_epsilon=max_epsilon,
                          lambda_reg=0.0, smoothing_fn=spatial_smoothing, smoothing_window=smoothing_window, max_steps=1)

    successes = 0
    l2s = []
    selected_indices = np.where(y_test == 1)[0]
    if len(selected_indices) == 0:
        selected_indices = np.arange(len(y_test))

    idxs = np.random.choice(selected_indices, size=min(n_eval, len(selected_indices)), replace=False)
    for idx in idxs:
        # reset to specific sample by monkey-patching internal idx (convenience)
        env._current_idx = int(idx)
        obs = env.X[env._current_idx].reshape(env.n_features)
        action, _ = policy.predict(obs, deterministic=True)
        _, _, terminated, truncated, info = env.step(action)
        if info.get("success", False):
            successes += 1
        l2s.append(info.get("l2_action", 0.0))

    asr = successes / float(len(idxs))
    return {"asr": asr, "avg_l2": float(np.mean(l2s)), "n_eval": len(idxs)}


def pgd_attack_tf(model, X: np.ndarray, epsilon: float = 0.05, alpha: float = 0.01, n_iter: int = 10) -> np.ndarray:
    """Simple PGD implementation that perturbs the provided X input (expects batched input shape (N, features, 1)).

    Returns adversarially perturbed copy of X.
    """
    import tensorflow as tf
    X_adv = X.copy().astype(np.float32)
    N = min(500, len(X_adv))
    idx = np.random.choice(len(X_adv), N, replace=False)
    X_sub = X_adv[idx]

    X_tensor = tf.constant(X_sub)
    delta = tf.Variable(tf.random.uniform(X_tensor.shape, -epsilon, epsilon))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for i in range(n_iter):
        with tf.GradientTape() as tape:
            tape.watch(delta)
            x_pert = tf.clip_by_value(X_tensor + delta, 0.0, 1.0)
            preds = model(x_pert, training=False)
            # use predicted label to compute gradient (untargeted)
            y_pred = tf.argmax(preds, axis=1)
            loss = loss_fn(y_pred, preds)
        grads = tape.gradient(loss, delta)
        signed = tf.sign(grads)
        delta.assign(delta + alpha * signed)
        delta.assign(tf.clip_by_value(delta, -epsilon, epsilon))

    X_adv[idx] = tf.clip_by_value(X_tensor + delta, 0.0, 1.0).numpy()
    return X_adv


def main():
    CACHE_PATH = "data/processed_cicddos2019.npz"
    POLICY_PATH = "models/rl_attacker_policy.zip"
    TF_WEIGHTS_PATH = None  # set if you saved TF weights; otherwise new model will be untrained

    if not os.path.exists(CACHE_PATH):
        raise RuntimeError(f"Cache missing: {CACHE_PATH} - run preprocessing first.")
    data = np.load(CACHE_PATH)
    X_test = data["X_test"]
    y_test = data["y_test"]

    input_shape = tuple(X_test.shape[1:])
    tf_model = build_cnn_model(input_shape=input_shape, n_classes=2)
    if TF_WEIGHTS_PATH is not None and os.path.exists(TF_WEIGHTS_PATH):
        try:
            tf_model.load_weights(TF_WEIGHTS_PATH)
            print(f"Loaded TF model weights from {TF_WEIGHTS_PATH}")
        except Exception as e:
            print(f"Failed to load TF weights: {e}")

    # Load RL policy
    if not os.path.exists(POLICY_PATH):
        raise RuntimeError(f"RL policy not found: {POLICY_PATH}. Train first with train_rl_attacker.py")
    policy = PPO.load(POLICY_PATH)

    # Evaluate RL policy
    rl_metrics = evaluate_rl_policy(policy, tf_model, X_test, y_test, max_epsilon=0.05, smoothing_window=1.0, n_eval=500)
    print("RL evaluation:", rl_metrics)

    # FGSM baseline (use small epsilon choices)
    eps_fgsm = 0.05
    X_fgsm = generate_fgsm(tf_model, X_test, epsilon=eps_fgsm)
    # compute ASR on malicious subset: model predict probabilities after smoothing
    from defense import spatial_smoothing
    X_fgsm_smoothed = spatial_smoothing(X_fgsm, window_radius=1.0)
    preds = tf_model.predict(X_fgsm_smoothed)
    p_mal = preds[:, 1]
    # on originally malicious samples
    mal_idx = np.where(y_test == 1)[0]
    if len(mal_idx) == 0:
        mal_idx = np.arange(len(y_test))
    asr_fgsm = np.mean(p_mal[mal_idx] < 0.5)
    print(f"FGSM ASR (eps={eps_fgsm}): {asr_fgsm:.3f}")

    # PGD baseline (simple implementation)
    X_pgd = pgd_attack_tf(tf_model, X_test, epsilon=0.05, alpha=0.01, n_iter=10)
    X_pgd_smoothed = spatial_smoothing(X_pgd, window_radius=1.0)
    preds_pgd = tf_model.predict(X_pgd_smoothed)
    p_mal_pgd = preds_pgd[:, 1]
    asr_pgd = np.mean(p_mal_pgd[mal_idx] < 0.5)
    print(f"PGD ASR (eps=0.05,iter=10): {asr_pgd:.3f}")

    # Summarize to CSV
    out = {
        "method": ["rl_policy", "fgsm", "pgd"],
        "asr": [rl_metrics["asr"], float(asr_fgsm), float(asr_pgd)],
        "avg_l2": [rl_metrics.get("avg_l2", float('nan')), float('nan'), float('nan')],
    }
    df = pd.DataFrame(out)
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "rl_vs_baselines_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")

    # Optional plots: ROC & confusion matrix (for RL vs clean)
    # Generate predictions for RL-attacked examples (apply policy to mal samples)
    policy_preds = []
    policy_targets = []
    selected_indices = mal_idx[:200]  # small set to keep plotting fast
    for idx in selected_indices:
        env = AdaptiveNIDSEnv(model=tf_model, X=X_test, y=y_test, max_epsilon=0.05, smoothing_fn=spatial_smoothing)
        env._current_idx = int(idx)
        obs = env.X[env._current_idx].reshape(env.n_features)
        action, _ = policy.predict(obs, deterministic=True)
        obs2, _, _, _, info = env.step(action)
        policy_preds.append(info.get("p_malicious", 1.0))
        policy_targets.append(0)  # target: benign (attack wants benign)

    # ROC using policy outputs (inverted to compute TPR/FPR for benign class vs malicious)
    fpr, tpr, _ = roc_curve([1] * len(policy_preds), policy_preds)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"RL policy ROC (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/rl_policy_roc.png", dpi=150)
    print("Saved ROC plot: results/rl_policy_roc.png")


if __name__ == "__main__":
    main()
