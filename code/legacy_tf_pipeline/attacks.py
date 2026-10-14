"""
attacks.py
Three adversarial attack methods from paper Section 3.3:
  1. JSMA  — Jacobian Saliency Map Attack   (Section 3.3.1, Eq. 13-14)
  2. FGSM  — Fast Gradient Sign Method      (Section 3.3.2, Eq. 15)
  3. C&W   — Carlini & Wagner               (Section 3.3.3, Eq. 16-17)

All attacks are fully vectorized (batch-based) for speed.
Attack samples are capped at MAX_ATTACK_SAMPLES to keep runtime practical.
"""

import numpy as np
import tensorflow as tf

# Cap how many test samples are used for attack generation
# Full test set is used for evaluation — attacks only need enough to show vulnerability
MAX_ATTACK_SAMPLES = 1000


def _get_attack_subset(X: np.ndarray, y: np.ndarray):
    """Take a random subset of test samples for attack generation."""
    n = min(MAX_ATTACK_SAMPLES, len(X))
    idx = np.random.choice(len(X), n, replace=False)
    return X[idx], y[idx], idx


# ─────────────────────────────────────────────────────────────────────────────
# 1. FGSM — fastest, do this first
# ─────────────────────────────────────────────────────────────────────────────

def generate_fgsm(model: tf.keras.Model, X: np.ndarray, epsilon: float) -> np.ndarray:
    """
    FGSM (Eq. 15): eta = epsilon * sign(∇_w I(Θ, w, z))
    Fully vectorized — runs entire batch at once.
    """
    X_sub, _, _ = _get_attack_subset(X, np.zeros(len(X)))  # labels not needed for FGSM

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    X_tensor = tf.constant(X_sub, dtype=tf.float32)

    # Use model predictions as proxy labels
    y_pred = tf.argmax(model(X_tensor, training=False), axis=1)

    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        preds = model(X_tensor, training=False)
        loss = loss_fn(y_pred, preds)

    grads = tape.gradient(loss, X_tensor)
    perturbation = epsilon * tf.sign(grads)
    X_adv = tf.clip_by_value(X_tensor + perturbation, 0.0, 1.0).numpy()

    # Return full test set with adversarial subset substituted back
    X_out = X.copy()
    idx = np.random.choice(len(X), len(X_sub), replace=False)
    X_out[idx] = X_adv
    return X_out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2. JSMA — vectorized batch version
# ─────────────────────────────────────────────────────────────────────────────

def generate_jsma(model: tf.keras.Model, X: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
    """
    JSMA (Eq. 13-14, Algorithm 1) — vectorized batch implementation.
    Perturbs the top-k most salient features per sample using the Jacobian.
    Much faster than the per-sample loop version.
    """
    X_sub, y_sub, sub_idx = _get_attack_subset(X, y)
    epsilon  = params["epsilon"]
    max_iter = params["max_iter"]

    X_adv = X_sub.copy().astype(np.float32)

    for iteration in range(max_iter):
        X_tensor = tf.constant(X_adv, dtype=tf.float32)

        with tf.GradientTape(persistent=False) as tape:
            tape.watch(X_tensor)
            preds = model(X_tensor, training=False)
            # Maximize the probability of the top predicted class
            top_class_probs = tf.reduce_max(preds, axis=1)
            loss = tf.reduce_sum(top_class_probs)

        # Jacobian: gradient of loss w.r.t. input — shape (N, features, 1)
        grads = tape.gradient(loss, X_tensor).numpy()

        # Saliency map: absolute gradient value per feature
        saliency = np.abs(grads)

        # Perturb top feature per sample (vectorized)
        flat_saliency = saliency.reshape(len(X_adv), -1)
        top_features = np.argmax(flat_saliency, axis=1)

        for i, feat in enumerate(top_features):
            idx = np.unravel_index(feat, X_adv.shape[1:])
            X_adv[i][idx] = np.clip(X_adv[i][idx] + epsilon, 0.0, 1.0)

        if (iteration + 1) % 5 == 0:
            print(f"      JSMA iteration {iteration + 1}/{max_iter}")

    # Substitute adversarial samples back into full test array
    X_out = X.copy()
    X_out[sub_idx] = X_adv
    return X_out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. C&W — batched optimization
# ─────────────────────────────────────────────────────────────────────────────

def generate_cw(model: tf.keras.Model, X: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
    """
    C&W L2 Attack (Eq. 16-17).
    min_delta ||delta||_2 + c * loss(x + delta, target)
    Batched Adam optimization — much faster than per-sample version.
    """
    X_sub, y_sub, sub_idx = _get_attack_subset(X, y)
    max_iter = params["max_iter"]
    c        = params["epsilon"]  # confidence constant

    loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy()
    delta     = tf.Variable(tf.zeros_like(X_sub, dtype=tf.float32))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    y_tensor  = tf.constant(y_sub, dtype=tf.int32)

    for iteration in range(max_iter):
        with tf.GradientTape() as tape:
            X_perturbed = tf.clip_by_value(X_sub + delta, 0.0, 1.0)
            preds       = model(X_perturbed, training=False)
            ce_loss     = loss_fn(y_tensor, preds)
            # L2 norm of perturbation per sample
            l2_loss     = tf.reduce_mean(
                tf.norm(tf.reshape(delta, [tf.shape(delta)[0], -1]), axis=1)
            )
            total_loss  = l2_loss + c * ce_loss

        grads = tape.gradient(total_loss, [delta])
        optimizer.apply_gradients(zip(grads, [delta]))

        if (iteration + 1) % 5 == 0:
            print(f"      C&W iteration {iteration + 1}/{max_iter}")

    X_adv = tf.clip_by_value(X_sub + delta, 0.0, 1.0).numpy().astype(np.float32)

    X_out = X.copy()
    X_out[sub_idx] = X_adv
    return X_out.astype(np.float32)