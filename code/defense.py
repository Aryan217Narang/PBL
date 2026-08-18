"""
defense.py
Three defense components from paper Section 3.5:
  1. PGD  - Projected Gradient Descent adversarial training  (Section 3.5.1, Eq. 18-20, Algorithm 2)
  2. SS   - Spatial Smoothing                                 (Section 3.5.2, Eq. 21-23, Algorithm 3)
  3. PIOA - Pigeon-Inspired Optimization Algorithm            (Section 3.5.3, Eq. 24-28, Algorithm 4)
"""

import numpy as np
import tensorflow as tf
from scipy.ndimage import median_filter

MAX_PGD_SAMPLES = 10000


# ─────────────────────────────────────────────────────────────────────────────
# 1. PROJECTED GRADIENT DESCENT (PGD)  -  Section 3.5.1
# ─────────────────────────────────────────────────────────────────────────────

def pgd_adversarial_training(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epsilon: float = 0.02,
    alpha: float   = 0.003,
    n_iter: int    = 10,
    batch_size: int = 256,
) -> tuple:
    """
    Generate PGD adversarial examples for training data (Algorithm 2),
    then mix with the original training set (50/50 clean+adv) for robust adversarial training.
    """
    print(f"      PGD: epsilon={epsilon:.4f}, alpha={alpha}, n_iter={n_iter}")

    n_pgd = min(MAX_PGD_SAMPLES, len(X_train))
    idx = np.random.choice(len(X_train), n_pgd, replace=False)
    X_pgd = X_train[idx].astype(np.float32)
    y_pgd = y_train[idx]

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    X_adv_list = []
    total_batches = int(np.ceil(n_pgd / batch_size))

    for batch_num, start in enumerate(range(0, n_pgd, batch_size)):
        end = min(start + batch_size, n_pgd)
        X_batch = tf.constant(X_pgd[start:end], dtype=tf.float32)
        y_batch = tf.constant(y_pgd[start:end], dtype=tf.int32)

        # Eq. 19: Random init within epsilon ball
        delta = tf.Variable(
            tf.random.uniform(X_batch.shape, -epsilon, epsilon, dtype=tf.float32)
        )

        for _ in range(n_iter):
            with tf.GradientTape() as tape:
                tape.watch(delta)
                X_perturbed = tf.clip_by_value(X_batch + delta, 0.0, 1.0)
                preds = model(X_perturbed, training=False)
                loss = loss_fn(y_batch, preds)

            grads = tape.gradient(loss, delta)
            delta.assign(delta + alpha * tf.sign(grads))
            delta.assign(tf.clip_by_value(delta, -epsilon, epsilon))

        X_adv_list.append(
            tf.clip_by_value(X_batch + delta, 0.0, 1.0).numpy().astype(np.float32)
        )

        if (batch_num + 1) % 5 == 0 or (batch_num + 1) == total_batches:
            print(f"      PGD batch {batch_num + 1}/{total_batches}")

    X_adv = np.concatenate(X_adv_list, axis=0)

    # Balanced mix of original training subset + PGD adversarial examples
    clean_sub_idx = np.random.choice(len(X_train), n_pgd, replace=False)
    X_mixed = np.concatenate([X_train[clean_sub_idx], X_adv], axis=0)
    y_mixed = np.concatenate([y_train[clean_sub_idx], y_pgd], axis=0)

    shuffle_idx = np.random.permutation(len(X_mixed))
    return X_mixed[shuffle_idx], y_mixed[shuffle_idx]


# ─────────────────────────────────────────────────────────────────────────────
# 2. SPATIAL SMOOTHING (SS)  -  Section 3.5.2
# ─────────────────────────────────────────────────────────────────────────────

def spatial_smoothing(X: np.ndarray, window_radius: float = 1.0) -> np.ndarray:
    """
    Spatial Smoothing defense (Algorithm 3) using Median Filtering across feature dimensions.
    Denoises adversarial perturbations while preserving non-linear tabular feature relationships.
    """
    w_size = 3 if window_radius >= 1.0 else 1
    if w_size == 1:
        return X.astype(np.float32)

    orig_shape = X.shape
    if X.ndim == 3:
        X_2d = X.reshape(X.shape[0], X.shape[1])
        smoothed_2d = np.array([
            median_filter(row, size=w_size, mode="nearest") for row in X_2d
        ])
        return smoothed_2d.reshape(orig_shape).astype(np.float32)
    else:
        smoothed = np.array([
            median_filter(row, size=w_size, mode="nearest") for row in X
        ])
        return smoothed.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PIGEON-INSPIRED OPTIMIZATION ALGORITHM (PIOA)  -  Section 3.5.3
# ─────────────────────────────────────────────────────────────────────────────

class PIOA:
    """
    Pigeon-Inspired Optimization Algorithm (Algorithm 4).
    - Map and Compass Operator (MCO): global navigation  [Eq. 24-25]
    - Landmark Operator (LO): convergence phase          [Eq. 26-28]
    """

    def __init__(
        self,
        n_pigeons: int      = 50,
        dimensions: int     = 10,
        max_iterations: int = 100,
        r: float            = 0.5,
        bounds: tuple       = (0.001, 0.05),
    ):
        self.Np = n_pigeons
        self.D  = dimensions
        self.Nc = max_iterations
        self.R  = r
        self.bounds = bounds

        self.Nc1 = int(np.ceil(self.Nc * 0.6))
        self.Nc2 = self.Nc - self.Nc1

        self.X = np.random.uniform(bounds[0], bounds[1], (self.Np, self.D))
        self.V = np.random.uniform(-0.005, 0.005, (self.Np, self.D))

        self.fitness = np.zeros(self.Np)
        self.Xg = None
        self.best_fitness = -np.inf

    def _default_fitness(self, x: np.ndarray) -> float:
        val = np.mean(x)
        return float(1.0 / (1.0 + abs(val - 0.02)))

    def optimize(self, fitness_fn=None) -> tuple:
        if fitness_fn is None:
            fitness_fn = self._default_fitness

        for i in range(self.Np):
            self.fitness[i] = fitness_fn(self.X[i])
            if self.fitness[i] > self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.Xg = self.X[i].copy()

        # Phase 1: Map and Compass Operator (MCO)
        for t in range(self.Nc1):
            decay = np.exp(-self.R * t)
            for i in range(self.Np):
                rand_factor = np.random.rand(self.D)
                self.V[i] = self.V[i] * decay + rand_factor * (self.Xg - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
                self.X[i] = np.clip(self.X[i], self.bounds[0], self.bounds[1])

                fit = fitness_fn(self.X[i])
                self.fitness[i] = fit
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.Xg = self.X[i].copy()

        # Phase 2: Landmark Operator (LO)
        current_Np = self.Np
        X_curr = self.X.copy()
        fit_curr = self.fitness.copy()

        for t in range(self.Nc2):
            current_Np = max(2, int(np.ceil(current_Np / 2)))
            sort_idx = np.argsort(fit_curr)[::-1]
            X_curr = X_curr[sort_idx[:current_Np]]
            fit_curr = fit_curr[sort_idx[:current_Np]]

            weights = fit_curr / (np.sum(fit_curr) + 1e-12)
            Xc = np.sum(X_curr * weights[:, np.newaxis], axis=0)

            for i in range(current_Np):
                rand_factor = np.random.rand(self.D)
                X_curr[i] = X_curr[i] + rand_factor * (Xc - X_curr[i])
                X_curr[i] = np.clip(X_curr[i], self.bounds[0], self.bounds[1])

                fit = fitness_fn(X_curr[i])
                fit_curr[i] = fit
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.Xg = X_curr[i].copy()

        best_scalar = float(np.mean(self.Xg))
        return best_scalar, self.best_fitness


def pioa_optimize(base_epsilon: float = 0.02, pioa_cfg: dict = None) -> float:
    cfg = pioa_cfg or {}
    pioa = PIOA(
        n_pigeons=cfg.get("n_pigeons", 20),
        dimensions=cfg.get("dimensions", 5),
        max_iterations=cfg.get("max_iterations", 20),
        r=cfg.get("r", 0.5),
        bounds=(base_epsilon * 0.5, base_epsilon * 2.0),
    )
    best_eps, best_fit = pioa.optimize()
    return best_eps