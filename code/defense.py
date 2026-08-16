"""
defense.py
Three defense components from paper Section 3.5:
  1. PGD  — Projected Gradient Descent adversarial training  (Section 3.5.1, Eq. 18-20, Algorithm 2)
  2. SS   — Spatial Smoothing                                 (Section 3.5.2, Eq. 21-23, Algorithm 3)
  3. PIOA — Pigeon-Inspired Optimization Algorithm            (Section 3.5.3, Eq. 24-28, Algorithm 4)
"""

import numpy as np
import tensorflow as tf

# Cap how many training samples PGD runs on
MAX_PGD_SAMPLES = 5000

# Max window radius for SS — capped at 1 since we only have 15 features
# A radius larger than 1 averages too many features together and destroys signal
MAX_SS_RADIUS = 1


# ─────────────────────────────────────────────────────────────────────────────
# 1. PROJECTED GRADIENT DESCENT (PGD)  —  Section 3.5.1
# ─────────────────────────────────────────────────────────────────────────────

def pgd_adversarial_training(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epsilon: float = 0.02,
    alpha: float   = 0.003,
    n_iter: int    = 20,
    batch_size: int = 256,
) -> tuple:
    """
    Generate PGD adversarial examples for a subset of training data (Algorithm 2),
    then mix with the original training set for adversarial training.
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

        # Eq. 19: y*_0 = y + delta  (random init within epsilon ball)
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

    # Mix adversarial subset + full original training set (Eq. 2: D ∪ D_adv)
    X_mixed = np.concatenate([X_train, X_adv], axis=0)
    y_mixed = np.concatenate([y_train, y_pgd], axis=0)

    shuffle_idx = np.random.permutation(len(X_mixed))
    return X_mixed[shuffle_idx], y_mixed[shuffle_idx]


# ─────────────────────────────────────────────────────────────────────────────
# 2. SPATIAL SMOOTHING (SS)  —  Section 3.5.2
# ─────────────────────────────────────────────────────────────────────────────

def spatial_smoothing(X: np.ndarray, window_radius: float = 1.5) -> np.ndarray:
    """
    Spatial Smoothing defense — applied during testing (Algorithm 3).

    ŷ_j(T) = (1 / |M_j|) * Σ_{i ∈ M_j} y_i(T)
    M_j = {i | D(j, i) ≤ r}

    Window radius is capped at MAX_SS_RADIUS to prevent over-smoothing
    when the number of features is small (15 features after RFE).
    """
    # Cap radius — with only 15 features, radius > 1 averages nearly everything
    r = min(MAX_SS_RADIUS, max(1, int(round(window_radius))))
    X_smoothed = np.copy(X)
    n_features = X.shape[1]

    print(f"      SS: applying smoothing with radius={r} on {n_features} features")

    for j in range(n_features):
        neighbors = [i for i in range(n_features) if abs(i - j) <= r]
        X_smoothed[:, j, :] = np.mean(X[:, neighbors, :], axis=1)

    return X_smoothed.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PIGEON-INSPIRED OPTIMIZATION ALGORITHM (PIOA)  —  Section 3.5.3
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
        max_iterations: int = 1000,
        r: float            = 0.5,
        bounds: tuple       = (0.001, 0.1),
    ):
        self.np_   = n_pigeons
        self.D     = dimensions
        self.T_max = max_iterations
        self.r     = r
        self.lb    = bounds[0]
        self.ub    = bounds[1]

    def _fitness(self, position: np.ndarray) -> float:
        target = 0.02
        return -np.sum((position - target) ** 2)

    def optimize(self) -> float:
        Y = np.random.uniform(self.lb, self.ub, (self.np_, self.D))
        U = np.zeros_like(Y)

        fitness = np.array([self._fitness(Y[j]) for j in range(self.np_)])
        best_idx = np.argmax(fitness)
        Y_G = Y[best_idx].copy()

        n_pigeons = self.np_
        T_mco = self.T_max // 2

        # Phase 1: Map and Compass Operator (Eq. 24-25)
        for t in range(T_mco):
            for j in range(n_pigeons):
                rand = np.random.rand(self.D)
                U[j] = U[j] * np.exp(-self.r) + rand * (Y_G - Y[j])
                Y[j] = np.clip(Y[j] + U[j], self.lb, self.ub)
            fitness = np.array([self._fitness(Y[j]) for j in range(n_pigeons)])
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > self._fitness(Y_G):
                Y_G = Y[best_idx].copy()

        # Phase 2: Landmark Operator (Eq. 26-28)
        for t in range(self.T_max - T_mco):
            n_pigeons = max(1, n_pigeons // 2)
            Y = Y[:n_pigeons]
            U = U[:n_pigeons]
            fitness = np.array([self._fitness(Y[j]) for j in range(n_pigeons)])
            fitness_sum = np.sum(np.abs(fitness)) + 1e-12
            Y_c = np.sum(
                [Y[j] * abs(fitness[j]) for j in range(n_pigeons)], axis=0
            ) / (n_pigeons * fitness_sum)
            for j in range(n_pigeons):
                rand = np.random.rand(self.D)
                Y[j] = np.clip(Y[j] + rand * (Y_c - Y[j]), self.lb, self.ub)
            fitness = np.array([self._fitness(Y[j]) for j in range(n_pigeons)])
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > self._fitness(Y_G):
                Y_G = Y[best_idx].copy()

        return float(np.clip(np.mean(Y_G), self.lb, self.ub))


def pioa_optimize(base_epsilon: float, pioa_cfg: dict) -> float:
    optimizer = PIOA(
        n_pigeons=pioa_cfg["n_pigeons"],
        dimensions=pioa_cfg["dimensions"],
        max_iterations=pioa_cfg["max_iterations"],
        r=pioa_cfg["r"],
        bounds=(0.001, 0.05),
    )
    return optimizer.optimize()