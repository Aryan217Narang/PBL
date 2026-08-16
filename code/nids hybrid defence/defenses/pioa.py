"""
Pigeon-Inspired Optimization Algorithm (PIOA), used in Barik & Misra (2025) to
optimize adversarial sample generation during training (outperformed PSO/GA in
the paper). Here PIOA searches PGD's hyperparameters (eps, alpha, n_iter) to
maximize the hardened model's robust accuracy on a validation adversarial set.

Two classic PIOA phases:
  1. Map & compass operator - pigeons use a "virtual compass" (gradient-like
     pull toward the global best) to explore.
  2. Landmark operator - pigeons close to the landmark (center of the best
     half of the population) home in on it, narrowing the search.
"""

import numpy as np


class PIOA:
    def __init__(
        self,
        bounds: dict,             # e.g. {"eps": (0.01, 0.3), "alpha": (0.005, 0.05), "n_iter": (3, 15)}
        n_pigeons: int = 15,
        n_map_compass_iters: int = 10,
        n_landmark_iters: int = 10,
        R: float = 0.2,           # map & compass factor
        seed: int = 42,
    ):
        self.bounds = bounds
        self.param_names = list(bounds.keys())
        self.dim = len(self.param_names)
        self.n_pigeons = n_pigeons
        self.n_map_compass_iters = n_map_compass_iters
        self.n_landmark_iters = n_landmark_iters
        self.R = R
        self.rng = np.random.RandomState(seed)

        lo = np.array([bounds[k][0] for k in self.param_names])
        hi = np.array([bounds[k][1] for k in self.param_names])
        self.lo, self.hi = lo, hi

        self.positions = self.rng.uniform(lo, hi, size=(n_pigeons, self.dim))
        self.velocities = np.zeros((n_pigeons, self.dim))

    def _clip(self, X):
        return np.clip(X, self.lo, self.hi)

    def _vec_to_params(self, vec):
        d = dict(zip(self.param_names, vec))
        if "n_iter" in d:
            d["n_iter"] = int(round(d["n_iter"]))
        return d

    def optimize(self, fitness_fn, verbose: bool = True):
        """
        fitness_fn(params: dict) -> float, HIGHER is better
        (e.g. robust accuracy of a quickly-trained/fine-tuned model under PGD attack).
        """
        fitness = np.array([fitness_fn(self._vec_to_params(p)) for p in self.positions])
        gbest_idx = np.argmax(fitness)
        gbest = self.positions[gbest_idx].copy()
        gbest_fit = fitness[gbest_idx]

        # --- Phase 1: Map & compass operator ---
        for t in range(1, self.n_map_compass_iters + 1):
            for i in range(self.n_pigeons):
                self.velocities[i] = (
                    self.velocities[i] * np.exp(-self.R * t)
                    + self.rng.rand(self.dim) * (gbest - self.positions[i])
                )
                self.positions[i] = self._clip(self.positions[i] + self.velocities[i])

            fitness = np.array([fitness_fn(self._vec_to_params(p)) for p in self.positions])
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > gbest_fit:
                gbest_fit = fitness[best_idx]
                gbest = self.positions[best_idx].copy()

            if verbose:
                print(f"[PIOA map&compass] iter {t}/{self.n_map_compass_iters} "
                      f"best_fitness={gbest_fit:.4f} params={self._vec_to_params(gbest)}")

        # --- Phase 2: Landmark operator ---
        n_remaining = self.n_pigeons
        for t in range(1, self.n_landmark_iters + 1):
            # keep top half (closer to landmark), discard rest
            n_remaining = max(2, n_remaining // 2)
            order = np.argsort(-fitness)
            self.positions = self.positions[order[:n_remaining]]
            fitness = fitness[order[:n_remaining]]

            landmark = self.positions[np.argmax(fitness)]
            center = np.average(self.positions, axis=0, weights=self._softmax(fitness))

            for i in range(len(self.positions)):
                self.positions[i] = self._clip(
                    self.positions[i] + self.rng.rand(self.dim) * (center - self.positions[i])
                )

            fitness = np.array([fitness_fn(self._vec_to_params(p)) for p in self.positions])
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > gbest_fit:
                gbest_fit = fitness[best_idx]
                gbest = self.positions[best_idx].copy()

            if verbose:
                print(f"[PIOA landmark] iter {t}/{self.n_landmark_iters} "
                      f"best_fitness={gbest_fit:.4f} params={self._vec_to_params(gbest)}")

        return self._vec_to_params(gbest), gbest_fit

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()


def build_pgd_fitness_fn(build_model_fn, X_train, y_train, X_val_adv, y_val, device="cpu", quick_epochs=2):
    """
    Returns a fitness function PIOA can call: trains a fresh model with candidate
    PGD hyperparams on a fast surrogate subset for 2 epochs and scores it on held-out adversarial validation.
    """
    from defenses.pgd_training import pgd_adversarial_train
    from models.cnn import evaluate_accuracy

    # Fast surrogate subset for instant fitness evaluation
    n_surrogate = min(5000, len(X_train))
    X_sub = X_train[:n_surrogate]
    y_sub = y_train[:n_surrogate]

    def fitness_fn(params):
        model = build_model_fn().to(device)
        pgd_adversarial_train(
            model, X_sub, y_sub,
            eps=params["eps"], alpha=params["alpha"], n_iter=params["n_iter"],
            epochs=quick_epochs, device=device,
        )
        return evaluate_accuracy(model, X_val_adv, y_val, device=device)

    return fitness_fn
