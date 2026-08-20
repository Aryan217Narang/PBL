"""
Adaptive RL environment for an adaptive black-box attacker against the NIDS defense pipeline.

This module exposes AdaptiveNIDSEnv, a Gymnasium-compatible environment that wraps the
MinMaxScaler->ICA->RFE preprocessing output (cached arrays), the Spatial Smoothing filter
(spatial_smoothing), and the trained 1D-CNN classifier.

Notes / Instructions for use
- The repository's preprocessing pipeline caches transformed arrays to `data/processed_cicddos2019.npz`.
  The environment expects preprocessed inputs (X, y) in the same format used by the 1D-CNN:
  X shape (N, n_features, 1), y shape (N,).
- The TensorFlow model should be a compiled `tf.keras.Model` that accepts inputs shaped like X.
  Load trained weights and pass the model instance into the environment constructor.
- The environment is black-box w.r.t. the classifier: only the final softmax probability is queried.

API
- AdaptiveNIDSEnv(model, X, y, max_epsilon=0.05, lambda_reg=0.01, smoothing_window=3, max_steps=1)

The environment produces a single-step episode by default: each episode samples a random malicious
example from the provided (preprocessed) dataset and gives the agent one chance to perturb the sample.
This is consistent with evaluating per-sample attack success; increase max_steps to allow iterative
multi-step attack episodes.

"""
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _ensure_3d(x: np.ndarray) -> np.ndarray:
    """Return array with shape (n_samples, n_features, 1)."""
    if x.ndim == 2:
        return x.reshape(x.shape[0], x.shape[1], 1)
    return x


class AdaptiveNIDSEnv(gym.Env):
    """
    Gymnasium environment that wraps the NIDS defense pipeline for a black-box adaptive attacker.

    Observation space
    - Box(0.0, 1.0, shape=(n_features,)) : A single normalized & reduced sample (flattened)

    Action space
    - Box(-max_epsilon, max_epsilon, shape=(n_features,)) : Per-feature continuous perturbation

    Reward
    - +10.0 on success (model predicts BENIGN, i.e. P(malicious) < 0.5)
    - -1.0 on failure
    - Regularization penalty: -lambda_reg * ||action||_2 applied on every step

    Parameters
    - model: tf.keras.Model — compiled model; forward pass should accept shape (1, n_features, 1)
    - X: np.ndarray — preprocessed dataset of shape (N, n_features, 1)
    - y: np.ndarray — labels array (N,) where 1 == malicious, 0 == benign
    - max_epsilon: float — action bounds per-feature
    - lambda_reg: float — L2 penalty weight for action magnitude
    - smoothing_fn: callable — spatial_smoothing function taking np.ndarray and returning np.ndarray
    - smoothing_window: float — passed to smoothing_fn
    - max_steps: int — maximum steps per episode (default 1 for single-shot attacks)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        max_epsilon: float = 0.05,
        lambda_reg: float = 0.01,
        smoothing_fn=None,
        smoothing_window: float = 1.0,
        max_steps: int = 1,
        rng_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Validate shapes and convert to expected internal shapes
        X = _ensure_3d(X)
        if X.ndim != 3 or X.shape[2] != 1:
            raise ValueError("X must have shape (N, n_features, 1)")

        self.model = model
        self.X = X.astype(np.float32)
        self.y = y
        self.n_samples, self.n_features, _ = self.X.shape

        self.max_epsilon = float(max_epsilon)
        self.lambda_reg = float(lambda_reg)
        self.smoothing_fn = smoothing_fn
        self.smoothing_window = smoothing_window
        self.max_steps = int(max_steps)

        # Gym spaces
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_features,), dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_epsilon, high=self.max_epsilon, shape=(self.n_features,), dtype=np.float32)

        # Episode bookkeeping
        self._step_count = 0
        self._current_idx = None
        self._rng = np.random.RandomState(rng_seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment. Randomly samples a malicious example from X (y==1) and returns its flattened features.

        Returns
        - observation (np.ndarray): flattened sample of shape (n_features,)
        - info (dict)
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        # Pick a random malicious sample; if none exist, pick random sample
        mal_idx = np.where(self.y == 1)[0]
        if len(mal_idx) == 0:
            self._current_idx = int(self._rng.choice(self.n_samples))
        else:
            self._current_idx = int(self._rng.choice(mal_idx))

        self._step_count = 0
        sample = self.X[self._current_idx].reshape(self.n_features)
        return sample.copy(), {}

    def step(self, action: np.ndarray):
        """
        Apply the continuous perturbation action, run Spatial Smoothing (if provided) and query the classifier.

        Returns (observation, reward, terminated, truncated, info)
        - observation: the perturbed (and optionally smoothed) flattened sample
        - reward: scalar as per reward design
        - terminated: True on success or when max_steps reached
        - truncated: True when max_steps reached without success
        - info: dictionary with diagnostics (p_malicious, l2_action, success)
        """
        self._step_count += 1

        # Clip action to the allowed range (safety)
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.max_epsilon, self.max_epsilon)

        # Original sample (flattened) and apply action
        orig = self.X[self._current_idx].reshape(self.n_features)
        perturbed = np.clip(orig + action, 0.0, 1.0)

        # Reshape for smoothing & model (n_features,1) -> (1,n_features,1)
        perturbed_3d = perturbed.reshape(1, self.n_features, 1)

        # Apply spatial smoothing if available
        if self.smoothing_fn is not None:
            smoothed = self.smoothing_fn(perturbed_3d, window_radius=self.smoothing_window)
        else:
            smoothed = perturbed_3d.astype(np.float32)

        # Model query (black-box): expect softmax outputs
        # Ensure batch dim present
        try:
            preds = self.model.predict(smoothed, verbose=0)
        except Exception:
            # As a fallback, try calling the model as a callable (some PyTorch wrappers expose predict differently)
            preds = np.asarray(self.model(smoothed))

        # preds shape -> (1, n_classes)
        p_malicious = float(preds[0][1]) if preds.shape[-1] >= 2 else float(preds[0][0])

        # Reward design
        success = p_malicious < 0.5  # classifier thinks benign
        reward = 10.0 if success else -1.0

        # Regularization penalty
        l2_action = float(np.linalg.norm(action))
        reward -= self.lambda_reg * l2_action

        terminated = success
        truncated = False
        if not terminated and self._step_count >= self.max_steps:
            truncated = True

        obs = smoothed.reshape(self.n_features).astype(np.float32)

        info = {
            "p_malicious": p_malicious,
            "l2_action": l2_action,
            "success": bool(success),
            "original_idx": int(self._current_idx),
        }

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode: str = "human") -> None:
        # Minimal render for debugging
        print(f"AdaptiveNIDSEnv: sample_idx={self._current_idx}, n_features={self.n_features}")

    def close(self) -> None:
        pass
