"""
Train an adaptive black-box RL attacker (PPO) against the NIDS defense pipeline.

The script expects:
- A preprocessed dataset cache produced by `preprocessing.load_and_preprocess()` (or run `main.py`)
  which writes `data/processed_cicddos2019.npz` (the preprocessing.py's CACHE_FILE).
- A trained TensorFlow 1D-CNN model instance. The easiest workflow is:
    1) Train the model using `python main.py` (or load your saved model weights).
    2) In a Python session, load the model architecture from `code/model.py` using build_cnn_model and
       then call model.load_weights("path/to/weights.h5") or model.load_weights on a saved checkpoint.
    3) Pass the model object into this script by editing the bottom example `if __name__ == '__main__'` block.

Important: This script focuses on clarity and compatibility on CPU-only machines. It uses stable-baselines3 (PPO)
with a DummyVecEnv wrapper for training.

Example usage (simplest, interactive):
    - Edit the `LOAD_MODEL_PATH` and `CACHE_PATH` below, then run:
        python code/train_rl_attacker.py

Outputs
- Trained RL policy saved to `models/rl_attacker_policy.zip`
- Training logs printed to stdout

"""
from typing import Optional, Dict
import os
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Local imports
from rl_env import AdaptiveNIDSEnv
from defense import spatial_smoothing
from model import build_cnn_model


class EvalCallback(BaseCallback):
    """Very small callback that evaluates current policy on a small held-out set and prints metrics."""

    def __init__(self, eval_env: gym.Env, eval_episodes: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_episodes = int(eval_episodes)

    def _on_step(self) -> bool:
        # Run a short evaluation every n calls; keep cheap
        if self.n_calls % 500 == 0:
            # compute ASR over eval_episodes
            successes = 0
            total_l2 = 0.0
            for _ in range(self.eval_episodes):
                obs, _ = self.eval_env.reset()
                action, _state = self.model.predict(obs, deterministic=True)
                _, _, terminated, truncated, info = self.eval_env.step(action)
                success = info.get("success", False)
                if success:
                    successes += 1
                total_l2 += info.get("l2_action", 0.0)
            asr = successes / float(self.eval_episodes)
            avg_l2 = total_l2 / float(self.eval_episodes)
            print(f"[EvalCallback] n_calls={self.n_calls} ASR={asr:.3f} avg_l2={avg_l2:.4f}")
        return True


def make_env_from_cache(
    model,
    cache_path: str = "data/processed_cicddos2019.npz",
    max_epsilon: float = 0.05,
    lambda_reg: float = 0.01,
    smoothing_window: float = 1.0,
    max_steps: int = 1,
    seed: Optional[int] = None,
) -> gym.Env:
    """
    Utility to load preprocessed data from the cache and build the AdaptiveNIDSEnv.

    Returns a plain (unvectorized) env; training script will wrap it with DummyVecEnv.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Preprocessed cache not found: {cache_path}. Run preprocessing/main pipeline first.")

    data = np.load(cache_path)
    X_test = data["X_test"]
    y_test = data["y_test"]

    env = AdaptiveNIDSEnv(
        model=model,
        X=X_test,
        y=y_test,
        max_epsilon=max_epsilon,
        lambda_reg=lambda_reg,
        smoothing_fn=spatial_smoothing,
        smoothing_window=smoothing_window,
        max_steps=max_steps,
        rng_seed=seed,
    )
    return env


def train_rl_attacker(
    model,
    out_path: str = "models/rl_attacker_policy.zip",
    total_timesteps: int = 20000,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    batch_size: int = 64,
    max_epsilon: float = 0.05,
    lambda_reg: float = 0.01,
    smoothing_window: float = 1.0,
    seed: Optional[int] = 42,
) -> Dict[str, str]:
    """
    Train a PPO agent to learn perturbations that bypass the Spatial Smoothing + 1D-CNN defense.

    Returns a dict with path to saved policy and basic training settings.
    """
    # Build env
    env_fn = lambda: make_env_from_cache(
        model=model,
        cache_path="data/processed_cicddos2019.npz",
        max_epsilon=max_epsilon,
        lambda_reg=lambda_reg,
        smoothing_window=smoothing_window,
        max_steps=1,
        seed=seed,
    )

    vec_env = DummyVecEnv([env_fn])

    policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])

    model_sb3 = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        seed=seed,
    )

    eval_env = env_fn()
    callback = EvalCallback(eval_env=eval_env, eval_episodes=100)

    print(f"Starting RL training: total_timesteps={total_timesteps}")
    model_sb3.learn(total_timesteps=total_timesteps, callback=callback)

    os.makedirs(os.path.dirname(out_path) or "models", exist_ok=True)
    model_sb3.save(out_path)
    print(f"Saved RL policy to: {out_path}")

    return {"policy_path": out_path}


if __name__ == "__main__":
    # Example quick-start: build model architecture and (optionally) load weights.
    # Edit the paths below to point to your trained CNN weights if available.

    # Paths you may want to change
    LOAD_MODEL_PATH = None  # e.g. "models/checkpoints/cnn_hybrid_weights.h5" (Keras HDF5) or a saved TF checkpoint
    CACHE_PATH = "data/processed_cicddos2019.npz"

    # Build a fresh CNN with the correct input shape by reading the cache
    if not os.path.exists(CACHE_PATH):
        raise RuntimeError(f"Cache not found: {CACHE_PATH} - run preprocessing/main.py first to create it.")

    data = np.load(CACHE_PATH)
    X_train = data.get("X_train")
    if X_train is None:
        raise RuntimeError("X_train not found inside cache; ensure preprocessing saved X_train in the cache.")

    input_shape = tuple(X_train.shape[1:])  # (n_features, 1)
    tf_model = build_cnn_model(input_shape=input_shape, n_classes=2)

    if LOAD_MODEL_PATH is not None and os.path.exists(LOAD_MODEL_PATH):
        print(f"Loading model weights from: {LOAD_MODEL_PATH}")
        try:
            tf_model.load_weights(LOAD_MODEL_PATH)
        except Exception as e:
            print(f"Failed to load weights using Keras load_weights(): {e}. Ensure the path is correct and the weights match the architecture.")

    # Train RL attacker (short default schedule - increase timesteps for research-quality runs)
    train_rl_attacker(
        model=tf_model,
        out_path="models/rl_attacker_policy.zip",
        total_timesteps=15000,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        max_epsilon=0.05,
        lambda_reg=0.02,
        smoothing_window=1.0,
        seed=42,
    )
