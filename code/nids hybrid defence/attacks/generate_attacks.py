"""
Adversarial attack generation against the trained NIDS CNN, using the
Adversarial Robustness Toolbox (ART): FGSM, JSMA, and C&W - matching the
three attacks evaluated in Barik & Misra (2025).

pip install adversarial-robustness-toolbox
"""

import numpy as np
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, SaliencyMapMethod, CarliniL2Method


def wrap_art_classifier(model: nn.Module, n_features: int, n_classes: int = 2, device: str = "cpu", clip_values=None):
    """Wrap a trained PyTorch NIDS_CNN in ART's PyTorchClassifier interface."""
    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        input_shape=(n_features,),
        nb_classes=n_classes,
        clip_values=clip_values,
        device_type="gpu" if device == "cuda" else "cpu",
    )
    return classifier


def generate_fgsm(classifier, X, eps: float = 0.1):
    attack = FastGradientMethod(estimator=classifier, eps=eps, batch_size=128)
    return attack.generate(x=X)


def generate_jsma(classifier, X, theta: float = 0.2, gamma: float = 0.15):
    """JSMA (Jacobian Saliency Map Attack)."""
    attack = SaliencyMapMethod(classifier=classifier, theta=theta, gamma=gamma, batch_size=128)
    return attack.generate(x=X)


def generate_cw(classifier, X, confidence: float = 0.0, max_iter: int = 10):
    attack = CarliniL2Method(
        classifier=classifier,
        confidence=confidence,
        max_iter=max_iter,
        batch_size=128,
        binary_search_steps=1,
        initial_const=0.1
    )
    return attack.generate(x=X)


def generate_all_attacks(
    classifier,
    X,
    y=None,
    subset_size: int = None,
    sequential: bool = True,
    seed: int = 42,
):
    """
    Generate adversarial test sets for all three attacks.
    When sequential=True, preserves chronological temporal stream order (X[:subset_size]).
    Always returns matching y_subset so labels are never scrambled.
    """
    if subset_size is not None and subset_size < len(X):
        if sequential:
            # Preserves chronological flow ordering
            X = X[:subset_size]
            if y is not None:
                y = y[:subset_size]
        else:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X), size=subset_size, replace=False)
            X = X[idx]
            if y is not None:
                y = y[idx]

    X = X.astype(np.float32)
    print(f"Generating adversarial samples on {len(X)} inputs...")
    X_fgsm = generate_fgsm(classifier, X).astype(np.float32)
    print("FGSM done.")
    X_jsma = generate_jsma(classifier, X).astype(np.float32)
    print("JSMA done.")
    X_cw = generate_cw(classifier, X).astype(np.float32)
    print("C&W done.")

    result = {"fgsm": X_fgsm, "jsma": X_jsma, "cw": X_cw, "clean": X}
    if y is not None:
        result["y"] = y
    return result
