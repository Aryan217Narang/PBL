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


def wrap_art_classifier(model: nn.Module, n_features: int, n_classes: int = 2, device: str = "cpu"):
    """Wrap a trained PyTorch NIDS_CNN in ART's PyTorchClassifier interface."""
    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        input_shape=(n_features,),
        nb_classes=n_classes,
        clip_values=(0.0, 1.0),  # data was MinMax-scaled to [0,1]
        device_type="gpu" if device == "cuda" else "cpu",
    )
    return classifier


def generate_fgsm(classifier, X, eps: float = 0.1):
    attack = FastGradientMethod(estimator=classifier, eps=eps, batch_size=128)
    return attack.generate(x=X)


def generate_jsma(classifier, X, theta: float = 0.2, gamma: float = 0.15):
    """
    JSMA (Jacobian Saliency Map Attack).
    """
    attack = SaliencyMapMethod(classifier=classifier, theta=theta, gamma=gamma, batch_size=128)
    return attack.generate(x=X)


def generate_cw(classifier, X, confidence: float = 0.0, max_iter: int = 10):
    attack = CarliniL2Method(classifier=classifier, confidence=confidence, max_iter=max_iter, batch_size=128, binary_search_steps=1, initial_const=0.1)
    return attack.generate(x=X)


def generate_all_attacks(classifier, X, subset_size: int = None, seed: int = 42):
    """
    Generate adversarial test sets for all three attacks.
    """
    if subset_size is not None and subset_size < len(X):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=subset_size, replace=False)
        X = X[idx]

    print(f"Generating adversarial samples on {len(X)} inputs...")
    X_fgsm = generate_fgsm(classifier, X)
    print("FGSM done.")
    X_jsma = generate_jsma(classifier, X)
    print("JSMA done.")
    X_cw = generate_cw(classifier, X)
    print("C&W done.")

    return {"fgsm": X_fgsm, "jsma": X_jsma, "cw": X_cw, "clean": X}


if __name__ == "__main__":
    # Example wiring (assumes a trained model + preprocessed test data already exist)
    from models.cnn import build_model
    import numpy as np

    n_features = 20
    model = build_model(n_features=n_features)
    model.load_state_dict(torch.load("models/checkpoints/cnn_clean.pt"))
    model.eval()

    data = np.load("data/processed/processed_data.npz")
    X_test, y_test = data["X_test"], data["y_test"]

    classifier = wrap_art_classifier(model, n_features=n_features)
    adv_sets = generate_all_attacks(classifier, X_test, subset_size=2000)

    np.savez("data/processed/adversarial_test_sets.npz", **adv_sets, y=y_test[:2000])
