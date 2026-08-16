"""
PGD (Projected Gradient Descent) adversarial training - the training-phase
defense in Barik & Misra (2025). Hardens the CNN by training on a mix of
clean and PGD-perturbed samples generated on-the-fly each epoch.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def pgd_attack_batch(model, X, y, eps: float, alpha: float, n_iter: int, device: str = "cpu"):
    """
    Generate PGD adversarial examples for a batch, in-graph (no ART dependency),
    since this runs inside the training loop every batch/epoch.
    """
    criterion = nn.CrossEntropyLoss()
    X_adv = X.clone().detach().to(device)
    X_adv = X_adv + torch.empty_like(X_adv).uniform_(-eps, eps)
    X_adv = torch.clamp(X_adv, 0.0, 1.0)

    for _ in range(n_iter):
        X_adv.requires_grad_(True)
        logits = model(X_adv)
        loss = criterion(logits, y)
        grad = torch.autograd.grad(loss, X_adv)[0]

        X_adv = X_adv.detach() + alpha * grad.sign()
        # project back into eps-ball around original X, then clip to valid range
        delta = torch.clamp(X_adv - X, min=-eps, max=eps)
        X_adv = torch.clamp(X + delta, 0.0, 1.0).detach()

    return X_adv


def pgd_adversarial_train(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    eps: float = 0.1,
    alpha: float = 0.02,
    n_iter: int = 7,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    adv_ratio: float = 0.5,  # fraction of each batch replaced with adversarial examples
    device: str = "cpu",
):
    """
    Train the CNN adversarially: each batch is a mix of clean + PGD examples.
    (eps, alpha, n_iter) are exactly the hyperparameters PIOA optimizes -
    see defenses/pioa.py.
    """
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            n_adv = int(len(xb) * adv_ratio)
            if n_adv > 0:
                x_adv_part = pgd_attack_batch(model, xb[:n_adv], yb[:n_adv], eps, alpha, n_iter, device)
                xb_mixed = torch.cat([x_adv_part, xb[n_adv:]], dim=0)
            else:
                xb_mixed = xb

            optimizer.zero_grad()
            logits = model(xb_mixed)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            n += xb.size(0)

        msg = f"[PGD-train] Epoch {epoch+1}/{epochs} - loss: {total_loss/n:.4f} - acc: {correct/n:.4f}"
        if X_val is not None:
            from models.cnn import evaluate_accuracy
            val_acc = evaluate_accuracy(model, X_val, y_val, device=device)
            msg += f" - val_acc: {val_acc:.4f}"
        print(msg)

    return model
