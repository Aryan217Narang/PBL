"""
CNN-based NIDS classifier (Barik & Misra, 2025):
5 layers, ReLU hidden activations, softmax output, Adam optimizer, batch size 32.

Input: tabular flow-feature vector (post RFE, e.g. length 20) reshaped to 1D "signal"
for Conv1d layers (a common way to apply CNNs to tabular NIDS features).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NIDS_CNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 2):
        super().__init__()
        self.n_features = n_features

        # Layer 1-2: Conv blocks
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Layer 3: Conv block
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        flattened_dim = 128 * (n_features // 2)  # after one pool of stride 2

        # Layer 4-5: Fully connected
        self.fc1 = nn.Linear(flattened_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, n_classes)  # softmax applied via CrossEntropyLoss / F.softmax at inference

    def forward(self, x):
        # x: (batch, n_features) -> (batch, 1, n_features)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits  # raw logits; use F.softmax(logits, dim=1) for probabilities

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


def build_model(n_features: int, n_classes: int = 2, device: str = "cpu") -> NIDS_CNN:
    model = NIDS_CNN(n_features=n_features, n_classes=n_classes).to(device)
    return model


def train_model(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs: int = 30,
    batch_size: int = 32,   # matches paper
    lr: float = 1e-3,
    device: str = "cpu",
):
    """Standard clean training loop. For adversarial hardening, see defenses/pgd_training.py"""
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer, per paper
    criterion = nn.CrossEntropyLoss()  # combined with softmax output

    model.train()
    for epoch in range(epochs):
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            n += xb.size(0)

        msg = f"Epoch {epoch+1}/{epochs} - loss: {total_loss/n:.4f} - acc: {correct/n:.4f}"
        if X_val is not None:
            val_acc = evaluate_accuracy(model, X_val, y_val, device=device)
            msg += f" - val_acc: {val_acc:.4f}"
        print(msg)

    return model


def evaluate_accuracy(model, X, y, device="cpu"):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        yt = torch.tensor(y, dtype=torch.long).to(device)
        logits = model(Xt)
        preds = logits.argmax(1)
        return (preds == yt).float().mean().item()
