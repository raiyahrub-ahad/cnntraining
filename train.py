"""Training loops. adv_method selects the training regime:
  None       -> standard clean training
  fgsm_at    -> FGSM adversarial training
  pgd_at     -> PGD adversarial training
  trades     -> TRADES (KL-regularized robust training)
  awp_pgd_at -> PGD-AT with Adversarial Weight Perturbation
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim

from .attacks import build_attack
from .robustness import trades_loss, awp_pgd_at_epoch


def _standard_epoch(model, loader, device, optimizer=None, adv_method=None, adv_params=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_n = 0.0, 0, 0

    atk = None
    if training and adv_method in ("fgsm_at", "pgd_at"):
        base = "fgsm" if adv_method == "fgsm_at" else "pgd"
        atk = build_attack(
            base, model,
            eps=adv_params.get("epsilon", 8 / 255),
            alpha=adv_params.get("alpha", 2 / 255),
            steps=adv_params.get("steps", 7),
        )

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if atk is not None:
            model.eval()
            x = atk(x, y)
            model.train()

        if training:
            optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
        total_n += x.size(0)

    return total_loss / total_n, total_correct / total_n


def _trades_epoch(model, loader, device, optimizer, adv_params):
    total_loss, total_correct, total_n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss, acc = trades_loss(
            model, x, y, optimizer,
            eps=adv_params.get("epsilon", 8 / 255),
            alpha=adv_params.get("alpha", 2 / 255),
            steps=adv_params.get("steps", 7),
            beta=adv_params.get("beta", 6.0),
        )
        total_loss += loss * x.size(0)
        total_correct += acc * x.size(0)
        total_n += x.size(0)
    return total_loss / total_n, total_correct / total_n


def train_model(model, train_loader, test_loader, device, epochs=5, lr=1e-3,
                 adv_method=None, adv_params=None, progress_cb=None):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    adv_params = adv_params or {}
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        if adv_method == "trades":
            tr_loss, tr_acc = _trades_epoch(model, train_loader, device, optimizer, adv_params)
        elif adv_method == "awp_pgd_at":
            tr_loss, tr_acc = awp_pgd_at_epoch(
                model, train_loader, optimizer, device,
                eps=adv_params.get("epsilon", 8 / 255),
                alpha=adv_params.get("alpha", 2 / 255),
                steps=adv_params.get("steps", 7),
                awp_gamma=adv_params.get("awp_gamma", 0.01),
            )
        else:
            tr_loss, tr_acc = _standard_epoch(model, train_loader, device, optimizer, adv_method, adv_params)

        val_loss, val_acc = _standard_epoch(model, test_loader, device, None, None, None)

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if progress_cb:
            progress_cb(epoch, epochs, tr_loss, tr_acc, val_loss, val_acc, time.time() - t0)

    return history
