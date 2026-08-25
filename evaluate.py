"""Clean and adversarial evaluation, plus statistical analysis: bootstrap
confidence intervals for accuracy, and a paired McNemar test comparing two
prediction sets (e.g. clean vs. under a given attack) on the same samples."""

import math
import numpy as np
import torch

from .attacks import build_attack, transfer_attack_examples


@torch.no_grad()
def clean_accuracy(model, loader, device, max_batches=None):
    model.eval()
    correct, n = 0, 0
    all_preds, all_labels = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        n += x.size(0)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    acc = correct / n if n else 0.0
    return acc, np.concatenate(all_preds), np.concatenate(all_labels)


def bootstrap_ci(preds, labels, n_boot=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(labels)
    correct = (preds == labels).astype(np.float64)
    accs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        accs[i] = correct[idx].mean()
    lo = float(np.quantile(accs, alpha / 2))
    hi = float(np.quantile(accs, 1 - alpha / 2))
    return lo, hi


def mcnemar_test(preds_a, preds_b, labels):
    """Paired McNemar's test (with continuity correction) comparing whether
    two models/conditions differ significantly on the same samples."""
    a_correct = preds_a == labels
    b_correct = preds_b == labels
    n01 = int(np.sum(a_correct & ~b_correct))  # a right, b wrong
    n10 = int(np.sum(~a_correct & b_correct))  # a wrong, b right
    if n01 + n10 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n01": n01, "n10": n10}
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = math.erfc(math.sqrt(stat / 2))  # chi-square(df=1) survival function
    return {"statistic": float(stat), "p_value": float(p_value), "n01": n01, "n10": n10}


def adversarial_accuracy(model, loader, device, attack_name, eps, alpha, steps, n_classes,
                          max_samples=500, surrogate_model=None, collect_examples=8):
    model.eval()
    correct, n = 0, 0
    all_preds, all_labels = [], []
    examples = []

    atk = None
    if attack_name != "transfer":
        atk = build_attack(attack_name, model, eps=eps, alpha=alpha, steps=steps, n_classes=n_classes)

    for x, y in loader:
        if n >= max_samples:
            break
        x, y = x.to(device), y.to(device)

        if attack_name == "transfer":
            x_adv = transfer_attack_examples(surrogate_model, model, x, y, eps, alpha, steps, device)
        else:
            x_adv = atk(x, y)

        with torch.no_grad():
            out = model(x_adv)
            pred = out.argmax(1)

        correct += (pred == y).sum().item()
        n += x.size(0)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(y.cpu().numpy())

        if len(examples) < collect_examples:
            for i in range(min(x.size(0), collect_examples - len(examples))):
                examples.append({
                    "clean": x[i].detach().cpu(),
                    "adv": x_adv[i].detach().cpu(),
                    "true": int(y[i].item()),
                    "pred": int(pred[i].item()),
                })

    acc = correct / n if n else 0.0
    return acc, np.concatenate(all_preds), np.concatenate(all_labels), examples
