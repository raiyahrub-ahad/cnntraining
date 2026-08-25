"""Robustness techniques beyond plain FGSM/PGD adversarial training:

- TRADES: TRadeoff-inspired Adversarial DEfense via Surrogate loss
  (Zhang et al., ICML 2019). Optimizes a clean CE loss plus a KL-divergence
  term between clean and adversarial output distributions, giving an explicit,
  tunable accuracy/robustness tradeoff (beta).

- AWP: Adversarial Weight Perturbation (Wu et al., NeurIPS 2020). During
  adversarial training, in addition to perturbing the *input*, the *model
  weights* are perturbed in the direction that most increases the adversarial
  loss before the gradient step — flattens the loss landscape and closes the
  robust generalization gap.

- Randomized Smoothing (Cohen et al., ICML 2019): an inference-time technique
  that classifies the majority vote over many Gaussian-noised copies of the
  input, giving both an empirical accuracy-under-noise number and a certified
  L2 robustness radius (no attack simulation needed for the certificate).

These are exposed as additional training/evaluation options alongside the
standard FGSM-AT / PGD-AT and plain evaluation blocks.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attacks import build_attack

ROBUSTNESS_TECHNIQUES = {
    "trades": "TRADES (KL-regularized robust training, tunable accuracy/robustness tradeoff)",
    "awp": "Adversarial Weight Perturbation on top of PGD-AT (flattens loss landscape)",
}


# ---------------------------------------------------------------------------
# TRADES
# ---------------------------------------------------------------------------

def trades_loss(model, x, y, optimizer, eps=8 / 255, alpha=2 / 255, steps=7, beta=6.0):
    """One TRADES training step. Returns the scalar loss (already used for
    backward+step by the caller's training loop pattern — here we do it inline
    since TRADES needs its own inner PGD-like loop on the KL term)."""
    model.eval()
    batch_size = x.size(0)
    x_adv = x.detach() + 0.001 * torch.randn_like(x)
    criterion_kl = nn.KLDivLoss(reduction="sum")

    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            loss_kl = criterion_kl(
                F.log_softmax(model(x_adv), dim=1),
                F.softmax(model(x), dim=1),
            )
        grad = torch.autograd.grad(loss_kl, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    optimizer.zero_grad()
    logits_clean = model(x)
    loss_natural = F.cross_entropy(logits_clean, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(model(x_adv), dim=1),
        F.softmax(logits_clean, dim=1),
    )
    loss = loss_natural + beta * loss_robust
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        acc = (logits_clean.argmax(1) == y).float().mean().item()
    return loss.item(), acc


# ---------------------------------------------------------------------------
# Adversarial Weight Perturbation (AWP)
# ---------------------------------------------------------------------------

class AWP:
    """Perturbs model weights in the direction that increases the adversarial
    loss, applies one training step, then restores the original weights.
    Wraps around a standard PGD-AT step."""

    def __init__(self, model, gamma=0.01):
        self.model = model
        self.gamma = gamma
        self.backup = {}

    def _filter_params(self):
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad and p.dim() > 1]

    def perturb(self, x_adv, y):
        self.model.zero_grad()
        loss = F.cross_entropy(self.model(x_adv), y)
        grads = torch.autograd.grad(loss, [p for _, p in self._filter_params()], retain_graph=False)
        self.backup = {}
        with torch.no_grad():
            for (name, p), g in zip(self._filter_params(), grads):
                self.backup[name] = p.detach().clone()
                norm_p = p.norm() + 1e-12
                norm_g = g.norm() + 1e-12
                p.add_(self.gamma * norm_p / norm_g * g)

    def restore(self):
        with torch.no_grad():
            for name, p in self._filter_params():
                if name in self.backup:
                    p.copy_(self.backup[name])
        self.backup = {}


def awp_pgd_at_epoch(model, loader, optimizer, device, eps=8 / 255, alpha=2 / 255, steps=7, awp_gamma=0.01):
    """One epoch of PGD adversarial training with AWP. Returns (loss, acc)."""
    model.train()
    awp = AWP(model, gamma=awp_gamma)
    total_loss, total_correct, total_n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        model.eval()
        atk = build_attack("pgd", model, eps=eps, alpha=alpha, steps=steps)
        x_adv = atk(x, y)
        model.train()

        awp.perturb(x_adv, y)
        optimizer.zero_grad()
        out = model(x_adv)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        awp.restore()

        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
        total_n += x.size(0)

    return total_loss / total_n, total_correct / total_n


# ---------------------------------------------------------------------------
# Randomized Smoothing (certified robustness)
# ---------------------------------------------------------------------------

@torch.no_grad()
def randomized_smoothing_eval(model, loader, device, sigma=0.25, n_samples=50, max_batches=None):
    """Empirical accuracy under randomized smoothing (majority vote over
    Gaussian-noised copies), plus an approximate certified L2 radius per
    correctly-classified example using the Cohen et al. formula:
        radius = sigma * Phi^-1(p_A)
    where p_A is the (lower-bound) fraction of noisy copies voting for the
    top class. This is a lightweight, non-conformal approximation intended
    for quick experimentation, not a full certification pipeline."""
    model.eval()
    correct, n = 0, 0
    radii = []

    for bi, (x, y) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        b = x.size(0)
        votes = torch.zeros(b, device=device)
        class_counts = None
        num_classes = None

        for _ in range(n_samples):
            noisy = x + sigma * torch.randn_like(x)
            noisy = torch.clamp(noisy, 0.0, 1.0)
            out = model(noisy)
            pred = out.argmax(1)
            if class_counts is None:
                num_classes = out.shape[1]
                class_counts = torch.zeros(b, num_classes, device=device)
            class_counts[torch.arange(b, device=device), pred] += 1

        top_count, top_class = class_counts.max(dim=1)
        p_a = (top_count / n_samples).clamp(1e-6, 1 - 1e-6)
        radius = sigma * torch.erfinv(2 * p_a - 1) * math.sqrt(2)

        correct += (top_class == y).sum().item()
        n += b
        mask = top_class == y
        radii.extend(radius[mask].detach().cpu().tolist())

    acc = correct / n if n else 0.0
    mean_radius = float(sum(radii) / len(radii)) if radii else 0.0
    return {
        "smoothed_accuracy": acc,
        "mean_certified_radius": mean_radius,
        "sigma": sigma,
        "n_noise_samples": n_samples,
        "n_evaluated": n,
    }
