"""Adversarial attack registry: white-box (FGSM, I-FGSM/BIM, PGD, AutoAttack)
and black-box (Square attack — query-based; transfer attack — surrogate-based).
Built on the `torchattacks` library. All attacks assume inputs in [0,1]
(no external Normalize layer — datasets are loaded with plain ToTensor)."""

import torchattacks

WHITEBOX_ATTACKS = ["fgsm", "ifgsm", "pgd", "autoattack"]
BLACKBOX_ATTACKS = ["square", "transfer"]
ALL_ATTACKS = WHITEBOX_ATTACKS + BLACKBOX_ATTACKS

ATTACK_INFO = {
    "fgsm": {"label": "FGSM", "type": "white-box", "desc": "Fast Gradient Sign Method — single-step."},
    "ifgsm": {"label": "I-FGSM / BIM", "type": "white-box", "desc": "Iterative FGSM (Basic Iterative Method)."},
    "pgd": {"label": "PGD", "type": "white-box", "desc": "Projected Gradient Descent — strong iterative attack."},
    "autoattack": {"label": "AutoAttack", "type": "white-box", "desc": "Ensemble of parameter-free attacks (APGD-CE/T, FAB, Square)."},
    "square": {"label": "Square Attack", "type": "black-box", "desc": "Query-based, gradient-free black-box attack."},
    "transfer": {"label": "Transfer Attack", "type": "black-box", "desc": "Crafted on a surrogate model, evaluated on the target (no target gradients used)."},
}


def build_attack(name, model, eps=8 / 255, alpha=2 / 255, steps=10, n_classes=10):
    name = name.lower()
    if name == "fgsm":
        return torchattacks.FGSM(model, eps=eps)
    if name == "ifgsm":
        return torchattacks.BIM(model, eps=eps, alpha=alpha, steps=steps)
    if name == "pgd":
        return torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=True)
    if name == "autoattack":
        return torchattacks.AutoAttack(model, eps=eps, n_classes=n_classes, version="standard")
    if name == "square":
        return torchattacks.Square(model, eps=eps, n_queries=1000)
    raise ValueError(f"'{name}' cannot be built directly (use transfer_attack_examples for transfer).")


def transfer_attack_examples(surrogate_model, target_model, x, y, eps=8 / 255, alpha=2 / 255, steps=10, device="cpu"):
    """Black-box transfer attack: craft PGD examples on a surrogate model that
    the target never exposes gradients to, then evaluate them against the
    target model. Simulates a realistic black-box threat model."""
    atk = torchattacks.PGD(surrogate_model, eps=eps, alpha=alpha, steps=steps, random_start=True)
    return atk(x, y)
