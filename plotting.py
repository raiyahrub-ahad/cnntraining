import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_training_curves(history, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title(f"Loss {title_suffix}"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["epoch"], history["train_acc"], label="train")
    axes[1].plot(history["epoch"], history["val_acc"], label="val")
    axes[1].set_title(f"Accuracy {title_suffix}"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_accuracy_bar(results: dict, title="Clean vs. Adversarial Accuracy"):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    names = list(results.keys())
    vals = [results[k] * 100 for k in names]
    bars = ax.bar(names, vals, color="#1f4e8c")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    return _fig_to_base64(fig)
