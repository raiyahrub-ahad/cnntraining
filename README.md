# CNN Adversarial Robustness Research Platform

A local full-stack research tool: load a dataset, train CNNs, attack them,
train them to be robust, splice in architectural enhancements, and get every
number and plot a paper needs — clean/adversarial accuracy, confidence
intervals, statistical significance, Grad-CAM, FLOPs/params/latency — all as
independent "one-click" blocks you can also chain into a pipeline.

Runs entirely on **your machine** (GPU strongly recommended for anything
beyond MNIST/CIFAR-scale experiments) — nothing is sent to any external
service except an optional call to the Kaggle API to download a dataset you
name.

## What's inside

**Datasets** — built-in (CIFAR-10/100, MNIST, FashionMNIST), Kaggle API
download, a list of image URLs + labels, or an uploaded ZIP (class-per-folder
layout).

**Models** — SimpleCNN (fast baseline/surrogate), ResNet-18/34, VGG-16,
MobileNetV2, EfficientNet-B0.

**Architectural enhancements** (spliced onto any backbone, one click) —
Pyramid Pooling Module (PSPNet-style), Adaptive Pyramid Pooling (learned
scale-gating), and a Feature Denoising block (non-local means, after
*Xie et al., "Feature Denoising for Improving Adversarial Robustness"*,
CVPR 2019) — a genuinely robustness-oriented architectural technique rather
than just a capacity add-on.

**Attacks**
- White-box: FGSM, I-FGSM/BIM, PGD, AutoAttack
- Black-box: Square Attack (query-based), Transfer Attack (crafted on a
  surrogate model, evaluated on the target with no gradient access)

**Robust training**
- Standard FGSM-AT / PGD-AT
- **TRADES** (Zhang et al., ICML 2019) — KL-regularized robust training with
  a tunable accuracy/robustness tradeoff (`beta`)
- **PGD-AT + Adversarial Weight Perturbation (AWP)** (Wu et al., NeurIPS
  2020) — perturbs the *weights*, not just the input, during training to
  flatten the loss landscape and reduce the robust generalization gap

**Certified robustness** — Randomized Smoothing (Cohen et al., ICML 2019):
majority vote over Gaussian-noised copies, giving both an empirical
"smoothed accuracy" and an approximate certified L2 radius, no attack
simulation required for the certificate.

**Analysis** — Grad-CAM (clean or under a chosen attack, auto-locates the
last conv layer of whatever model is active), bootstrap 95% confidence
intervals on every accuracy number, paired McNemar significance tests
(clean vs. each attack), FLOPs/params (via `thop`), and inference latency.

**Report** — one click bundles everything computed so far (setup, metrics,
accuracy table, certified radius, statistics, plots, Grad-CAM gallery) into
a single downloadable HTML report.

## How blocks work

Every block (Train, Attack, Grad-CAM, ...) is its own API call that returns
a `job_id` immediately and runs in a background thread. That means:

- **Click any block on its own** — it runs right away.
- **Click several blocks in a row** — they run *concurrently*, each in its
  own thread (careful: two blocks that both call `model.train()`/`eval()`
  on the *same* model at the same time will interfere with each other —
  training and evaluation blocks are best run one at a time; independent
  blocks like Grad-CAM after training and Metrics can safely overlap).
- **Use the Pipeline block** to tick several steps and run them as one
  *ordered, sequential* job — the safe way to chain e.g. "train → evaluate →
  attack → stats" in one click.

## Setup

Requires **Python 3.10+** and, for real experiments, an **NVIDIA GPU with
CUDA** (CPU works for MNIST-scale smoke tests but adversarial training/attacks
are slow without a GPU).

### 1. Install PyTorch for your GPU (recommended, do this first)

Go to https://pytorch.org/get-started/locally/, pick your OS/CUDA version,
and run the command it gives you, e.g.:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If you skip this, `pip install -r requirements.txt` will install a CPU-only
build of PyTorch.

### 2. One-command setup + run

**Linux / macOS:**
```bash
chmod +x run.sh
./run.sh
```

**Windows:**
```bat
run.bat
```

This creates a virtual environment, installs everything in
`requirements.txt`, and starts the server. Then open:

```
http://localhost:8000
```

### 3. (Optional) Kaggle API credentials

To use the "Kaggle API" dataset tab, you need a Kaggle account API token:
Kaggle → Account → "Create New API Token" → downloads `kaggle.json`
containing your `username` and `key`. Paste those two values directly into
the Kaggle tab in the UI — they're only used server-side, for that request,
to call the Kaggle API.

## Project layout

```
backend/
  main.py          FastAPI app — every block is one endpoint
  models.py        CNN model zoo (SimpleCNN, ResNet, VGG, MobileNet, EfficientNet)
  enhancements.py  Pyramid Pooling / Adaptive PPM / Feature Denoising
  attacks.py       FGSM / I-FGSM / PGD / AutoAttack / Square / Transfer
  robustness.py    TRADES, Adversarial Weight Perturbation, Randomized Smoothing
  train.py         Clean + adversarial/robust training loops
  evaluate.py       Clean/adversarial accuracy, bootstrap CI, McNemar test
  gradcam.py       Grad-CAM (auto-locates last conv layer)
  metrics.py       FLOPs / params / inference latency
  plotting.py      Training curves, accuracy comparison bar charts
  data.py          Built-in / Kaggle / URL-list / ZIP-upload dataset loading
  report.py        HTML research report generator
  state.py         In-memory app state (single-user, local tool)
  jobs.py          Background job manager (each block runs in its own thread)
  schemas.py       Request models for every endpoint
frontend/
  index.html       Dashboard — one section per block
  style.css
  app.js           Fetch calls, job polling, dynamic attack grid, results dashboard
requirements.txt
run.sh / run.bat
```

## Notes on how experiments are set up

- All images are loaded as plain `ToTensor()` — pixel values in `[0,1]`, no
  extra normalization — so epsilon budgets (e.g. `8/255 ≈ 0.031`) are
  directly meaningful in pixel space, and `torchattacks` (which assumes
  `[0,1]` inputs) works out of the box.
- Selecting a model creates a fresh `SimpleCNN` as the fixed surrogate for
  Transfer attacks, so a black-box transfer attack always uses a genuinely
  different architecture than the target.
- The Results Dashboard and the exported HTML report both read from the same
  in-memory state, so anything you've run appears in the final report —
  run the blocks you want included before exporting.
- This is a single-user local tool: state is a global in-memory object, not
  a database, so a server restart clears trained models/results.

## Extending it

- New backbone: add a branch in `models.py::build_model` and
  `get_feature_extractor`.
- New attack: add a case in `attacks.py::build_attack` (most things in
  `torchattacks` slot in directly).
- New robustness technique: add a function to `robustness.py`, wire it into
  `train.py::train_model`'s `adv_method` branch, add it to the
  `adv-method` dropdown in `index.html`.
- New architectural enhancement: subclass `nn.Module` in `enhancements.py`
  taking `(x: B,C,H,W) -> B,C,H,W` (or adjust `EnhancedCNN` if it changes
  channel count), register it in `ENHANCEMENT_REGISTRY`.
