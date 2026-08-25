import os
import time
import shutil
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from . import data as data_mod
from . import models as models_mod
from . import enhancements as enh_mod
from . import train as train_mod
from . import evaluate as eval_mod
from . import gradcam as gradcam_mod
from . import metrics as metrics_mod
from . import plotting as plot_mod
from . import report as report_mod
from . import robustness as robust_mod
from .attacks import build_attack, ATTACK_INFO
from .schemas import (
    BuiltinDatasetReq, KaggleDatasetReq, UrlDatasetReq, ModelSelectReq,
    ModelEnhanceReq, TrainCleanReq, TrainAdvReq, AttackReq, GradCamReq,
    SmoothingReq, PipelineReq,
)
from .state import STATE
from .jobs import create_job, get_job, run_in_background

app = FastAPI(title="CNN Adversarial Robustness Research Platform")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Fixed: Use the root directory where index.html, style.css, and app.js are located
FRONTEND_DIR = os.path.dirname(os.path.dirname(__file__))


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(FRONTEND_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/api/registry")
def registry():
    return {
        "models": models_mod.MODEL_REGISTRY,
        "enhancements": enh_mod.ENHANCEMENT_REGISTRY,
        "attacks": ATTACK_INFO,
        "robustness_techniques": robust_mod.ROBUSTNESS_TECHNIQUES,
        "datasets": list(data_mod.BUILTIN_DATASETS.keys()),
    }


@app.get("/api/state")
def get_state():
    return STATE.summary()


@app.get("/api/plots/{name}")
def get_plot(name: str):
    if name not in STATE.plots:
        raise HTTPException(404, "plot not found")
    return {"name": name, "image_b64": STATE.plots[name]}


@app.get("/api/gradcam/results")
def get_gradcam_results():
    return {"examples": STATE.gradcam_results}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job.to_dict()


# ============================== DATASET ==================================

@app.post("/api/dataset/builtin")
def dataset_builtin(req: BuiltinDatasetReq):
    job = create_job("dataset_builtin")

    def run(job):
        job.message = f"Loading {req.name}..."
        train_loader, test_loader, n_classes, ch, size = data_mod.load_builtin(
            req.name, req.img_size, req.batch_size)
        with STATE.lock:
            STATE.dataset_name = req.name
            STATE.train_loader = train_loader
            STATE.test_loader = test_loader
            STATE.num_classes = n_classes
            STATE.in_channels = ch
            STATE.img_size = size
        STATE.log(f"Loaded builtin dataset {req.name}: {n_classes} classes, {size}x{size}x{ch}")
        return {"dataset": req.name, "num_classes": n_classes, "img_size": size, "channels": ch}

    run_in_background(job, run)
    return {"job_id": job.id}


@app.post("/api/dataset/kaggle")
def dataset_kaggle(req: KaggleDatasetReq):
    job = create_job("dataset_kaggle")

    def run(job):
        job.message = f"Downloading Kaggle dataset {req.slug}..."
        dest = os.path.join(data_mod.DATA_ROOT, "kaggle", req.slug.replace("/", "_"))
        path = data_mod.download_kaggle_dataset(req.slug, req.username, req.key, dest)
        job.message = "Building dataloaders..."
        train_loader, test_loader, n_classes, ch, size = data_mod.load_image_folder(
            path, req.img_size, req.batch_size, req.val_split)
        with STATE.lock:
            STATE.dataset_name = req.slug
            STATE.train_loader = train_loader
            STATE.test_loader = test_loader
            STATE.num_classes = n_classes
            STATE.in_channels = ch
            STATE.img_size = size
        STATE.log(f"Loaded Kaggle dataset {req.slug}: {n_classes} classes")
        return {"dataset": req.slug, "num_classes": n_classes, "img_size": size, "channels": ch}

    run_in_background(job, run)
    return {"job_id": job.id}


@app.post("/api/dataset/urls")
def dataset_urls(req: UrlDatasetReq):
    job = create_job("dataset_urls")

    def run(job):
        job.message = "Downloading images from links..."
        dest = os.path.join(data_mod.DATA_ROOT, "url_dataset")
        items = [{"url": i.url, "label": i.label} for i in req.items]
        path = data_mod.download_images_from_urls(items, dest)
        job.message = "Building dataloaders..."
        train_loader, test_loader, n_classes, ch, size = data_mod.load_image_folder(
            path, req.img_size, req.batch_size, req.val_split)
        with STATE.lock:
            STATE.dataset_name = "custom_urls"
            STATE.train_loader = train_loader
            STATE.test_loader = test_loader
            STATE.num_classes = n_classes
            STATE.in_channels = ch
            STATE.img_size = size
        STATE.log(f"Loaded URL dataset: {n_classes} classes")
        return {"dataset": "custom_urls", "num_classes": n_classes, "img_size": size, "channels": ch}

    run_in_background(job, run)
    return {"job_id": job.id}


@app.post("/api/dataset/upload")
async def dataset_upload(file: UploadFile = File(...), img_size: int = 128,
                          batch_size: int = 32, val_split: float = 0.2):
    upload_dir = os.path.join(data_mod.DATA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    zip_path = os.path.join(upload_dir, file.filename)
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job = create_job("dataset_upload")

    def run(job):
        job.message = "Extracting archive..."
        extract_to = os.path.join(data_mod.DATA_ROOT, "uploaded_dataset")
        path = data_mod.extract_zip_dataset(zip_path, extract_to)
        job.message = "Building dataloaders..."
        train_loader, test_loader, n_classes, ch, size = data_mod.load_image_folder(
            path, img_size, batch_size, val_split)
        with STATE.lock:
            STATE.dataset_name = file.filename
            STATE.train_loader = train_loader
            STATE.test_loader = test_loader
            STATE.num_classes = n_classes
            STATE.in_channels = ch
            STATE.img_size = size
        STATE.log(f"Loaded uploaded dataset {file.filename}: {n_classes} classes")
        return {"dataset": file.filename, "num_classes": n_classes, "img_size": size, "channels": ch}

    run_in_background(job, run)
    return {"job_id": job.id}


# ================================ MODEL ===================================

@app.post("/api/model/select")
def model_select(req: ModelSelectReq):
    if STATE.num_classes is None:
        raise HTTPException(400, "Load a dataset first.")
    backbone = models_mod.build_model(req.name, STATE.num_classes, STATE.in_channels, req.pretrained)
    with STATE.lock:
        STATE.backbone_name = req.name
        STATE.backbone = backbone
        STATE.model = backbone
        STATE.enhancement = "none"
        STATE.surrogate_model = models_mod.build_model("simplecnn", STATE.num_classes, STATE.in_channels).to(STATE.device)
    n_params = sum(p.numel() for p in backbone.parameters())
    STATE.log(f"Selected backbone {req.name} ({n_params / 1e6:.2f}M params)")
    return {"backbone": req.name, "params": n_params}


@app.post("/api/model/enhance")
def model_enhance(req: ModelEnhanceReq):
    if STATE.backbone is None:
        raise HTTPException(400, "Select a model first.")
    model = enh_mod.apply_enhancement(STATE.backbone, STATE.backbone_name, STATE.num_classes, req.kind)
    with STATE.lock:
        STATE.enhancement = req.kind
        STATE.model = model.to(STATE.device)
    n_params = sum(p.numel() for p in model.parameters())
    STATE.log(f"Applied enhancement={req.kind} ({n_params / 1e6:.2f}M params)")
    return {"enhancement": req.kind, "params": n_params}


# =============================== TRAINING =================================

@app.post("/api/train/clean")
def train_clean(req: TrainCleanReq):
    if STATE.model is None or STATE.train_loader is None:
        raise HTTPException(400, "Load a dataset and select a model first.")
    job = create_job("train_clean")

    def progress_cb(epoch, epochs, tr_loss, tr_acc, val_loss, val_acc, dt):
        job.progress = epoch / epochs
        job.message = f"Epoch {epoch}/{epochs} — val_acc={val_acc * 100:.1f}%"
        job.log_lines.append(job.message)

    def run(job):
        history = train_mod.train_model(
            STATE.model, STATE.train_loader, STATE.test_loader, STATE.device,
            epochs=req.epochs, lr=req.lr, progress_cb=progress_cb)
        with STATE.lock:
            STATE.clean_history = history
            STATE.plots["training_curves_clean"] = plot_mod.plot_training_curves(history, "(clean)")
        STATE.log("Clean training complete.")
        return {"history": history}

    run_in_background(job, run)
    return {"job_id": job.id}


@app.post("/api/train/adversarial")
def train_adversarial(req: TrainAdvReq):
    if STATE.model is None or STATE.train_loader is None:
        raise HTTPException(400, "Load a dataset and select a model first.")
    job = create_job("train_adversarial")

    def progress_cb(epoch, epochs, tr_loss, tr_acc, val_loss, val_acc, dt):
        job.progress = epoch / epochs
        job.message = f"[{req.method}] Epoch {epoch}/{epochs} — val_acc={val_acc * 100:.1f}%"
        job.log_lines.append(job.message)

    def run(job):
        history = train_mod.train_model(
            STATE.model, STATE.train_loader, STATE.test_loader, STATE.device,
            epochs=req.epochs, lr=req.lr, adv_method=req.method,
            adv_params={"epsilon": req.epsilon, "alpha": req.alpha, "steps": req.steps,
                        "beta": req.beta, "awp_gamma": req.awp_gamma},
            progress_cb=progress_cb)
        with STATE.lock:
            STATE.adv_train_history[req.method] = history
            STATE.plots[f"training_curves_{req.method}"] = plot_mod.plot_training_curves(history, f"({req.method})")
        STATE.log(f"Adversarial/robust training ({req.method}) complete.")
        return {"history": history}

    run_in_background(job, run)
    return {"job_id": job.id}


# ============================== EVALUATION =================================

@app.post("/api/eval/clean")
def eval_clean():
    if STATE.model is None or STATE.test_loader is None:
        raise HTTPException(400, "Load a dataset and select a model first.")
    job = create_job("eval_clean")

    def run(job):
        acc, preds, labels = eval_mod.clean_accuracy(STATE.model, STATE.test_loader, STATE.device)
        lo, hi = eval_mod.bootstrap_ci(preds, labels)
        with STATE.lock:
            STATE.clean_eval = {"accuracy": acc, "ci_low": lo, "ci_high": hi}
            STATE.clean_preds = preds
            STATE.clean_labels = labels
        STATE.log(f"Clean accuracy: {acc * 100:.2f}% (95% CI {lo * 100:.2f}-{hi * 100:.2f}%)")
        return STATE.clean_eval

    run_in_background(job, run)
    return {"job_id": job.id}


@app.post("/api/attack/run")
def attack_run(req: AttackReq):
    if STATE.model is None or STATE.test_loader is None:
        raise HTTPException(400, "Load a dataset and select a model first.")
    job = create_job(f"attack_{req.attack}")

    def run(job):
        job.message = f"Running {req.attack} attack..."
        acc, preds, labels, examples = eval_mod.adversarial_accuracy(
            STATE.model, STATE.test_loader, STATE.device, req.attack,
            req.epsilon, req.alpha, req.steps, STATE.num_classes,
            max_samples=req.n_samples, surrogate_model=STATE.surrogate_model)
        lo, hi = eval_mod.bootstrap_ci(preds, labels)

        example_imgs = [{
            "true": ex["true"], "pred": ex["pred"],
            "clean_b64": gradcam_mod.tensor_to_base64(ex["clean"]),
            "adv_b64": gradcam_mod.tensor_to_base64(ex["adv"]),
        } for ex in examples]

        result = {"accuracy": acc, "ci_low": lo, "ci_high": hi, "n_samples": int(len(labels)),
                   "example_images": example_imgs}
        with STATE.lock:
            STATE.attack_results[req.attack] = result
            STATE.attack_preds[req.attack] = (preds, labels)
            acc_map = {}
            if STATE.clean_eval:
                acc_map["clean"] = STATE.clean_eval["accuracy"]
            acc_map.update({k: v["accuracy"] for k, v in STATE.attack_results.items()})
            STATE.plots["adversarial_accuracy_bar"] = plot_mod.plot_accuracy_bar(acc_map)
        STATE.log(f"{req.attack} adversarial accuracy: {acc * 100:.2f}%")
        return result

    run_in_background(job, run)
    return {"job_id": job.id}


@app.post("/api/smoothing/run")
def smoothing_run(req: SmoothingReq):
    if STATE.model is None or STATE.test_loader is None:
        raise HTTPException(400, "Load a dataset and select a model first.")
    job = create_job("smoothing")

    def run(job):
        job.message = "Evaluating randomized smoothing..."
        result = robust_mod.randomized_smoothing_eval(
            STATE.model, STATE.test_loader, STATE.device,
            sigma=req.sigma, n_samples=req.n_noise_samples, max_batches=req.max_batches)
        with STATE.lock:
            STATE.smoothing_result = result
        STATE.log(f"Randomized smoothing: {result['smoothed_accuracy'] * 100:.2f}% "
                   f"(mean certified radius {result['mean_certified_radius']:.3f})")
        return result

    run_in_background(job, run)
    return {"job_id": job.id}


# ================================ GRAD-CAM =================================

@app.post("/api/gradcam/run")
def gradcam_run(req: GradCamReq):
    if STATE.model is None or STATE.test_loader is None:
        raise HTTPException(400, "Load a dataset and select a model first.")
    job = create_job("gradcam")

    def run(job):
        cam_engine = gradcam_mod.GradCAM(STATE.model)
        results = []
        collected = 0
        for x, y in STATE.test_loader:
            if collected >= req.n_samples:
                break
            x, y = x.to(STATE.device), y.to(STATE.device)
            x_in = x
            if req.mode == "adversarial":
                atk = build_attack(req.attack, STATE.model, eps=0.031,
                                    alpha=2 / 255, steps=10, n_classes=STATE.num_classes)
                x_in = atk(x, y)
            cams, out = cam_engine(x_in, target_class=y)
            preds = out.argmax(1)
            for i in range(x_in.size(0)):
                if collected >= req.n_samples:
                    break
                b64 = gradcam_mod.overlay_cam_on_image(x_in[i].detach().cpu(), cams[i])
                results.append({"image": b64, "true": int(y[i].item()), "pred": int(preds[i].item())})
                collected += 1
        with STATE.lock:
            STATE.gradcam_results = results
        STATE.log(f"Grad-CAM generated for {len(results)} samples ({req.mode}).")
        return {"count": len(results)}

    run_in_background(job, run)
    return {"job_id": job.id}


# ================================ METRICS ==================================

@app.post("/api/metrics/run")
def metrics_run():
    if STATE.model is None:
        raise HTTPException(400, "Select a model first.")
    job = create_job("metrics")

    def run(job):
        shape = (STATE.in_channels, STATE.img_size, STATE.img_size)
        fp = metrics_mod.flops_and_params(STATE.model, shape, STATE.device)
        lat = metrics_mod.inference_latency(STATE.model, shape, STATE.device)
        result = {**fp, "latency": lat}
        with STATE.lock:
            STATE.metrics = result
        STATE.log(f"Metrics: {fp['params_millions']}M params, {fp['flops_gflops']} GFLOPs")
        return result

    run_in_background(job, run)
    return {"job_id": job.id}


# =============================== STATISTICS =================================

@app.post("/api/stats/run")
def stats_run():
    if STATE.clean_eval is None:
        raise HTTPException(400, "Run clean evaluation first.")
    job = create_job("stats")

    def run(job):
        stats = {}
        clean_preds = STATE.clean_preds
        for name, (preds, labels) in STATE.attack_preds.items():
            n = min(len(clean_preds), len(preds))
            stats[f"clean_vs_{name}"] = eval_mod.mcnemar_test(clean_preds[:n], preds[:n], labels[:n])
        with STATE.lock:
            STATE.stats = stats
        STATE.log(f"Computed statistical comparisons for {len(stats)} attack(s).")
        return stats

    run_in_background(job, run)
    return {"job_id": job.id}


# ================================ REPORT ====================================

@app.get("/api/report/download")
def report_download():
    html = report_mod.generate_report_html(STATE)
    out_path = os.path.join(data_mod.DATA_ROOT, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return FileResponse(out_path, filename="adversarial_robustness_report.html", media_type="text/html")


# ================================ PIPELINE ===================================
# Lets the user tick several blocks and run them as one ordered sequence.
# Each block also still works independently — clicking several block buttons
# separately runs them as independent concurrent background jobs.

BLOCK_DISPATCH = {
    "train_clean": lambda p: train_clean(TrainCleanReq(**p)),
    "train_adversarial": lambda p: train_adversarial(TrainAdvReq(**p)),
    "eval_clean": lambda p: eval_clean(),
    "attack": lambda p: attack_run(AttackReq(**p)),
    "smoothing": lambda p: smoothing_run(SmoothingReq(**p)),
    "gradcam": lambda p: gradcam_run(GradCamReq(**p)),
    "metrics": lambda p: metrics_run(),
    "stats": lambda p: stats_run(),
}


@app.post("/api/pipeline/run")
def pipeline_run(req: PipelineReq):
    job = create_job("pipeline")

    def run(job):
        results = []
        for i, step in enumerate(req.steps):
            job.message = f"Running block {i + 1}/{len(req.steps)}: {step.type}"
            job.progress = i / max(len(req.steps), 1)
            fn = BLOCK_DISPATCH.get(step.type)
            if fn is None:
                results.append({"type": step.type, "error": "unknown block type"})
                continue
            sub = fn(step.params)
            sub_job_id = sub["job_id"]
            while True:
                sub_job = get_job(sub_job_id)
                if sub_job.status in ("done", "error"):
                    break
                time.sleep(0.5)
            results.append({"type": step.type, "status": sub_job.status,
                             "result": sub_job.result, "error": sub_job.error})
            job.log_lines.append(f"{step.type}: {sub_job.status}")
        return {"steps": results}

    run_in_background(job, run)
    return {"job_id": job.id}
