// ===================== helpers =====================

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`${res.status}: ${txt}`);
  }
  return res.json();
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function pct(x) {
  return x === null || x === undefined ? "—" : (x * 100).toFixed(2) + "%";
}

function setProgress(elId, frac) {
  const wrap = document.getElementById(elId);
  if (!wrap) return;
  wrap.classList.add("active");
  wrap.querySelector(".progress-bar").style.width = `${Math.round(frac * 100)}%`;
}

function clearProgress(elId) {
  const wrap = document.getElementById(elId);
  if (!wrap) return;
  wrap.classList.remove("active");
  wrap.querySelector(".progress-bar").style.width = "0%";
}

function statusPill(status) {
  return `<span class="status-pill ${status}">${status}</span>`;
}

async function pollJob(jobId, { onTick, progressElId } = {}) {
  while (true) {
    const job = await getJSON(`/api/jobs/${jobId}`);
    if (onTick) onTick(job);
    if (progressElId) setProgress(progressElId, job.progress || 0);
    if (job.status === "done" || job.status === "error") {
      if (progressElId) clearProgress(progressElId);
      return job;
    }
    await new Promise((r) => setTimeout(r, 900));
  }
}

function renderError(elId, err) {
  document.getElementById(elId).innerHTML =
    `<div class="status-pill error">error</div><pre style="white-space:pre-wrap;color:#c43d3d;font-size:0.75rem;">${err.message || err}</pre>`;
}

// ===================== tabs =====================

document.querySelectorAll("#dataset-tabs .tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#dataset-tabs .tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.querySelector(`.tab-panel[data-panel="${btn.dataset.tab}"]`).classList.add("active");
  });
});

// ===================== 1. dataset =====================

async function runBuiltinDataset() {
  const box = "result-dataset";
  document.getElementById(box).innerHTML = statusPill("running");
  try {
    const { job_id } = await postJSON("/api/dataset/builtin", {
      name: document.getElementById("builtin-name").value,
      batch_size: Number(document.getElementById("builtin-batch").value),
    });
    const job = await pollJob(job_id);
    renderDatasetResult(box, job);
  } catch (e) { renderError(box, e); }
}

async function runKaggleDataset() {
  const box = "result-dataset";
  document.getElementById(box).innerHTML = statusPill("running") + " downloading from Kaggle — this can take a while for large datasets…";
  try {
    const { job_id } = await postJSON("/api/dataset/kaggle", {
      slug: document.getElementById("kaggle-slug").value,
      username: document.getElementById("kaggle-username").value,
      key: document.getElementById("kaggle-key").value,
      img_size: Number(document.getElementById("kaggle-imgsize").value),
      batch_size: Number(document.getElementById("kaggle-batch").value),
      val_split: Number(document.getElementById("kaggle-valsplit").value),
    });
    const job = await pollJob(job_id);
    renderDatasetResult(box, job);
  } catch (e) { renderError(box, e); }
}

async function runUrlDataset() {
  const box = "result-dataset";
  const lines = document.getElementById("urls-textarea").value.split("\n").map((l) => l.trim()).filter(Boolean);
  const items = lines.map((l) => {
    const [label, ...rest] = l.split(",");
    return { label: label.trim(), url: rest.join(",").trim() };
  });
  document.getElementById(box).innerHTML = statusPill("running") + ` downloading ${items.length} images…`;
  try {
    const { job_id } = await postJSON("/api/dataset/urls", {
      items,
      img_size: Number(document.getElementById("urls-imgsize").value),
      batch_size: Number(document.getElementById("urls-batch").value),
    });
    const job = await pollJob(job_id);
    renderDatasetResult(box, job);
  } catch (e) { renderError(box, e); }
}

async function runUploadDataset() {
  const box = "result-dataset";
  const file = document.getElementById("upload-file").files[0];
  if (!file) { alert("Choose a .zip file first."); return; }
  const fd = new FormData();
  fd.append("file", file);
  const imgSize = document.getElementById("upload-imgsize").value;
  const batch = document.getElementById("upload-batch").value;
  document.getElementById(box).innerHTML = statusPill("running") + " uploading & extracting…";
  try {
    const res = await fetch(`/api/dataset/upload?img_size=${imgSize}&batch_size=${batch}`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    const { job_id } = await res.json();
    const job = await pollJob(job_id);
    renderDatasetResult(box, job);
  } catch (e) { renderError(box, e); }
}

function renderDatasetResult(box, job) {
  if (job.status === "error") { renderError(box, { message: job.error }); return; }
  const r = job.result;
  document.getElementById(box).innerHTML =
    `${statusPill("done")} <b>${r.dataset}</b> — ${r.num_classes} classes, ${r.img_size}×${r.img_size}×${r.channels}`;
  refreshState();
}

// ===================== 2. model =====================

async function runModelSelect() {
  const box = "result-model";
  document.getElementById(box).innerHTML = statusPill("running");
  try {
    const r = await postJSON("/api/model/select", {
      name: document.getElementById("model-name").value,
      pretrained: document.getElementById("model-pretrained").checked,
    });
    document.getElementById(box).innerHTML =
      `${statusPill("done")} <b>${r.backbone}</b> — ${(r.params / 1e6).toFixed(2)}M parameters`;
    refreshState();
  } catch (e) { renderError(box, e); }
}

// ===================== 3. enhancement =====================

async function runEnhance() {
  const box = "result-enhance";
  document.getElementById(box).innerHTML = statusPill("running");
  try {
    const r = await postJSON("/api/model/enhance", { kind: document.getElementById("enhance-kind").value });
    document.getElementById(box).innerHTML =
      `${statusPill("done")} enhancement=<b>${r.enhancement}</b> — ${(r.params / 1e6).toFixed(2)}M parameters`;
    refreshState();
  } catch (e) { renderError(box, e); }
}

// ===================== 4. clean training =====================

async function runTrainClean() {
  const box = "result-train-clean";
  document.getElementById(box).innerHTML = statusPill("running");
  try {
    const { job_id } = await postJSON("/api/train/clean", {
      epochs: Number(document.getElementById("clean-epochs").value),
      lr: Number(document.getElementById("clean-lr").value),
    });
    const job = await pollJob(job_id, {
      progressElId: "progress-train-clean",
      onTick: (j) => { document.getElementById(box).innerHTML = statusPill(j.status) + " " + (j.message || ""); },
    });
    if (job.status === "error") { renderError(box, { message: job.error }); return; }
    const h = job.result.history;
    const last = h.val_acc.length - 1;
    document.getElementById(box).innerHTML =
      `${statusPill("done")} final val accuracy: <b>${pct(h.val_acc[last])}</b>`;
    refreshState();
  } catch (e) { renderError(box, e); }
}

// ===================== 5. adversarial/robust training =====================

async function runTrainAdv() {
  const box = "result-train-adv";
  document.getElementById(box).innerHTML = statusPill("running");
  try {
    const { job_id } = await postJSON("/api/train/adversarial", {
      method: document.getElementById("adv-method").value,
      epsilon: Number(document.getElementById("adv-eps").value),
      alpha: Number(document.getElementById("adv-alpha").value),
      steps: Number(document.getElementById("adv-steps").value),
      epochs: Number(document.getElementById("adv-epochs").value),
      lr: Number(document.getElementById("adv-lr").value),
      beta: Number(document.getElementById("adv-beta").value),
      awp_gamma: Number(document.getElementById("adv-awpgamma").value),
    });
    const job = await pollJob(job_id, {
      progressElId: "progress-train-adv",
      onTick: (j) => { document.getElementById(box).innerHTML = statusPill(j.status) + " " + (j.message || ""); },
    });
    if (job.status === "error") { renderError(box, { message: job.error }); return; }
    const h = job.result.history;
    const last = h.val_acc.length - 1;
    document.getElementById(box).innerHTML =
      `${statusPill("done")} final val accuracy: <b>${pct(h.val_acc[last])}</b>`;
    refreshState();
  } catch (e) { renderError(box, e); }
}

// ===================== 6. clean eval =====================

async function runEvalClean() {
  const box = "result-eval-clean";
  document.getElementById(box).innerHTML = statusPill("running");
  try {
    const { job_id } = await postJSON("/api/eval/clean", {});
    const job = await pollJob(job_id);
    if (job.status === "error") { renderError(box, { message: job.error }); return; }
    const r = job.result;
    document.getElementById(box).innerHTML =
      `${statusPill("done")} clean accuracy: <b>${pct(r.accuracy)}</b> (95% CI ${pct(r.ci_low)} – ${pct(r.ci_high)})`;
    refreshState();
  } catch (e) { renderError(box, e); }
}

// ===================== 7. attacks (dynamic grid) =====================

const ATTACK_DEFAULTS = {
  fgsm: { eps: 0.031, alpha: 0.0078, steps: 1 },
  ifgsm: { eps: 0.031, alpha: 0.0078, steps: 10 },
  pgd: { eps: 0.031, alpha: 0.0078, steps: 10 },
  autoattack: { eps: 0.031, alpha: 0.0078, steps: 10 },
  square: { eps: 0.031, alpha: 0.0078, steps: 10 },
  transfer: { eps: 0.031, alpha: 0.0078, steps: 10 },
};

async function buildAttackGrid() {
  const reg = await getJSON("/api/registry");
  const grid = document.getElementById("attack-grid");
  grid.innerHTML = "";
  Object.entries(reg.attacks).forEach(([key, info]) => {
    const d = ATTACK_DEFAULTS[key] || { eps: 0.031, alpha: 0.0078, steps: 10 };
    const card = document.createElement("div");
    card.className = "attack-card";
    card.innerHTML = `
      <span class="tag">${info.type}</span>
      <h3>${info.label}</h3>
      <p>${info.desc}</p>
      <div class="field-row">
        <label>Epsilon <input type="number" step="0.001" id="atk-${key}-eps" value="${d.eps}"></label>
        <label>Alpha <input type="number" step="0.001" id="atk-${key}-alpha" value="${d.alpha}"></label>
        <label>Steps <input type="number" id="atk-${key}-steps" value="${d.steps}"></label>
        <label>N samples <input type="number" id="atk-${key}-n" value="300"></label>
      </div>
      <button class="btn-run" onclick="runAttack('${key}')">Run ${info.label}</button>
      <div class="result-box" id="result-atk-${key}"></div>
    `;
    grid.appendChild(card);
  });
}

async function runAttack(key) {
  const box = `result-atk-${key}`;
  document.getElementById(box).innerHTML = statusPill("running");
  try {
    const { job_id } = await postJSON("/api/attack/run", {
      attack: key,
      epsilon: Number(document.getElementById(`atk-${key}-eps`).value),
      alpha: Number(document.getElementById(`atk-${key}-alpha`).value),
      steps: Number(document.getElementById(`atk-${key}-steps`).value),
      n_samples: Number(document.getElementById(`atk-${key}-n`).value),
    });
    const job = await pollJob(job_id);
    if (job.status === "error") { renderError(box, { message: job.error }); return; }
    const r = job.result;
    let examplesHtml = "";
    (r.example_images || []).slice(0, 4).forEach((ex) => {
      examplesHtml += `
        <div class="example-pair">
          <figure><img src="data:image/png;base64,${ex.clean_b64}"><figcaption>clean (${ex.true})</figcaption></figure>
          <figure><img src="data:image/png;base64,${ex.adv_b64}"><figcaption>adv → ${ex.pred}</figcaption></figure>
        </div>`;
    });
    document.getElementById(box).innerHTML =
      `${statusPill("done")} adversarial accuracy: <b>${pct(r.accuracy)}</b> (95% CI ${pct(r.ci_low)} – ${pct(r.ci_high)}, n=${r.n_samples})
       <div class="gallery">${examplesHtml}</div>`;
    refreshState();
  } catch (e) { renderError(box, e); }
}

// ===================== 8. certified robustness =====================

async function runSmoothing() {
  const box = "result-smoothing";
  document.getElementById(box).innerHTML = statusPill("running");
  try {
    const { job_id } = await postJSON("/api/smoothing/run", {
      sigma: Number(document.getElementById("smooth-sigma").value),
      n_noise_samples: Number(document.getElementById("smooth-n").value),
      max_batches: Number(document.getElementById("smooth-maxbatch").value),
    });
    const job = await pollJob(job_id);
    if (job.status === "error") { renderError(box, { message: job.error }); return; }
    const r = job.result;
    document.getElementById(box).innerHTML =
      `${statusPill("done")} smoothed accuracy: <b>${pct(r.smoothed_accuracy)}</b>, mean certified L2 radius: <b>${r.mean_certified_radius.toFixed(4)}</b> (σ=${r.sigma}, n=${r.n_evaluated})`;
    refreshState();
  } catch (e) { renderError(box, e); }
}

// ===================== 9. gradcam =====================

async function runGradcam() {
  const box = "result-gradcam";
  document.getElementById(box).innerHTML = statusPill("running");
  try {
    const { job_id } = await postJSON("/api/gradcam/run", {
      mode: document.getElementById("gradcam-mode").value,
      attack: document.getElementById("gradcam-attack").value,
      n_samples: Number(document.getElementById("gradcam-n").value),
    });
    const job = await pollJob(job_id);
    if (job.status === "error") { renderError(box, { message: job.error }); return; }
    const gc = await getJSON("/api/gradcam/results");
    const imgs = gc.examples.map((ex) =>
      `<figure><img src="data:image/png;base64,${ex.image}"><figcaption>true=${ex.true} pred=${ex.pred}</figcaption></figure>`
    ).join("");
    document.getElementById(box).innerHTML =
      `${statusPill("done")} ${job.result.count} Grad-CAM samples generated<div class="gallery">${imgs}</div>`;
    refreshState();
  } catch (e) { renderError(box, e); }
}

// ===================== 10. metrics =====================

async function runMetrics() {
  const box = "result-metrics";
  document.getElementById(box).innerHTML = statusPill("running");
  try {
    const { job_id } = await postJSON("/api/metrics/run", {});
    const job = await pollJob(job_id);
    if (job.status === "error") { renderError(box, { message: job.error }); return; }
    const r = job.result;
    document.getElementById(box).innerHTML = `${statusPill("done")}
      <table>
        <tr><th>Params</th><td>${r.params_millions}M</td></tr>
        <tr><th>FLOPs</th><td>${r.flops_gflops} GFLOPs</td></tr>
        <tr><th>Latency (mean)</th><td>${r.latency.mean_ms.toFixed(2)} ms</td></tr>
        <tr><th>Latency (p95)</th><td>${r.latency.p95_ms.toFixed(2)} ms</td></tr>
      </table>`;
    refreshState();
  } catch (e) { renderError(box, e); }
}

// ===================== 11. statistics =====================

async function runStats() {
  const box = "result-stats";
  document.getElementById(box).innerHTML = statusPill("running");
  try {
    const { job_id } = await postJSON("/api/stats/run", {});
    const job = await pollJob(job_id);
    if (job.status === "error") { renderError(box, { message: job.error }); return; }
    const rows = Object.entries(job.result).map(([k, v]) =>
      `<tr><td>${k}</td><td>${v.statistic.toFixed(3)}</td><td>${v.p_value.toExponential(3)}</td></tr>`).join("");
    document.getElementById(box).innerHTML = `${statusPill("done")}
      <table><tr><th>Comparison</th><th>Statistic</th><th>p-value</th></tr>${rows}</table>`;
    refreshState();
  } catch (e) { renderError(box, e); }
}

// ===================== 12. pipeline =====================

async function runPipeline() {
  const box = "result-pipeline";
  const checks = Array.from(document.querySelectorAll(".pipeline-step:checked"));
  if (checks.length === 0) { alert("Tick at least one block to run."); return; }

  const steps = checks.map((c) => {
    if (c.value === "train_clean") {
      return { type: "train_clean", params: {
        epochs: Number(document.getElementById("clean-epochs").value),
        lr: Number(document.getElementById("clean-lr").value) } };
    }
    if (c.value === "train_adversarial") {
      return { type: "train_adversarial", params: {
        method: document.getElementById("adv-method").value,
        epsilon: Number(document.getElementById("adv-eps").value),
        alpha: Number(document.getElementById("adv-alpha").value),
        steps: Number(document.getElementById("adv-steps").value),
        epochs: Number(document.getElementById("adv-epochs").value),
        lr: Number(document.getElementById("adv-lr").value),
        beta: Number(document.getElementById("adv-beta").value),
        awp_gamma: Number(document.getElementById("adv-awpgamma").value) } };
    }
    if (c.value === "eval_clean") return { type: "eval_clean", params: {} };
    if (c.value === "attack") {
      const key = c.dataset.attack;
      const d = ATTACK_DEFAULTS[key] || { eps: 0.031, alpha: 0.0078, steps: 10 };
      return { type: "attack", params: { attack: key, epsilon: d.eps, alpha: d.alpha, steps: d.steps, n_samples: 300 } };
    }
    if (c.value === "smoothing") {
      return { type: "smoothing", params: {
        sigma: Number(document.getElementById("smooth-sigma").value),
        n_noise_samples: Number(document.getElementById("smooth-n").value),
        max_batches: Number(document.getElementById("smooth-maxbatch").value) } };
    }
    if (c.value === "gradcam") {
      return { type: "gradcam", params: {
        mode: document.getElementById("gradcam-mode").value,
        attack: document.getElementById("gradcam-attack").value,
        n_samples: Number(document.getElementById("gradcam-n").value) } };
    }
    if (c.value === "metrics") return { type: "metrics", params: {} };
    if (c.value === "stats") return { type: "stats", params: {} };
    return null;
  }).filter(Boolean);

  document.getElementById(box).innerHTML = statusPill("running") + ` running ${steps.length} block(s)…`;
  try {
    const { job_id } = await postJSON("/api/pipeline/run", { steps });
    const job = await pollJob(job_id, {
      progressElId: "progress-pipeline",
      onTick: (j) => { document.getElementById(box).innerHTML = statusPill(j.status) + " " + (j.message || ""); },
    });
    if (job.status === "error") { renderError(box, { message: job.error }); return; }
    const rows = job.result.steps.map((s) =>
      `<tr><td>${s.type}</td><td>${statusPill(s.status)}</td></tr>`).join("");
    document.getElementById(box).innerHTML = `${statusPill("done")} pipeline complete
      <table><tr><th>Block</th><th>Status</th></tr>${rows}</table>`;
    refreshState();
  } catch (e) { renderError(box, e); }
}

// ===================== 13. results dashboard =====================

async function refreshState() {
  try {
    const s = await getJSON("/api/state");

    document.getElementById("device-label").textContent =
      `${s.device} · ${s.dataset || "no dataset"} · ${s.backbone || "no model"}${s.enhancement && s.enhancement !== "none" ? " +" + s.enhancement : ""}`;
    const dot = document.getElementById("device-dot");
    dot.className = "dot " + (s.device.includes("cuda") ? "gpu" : "ok");

    let html = "<table><tr><th>Metric</th><th>Value</th></tr>";
    if (s.clean_eval) html += `<tr><td>Clean accuracy</td><td>${pct(s.clean_eval.accuracy)} (CI ${pct(s.clean_eval.ci_low)}–${pct(s.clean_eval.ci_high)})</td></tr>`;
    Object.entries(s.attack_results || {}).forEach(([k, v]) => {
      html += `<tr><td>${k} adversarial accuracy</td><td>${pct(v.accuracy)} (n=${v.n_samples})</td></tr>`;
    });
    if (s.smoothing_result) html += `<tr><td>Smoothed accuracy</td><td>${pct(s.smoothing_result.smoothed_accuracy)}, radius ${s.smoothing_result.mean_certified_radius.toFixed(3)}</td></tr>`;
    if (s.metrics) html += `<tr><td>Params / FLOPs</td><td>${s.metrics.params_millions}M / ${s.metrics.flops_gflops} GFLOPs</td></tr>`;
    if (s.metrics) html += `<tr><td>Inference latency</td><td>${s.metrics.latency.mean_ms.toFixed(2)} ms (mean)</td></tr>`;
    html += "</table>";
    document.getElementById("results-summary").innerHTML = html;

    const plotsDiv = document.getElementById("results-plots");
    const currentPlotNames = new Set(s.plots || []);
    if (!window.__renderedPlots) window.__renderedPlots = new Set();
    for (const name of window.__renderedPlots) {
      if (!currentPlotNames.has(name)) window.__renderedPlots.delete(name);
    }
    for (const name of currentPlotNames) {
      if (window.__renderedPlots.has(name)) continue;
      try {
        const p = await getJSON(`/api/plots/${name}`);
        const fig = document.createElement("figure");
        fig.id = `plot-${name}`;
        fig.innerHTML = `<img src="data:image/png;base64,${p.image_b64}" style="width:320px;height:auto;">
                          <figcaption>${name}</figcaption>`;
        plotsDiv.appendChild(fig);
        window.__renderedPlots.add(name);
      } catch (e) { /* ignore */ }
    }

    const logDiv = document.getElementById("log-lines");
    logDiv.innerHTML = (s.recent_logs || []).map((l) => `<div>${l}</div>`).join("");
    logDiv.scrollTop = logDiv.scrollHeight;
  } catch (e) {
    document.getElementById("device-label").textContent = "disconnected";
  }
}

// ===================== init =====================

buildAttackGrid();
refreshState();
setInterval(refreshState, 4000);
