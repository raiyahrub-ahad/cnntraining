// ===================== WORKFLOW VISUALIZATION =====================

class WorkflowCanvas {
  constructor() {
    this.canvas = document.getElementById('workflow-canvas');
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.blocks = [];
    this.selectedBlock = null;
    this.isDragging = false;
    this.dragOffset = { x: 0, y: 0 };
    
    this.colors = {
      dataset: '#FF6B6B',
      model: '#4ECDC4',
      train: '#45B7D1',
      evaluate: '#FFA07A',
      attack: '#FF6B9D',
      robustness: '#C44569',
      analysis: '#95E1D3',
      export: '#F38181'
    };

    this.setupCanvas();
    this.setupEventListeners();
  }

  setupCanvas() {
    if (!this.canvas) return;
    this.canvas.width = this.canvas.offsetWidth;
    this.canvas.height = 500;
    window.addEventListener('resize', () => this.resizeCanvas());
  }

  resizeCanvas() {
    if (!this.canvas) return;
    this.canvas.width = this.canvas.offsetWidth;
    this.redraw();
  }

  setupEventListeners() {
    if (!this.canvas) return;
    this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
    this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
    this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
  }

  addBlock(type, label, x = null, y = null) {
    if (x === null) x = this.blocks.length * 160 + 30;
    if (y === null) y = 200;

    const block = {
      id: Date.now(),
      type: type,
      label: label,
      x: x,
      y: y,
      width: 130,
      height: 70,
      color: this.colors[type] || '#999',
      status: 'pending',
      progress: 0
    };

    this.blocks.push(block);
    this.redraw();
    return block;
  }

  updateBlockStatus(blockId, status, progress = 0) {
    const block = this.blocks.find(b => b.id === blockId);
    if (block) {
      block.status = status;
      block.progress = progress;
      this.redraw();
    }
  }

  handleMouseDown(e) {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    this.selectedBlock = this.blocks.find(b =>
      x >= b.x && x <= b.x + b.width && y >= b.y && y <= b.y + b.height
    );

    if (this.selectedBlock) {
      this.isDragging = true;
      this.dragOffset = { x: x - this.selectedBlock.x, y: y - this.selectedBlock.y };
    }
  }

  handleMouseMove(e) {
    if (!this.isDragging || !this.selectedBlock || !this.canvas) return;

    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    this.selectedBlock.x = Math.max(0, Math.min(x - this.dragOffset.x, this.canvas.width - this.selectedBlock.width));
    this.selectedBlock.y = Math.max(0, Math.min(y - this.dragOffset.y, this.canvas.height - this.selectedBlock.height));

    this.redraw();
  }

  handleMouseUp() {
    this.isDragging = false;
  }

  redraw() {
    if (!this.canvas) return;
    this.ctx.fillStyle = '#0f0f1e';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.drawBlocks();
  }

  drawBlocks() {
    this.blocks.forEach(block => {
      // Background
      this.ctx.fillStyle = block.color;
      this.ctx.globalAlpha = block.status === 'done' ? 1 : (block.status === 'running' ? 0.85 : 0.65);
      this.ctx.fillRect(block.x, block.y, block.width, block.height);

      // Border
      this.ctx.strokeStyle = block === this.selectedBlock ? '#fff' : '#333';
      this.ctx.lineWidth = block === this.selectedBlock ? 2 : 1;
      this.ctx.strokeRect(block.x, block.y, block.width, block.height);

      this.ctx.globalAlpha = 1;

      // Progress bar
      if (block.status === 'running' && block.progress > 0) {
        this.ctx.fillStyle = 'rgba(76, 175, 80, 0.5)';
        this.ctx.fillRect(block.x, block.y + block.height - 5, block.width * block.progress, 5);
      }

      // Text
      this.ctx.fillStyle = '#fff';
      this.ctx.font = 'bold 11px Arial';
      this.ctx.textAlign = 'center';
      this.ctx.fillText(block.label, block.x + block.width / 2, block.y + 20);

      // Status
      this.ctx.font = '9px Arial';
      this.ctx.fillStyle = '#ddd';
      this.ctx.fillText(block.status, block.x + block.width / 2, block.y + 40);
      if (block.status === 'running') {
        this.ctx.fillText(`${Math.round(block.progress * 100)}%`, block.x + block.width / 2, block.y + 55);
      }
    });
  }

  clearAll() {
    this.blocks = [];
    this.redraw();
  }
}

let workflow = new WorkflowCanvas();

// ===================== HELPERS =====================

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

function addLog(message, type = 'info') {
  const logDiv = document.getElementById("log-lines");
  if (!logDiv) return;
  const entry = document.createElement("div");
  entry.className = `log-entry ${type}`;
  const time = new Date().toLocaleTimeString();
  entry.textContent = `[${time}] ${message}`;
  logDiv.appendChild(entry);
  logDiv.scrollTop = logDiv.scrollHeight;
}

function statusPill(status) {
  const colors = {
    running: '#45B7D1',
    done: '#95E1D3',
    error: '#FF6B6B',
    pending: '#999'
  };
  return `<span style="display:inline-block;padding:4px 8px;background:${colors[status] || '#999'};color:white;border-radius:3px;font-size:11px;font-weight:bold;">${status}</span>`;
}

async function pollJob(jobId, { onTick, blockId } = {}) {
  while (true) {
    const job = await getJSON(`/api/jobs/${jobId}`);
    if (onTick) onTick(job);
    if (blockId && workflow) {
      workflow.updateBlockStatus(blockId, job.status, job.progress || 0);
    }
    if (job.status === "done" || job.status === "error") {
      return job;
    }
    await new Promise((r) => setTimeout(r, 900));
  }
}

function renderError(elId, err) {
  const el = document.getElementById(elId);
  if (!el) return;
  el.innerHTML = `${statusPill("error")} <span style="color:#ff6b6b;">${err.message || err}</span>`;
  addLog(`Error: ${err.message}`, 'error');
}

// ===================== TAB SWITCHING =====================

document.querySelectorAll("#dataset-tabs .tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#dataset-tabs .tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.querySelector(`.tab-panel[data-panel="${btn.dataset.tab}"]`).classList.add("active");
  });
});

// ===================== 1. DATASET =====================

async function runBuiltinDataset() {
  const box = "result-dataset";
  const name = document.getElementById("builtin-name").value;
  const batch_size = Number(document.getElementById("builtin-batch").value);
  
  const blockId = workflow.addBlock("dataset", `Dataset: ${name}`).id;
  document.getElementById(box).innerHTML = statusPill("running");
  addLog(`Loading built-in dataset: ${name}...`, 'info');
  
  try {
    const { job_id } = await postJSON("/api/dataset/builtin", {
      name: name,
      batch_size: batch_size,
    });
    const job = await pollJob(job_id, { blockId });
    if (job.status === "done") {
      workflow.updateBlockStatus(blockId, "done");
      document.getElementById(box).innerHTML = `${statusPill("done")} <b>${job.result.dataset}</b> — ${job.result.num_classes} classes, ${job.result.img_size}×${job.result.img_size}`;
      addLog(`✓ Dataset loaded: ${job.result.num_classes} classes, ${job.result.img_size}×${job.result.img_size}`, 'success');
    } else {
      workflow.updateBlockStatus(blockId, "error");
      renderError(box, { message: job.error });
    }
    refreshState();
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error");
    renderError(box, e);
  }
}

// ===================== 2. MODEL =====================

async function runModelSelect() {
  const box = "result-model";
  const name = document.getElementById("model-name").value;
  const pretrained = document.getElementById("model-pretrained").checked;
  
  const blockId = workflow.addBlock("model", `Model: ${name}`).id;
  document.getElementById(box).innerHTML = statusPill("running");
  addLog(`Selecting model: ${name}${pretrained ? ' (pretrained)' : ''}...`, 'info');
  
  try {
    const r = await postJSON("/api/model/select", {
      name: name,
      pretrained: pretrained,
    });
    workflow.updateBlockStatus(blockId, "done");
    document.getElementById(box).innerHTML = `${statusPill("done")} <b>${r.backbone}</b> — ${(r.params / 1e6).toFixed(2)}M params`;
    addLog(`✓ Model selected: ${(r.params / 1e6).toFixed(2)}M parameters`, 'success');
    refreshState();
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error");
    renderError(box, e);
  }
}

// ===================== 3. ENHANCEMENT =====================

async function runEnhance() {
  const box = "result-enhance";
  const kind = document.getElementById("enhance-kind").value;
  
  const blockId = workflow.addBlock("model", `Enhancement: ${kind}`).id;
  document.getElementById(box).innerHTML = statusPill("running");
  addLog(`Applying enhancement: ${kind}...`, 'info');
  
  try {
    const r = await postJSON("/api/model/enhance", { kind: kind });
    workflow.updateBlockStatus(blockId, "done");
    document.getElementById(box).innerHTML = `${statusPill("done")} enhancement=<b>${r.enhancement}</b> — ${(r.params / 1e6).toFixed(2)}M params`;
    addLog(`✓ Enhancement applied: ${(r.params / 1e6).toFixed(2)}M parameters`, 'success');
    refreshState();
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error");
    renderError(box, e);
  }
}

// ===================== 4. CLEAN TRAINING =====================

async function runTrainClean() {
  const box = "result-train-clean";
  const epochs = Number(document.getElementById("clean-epochs").value);
  const lr = Number(document.getElementById("clean-lr").value);
  
  const blockId = workflow.addBlock("train", `Clean Train (${epochs}e)`).id;
  document.getElementById(box).innerHTML = statusPill("running");
  addLog(`Starting clean training: ${epochs} epochs, lr=${lr}...`, 'info');
  
  try {
    const { job_id } = await postJSON("/api/train/clean", {
      epochs: epochs,
      lr: lr,
    });
    const job = await pollJob(job_id, {
      blockId,
      onTick: (j) => {
        if (j.message) addLog(j.message, 'info');
        document.getElementById(box).innerHTML = statusPill(j.status) + " " + (j.message || "");
      }
    });
    if (job.status === "done") {
      workflow.updateBlockStatus(blockId, "done");
      const h = job.result.history;
      const last = h.val_acc[h.val_acc.length - 1];
      document.getElementById(box).innerHTML = `${statusPill("done")} final accuracy: <b>${pct(last)}</b>`;
      addLog(`✓ Training complete! Final accuracy: ${pct(last)}`, 'success');
    } else {
      workflow.updateBlockStatus(blockId, "error");
      renderError(box, { message: job.error });
    }
    refreshState();
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error");
    renderError(box, e);
  }
}

// ===================== 5. ADVERSARIAL TRAINING =====================

async function runTrainAdv() {
  const box = "result-train-adv";
  const method = document.getElementById("adv-method").value;
  const epochs = Number(document.getElementById("adv-epochs").value);
  
  const blockId = workflow.addBlock("train", `Adv Train (${method})`).id;
  document.getElementById(box).innerHTML = statusPill("running");
  addLog(`Starting adversarial training: ${method}, ${epochs} epochs...`, 'info');
  
  try {
    const { job_id } = await postJSON("/api/train/adversarial", {
      method: method,
      epsilon: Number(document.getElementById("adv-eps").value),
      alpha: Number(document.getElementById("adv-alpha").value),
      steps: Number(document.getElementById("adv-steps").value),
      epochs: epochs,
      lr: Number(document.getElementById("adv-lr").value),
      beta: Number(document.getElementById("adv-beta").value),
      awp_gamma: Number(document.getElementById("adv-awpgamma").value),
    });
    const job = await pollJob(job_id, {
      blockId,
      onTick: (j) => {
        if (j.message) addLog(j.message, 'info');
        document.getElementById(box).innerHTML = statusPill(j.status) + " " + (j.message || "");
      }
    });
    if (job.status === "done") {
      workflow.updateBlockStatus(blockId, "done");
      const h = job.result.history;
      const last = h.val_acc[h.val_acc.length - 1];
      document.getElementById(box).innerHTML = `${statusPill("done")} final accuracy: <b>${pct(last)}</b>`;
      addLog(`✓ Training complete! Final accuracy: ${pct(last)}`, 'success');
    } else {
      workflow.updateBlockStatus(blockId, "error");
      renderError(box, { message: job.error });
    }
    refreshState();
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error");
    renderError(box, e);
  }
}

// ===================== 6. CLEAN EVALUATION =====================

async function runEvalClean() {
  const box = "result-eval-clean";
  const blockId = workflow.addBlock("evaluate", "Eval: Clean").id;
  document.getElementById(box).innerHTML = statusPill("running");
  addLog("Evaluating clean accuracy...", 'info');
  
  try {
    const { job_id } = await postJSON("/api/eval/clean", {});
    const job = await pollJob(job_id, { blockId });
    if (job.status === "done") {
      workflow.updateBlockStatus(blockId, "done");
      const r = job.result;
      document.getElementById(box).innerHTML = `${statusPill("done")} clean accuracy: <b>${pct(r.accuracy)}</b> (95% CI ${pct(r.ci_low)}–${pct(r.ci_high)})`;
      addLog(`✓ Clean accuracy: ${pct(r.accuracy)} (CI ${pct(r.ci_low)}–${pct(r.ci_high)})`, 'success');
    } else {
      workflow.updateBlockStatus(blockId, "error");
      renderError(box, { message: job.error });
    }
    refreshState();
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error");
    renderError(box, e);
  }
}

// ===================== 7. ATTACKS =====================

const ATTACK_DEFAULTS = {
  fgsm: { eps: 0.031, alpha: 0.0078, steps: 1 },
  ifgsm: { eps: 0.031, alpha: 0.0078, steps: 10 },
  pgd: { eps: 0.031, alpha: 0.0078, steps: 10 },
};

async function buildAttackGrid() {
  try {
    const reg = await getJSON("/api/registry");
    const grid = document.getElementById("attack-grid");
    grid.innerHTML = "";
    Object.entries(reg.attacks || {}).forEach(([key, info]) => {
      const d = ATTACK_DEFAULTS[key] || { eps: 0.031, alpha: 0.0078, steps: 10 };
      const card = document.createElement("div");
      card.className = "attack-card";
      card.innerHTML = `
        <span class="tag">${info.type}</span>
        <h3>${info.label}</h3>
        <p>${info.desc}</p>
        <div class="field-row">
          <label>Epsilon <input type="number" step="0.001" id="atk-${key}-eps" value="${d.eps}"></label>
          <label>Steps <input type="number" id="atk-${key}-steps" value="${d.steps}"></label>
          <label>N <input type="number" id="atk-${key}-n" value="300"></label>
        </div>
        <button class="btn-run" onclick="runAttack('${key}')">Run ${info.label}</button>
        <div class="result-box" id="result-atk-${key}"></div>
      `;
      grid.appendChild(card);
    });
  } catch (e) {
    console.error("Error building attack grid:", e);
  }
}

async function runAttack(key) {
  const box = `result-atk-${key}`;
  const blockId = workflow.addBlock("attack", `Attack: ${key.toUpperCase()}`).id;
  document.getElementById(box).innerHTML = statusPill("running");
  addLog(`Running ${key} attack...`, 'info');
  
  try {
    const { job_id } = await postJSON("/api/attack/run", {
      attack: key,
      epsilon: Number(document.getElementById(`atk-${key}-eps`).value),
      alpha: 0.0078,
      steps: Number(document.getElementById(`atk-${key}-steps`).value),
      n_samples: Number(document.getElementById(`atk-${key}-n`).value),
    });
    const job = await pollJob(job_id, { blockId });
    if (job.status === "done") {
      workflow.updateBlockStatus(blockId, "done");
      const r = job.result;
      document.getElementById(box).innerHTML = `${statusPill("done")} adversarial accuracy: <b>${pct(r.accuracy)}</b>`;
      addLog(`✓ ${key} attack: ${pct(r.accuracy)}`, 'success');
    } else {
      workflow.updateBlockStatus(blockId, "error");
      renderError(box, { message: job.error });
    }
    refreshState();
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error");
    renderError(box, e);
  }
}

// ===================== 8. SMOOTHING =====================

async function runSmoothing() {
  const box = "result-smoothing";
  const blockId = workflow.addBlock("robustness", "Smoothing").id;
  document.getElementById(box).innerHTML = statusPill("running");
  addLog("Running randomized smoothing...", 'info');
  
  try {
    const { job_id } = await postJSON("/api/smoothing/run", {
      sigma: Number(document.getElementById("smooth-sigma").value),
      n_noise_samples: Number(document.getElementById("smooth-n").value),
      max_batches: Number(document.getElementById("smooth-maxbatch").value),
    });
    const job = await pollJob(job_id, { blockId });
    if (job.status === "done") {
      workflow.updateBlockStatus(blockId, "done");
      const r = job.result;
      document.getElementById(box).innerHTML = `${statusPill("done")} smoothed accuracy: <b>${pct(r.smoothed_accuracy)}</b>, radius: <b>${r.mean_certified_radius.toFixed(4)}</b>`;
      addLog(`✓ Smoothing: ${pct(r.smoothed_accuracy)}, radius ${r.mean_certified_radius.toFixed(4)}`, 'success');
    } else {
      workflow.updateBlockStatus(blockId, "error");
      renderError(box, { message: job.error });
    }
    refreshState();
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error");
    renderError(box, e);
  }
}

// ===================== 9. GRAD-CAM =====================

async function runGradcam() {
  const box = "result-gradcam";
  const blockId = workflow.addBlock("analysis", "Grad-CAM").id;
  document.getElementById(box).innerHTML = statusPill("running");
  addLog("Generating Grad-CAM visualizations...", 'info');
  
  try {
    const { job_id } = await postJSON("/api/gradcam/run", {
      mode: document.getElementById("gradcam-mode").value,
      attack: document.getElementById("gradcam-attack").value,
      n_samples: Number(document.getElementById("gradcam-n").value),
    });
    const job = await pollJob(job_id, { blockId });
    if (job.status === "done") {
      workflow.updateBlockStatus(blockId, "done");
      document.getElementById(box).innerHTML = `${statusPill("done")} Generated ${job.result.count} Grad-CAM samples`;
      addLog(`✓ Grad-CAM: ${job.result.count} samples generated`, 'success');
    } else {
      workflow.updateBlockStatus(blockId, "error");
      renderError(box, { message: job.error });
    }
    refreshState();
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error");
    renderError(box, e);
  }
}

// ===================== 10. METRICS =====================

async function runMetrics() {
  const box = "result-metrics";
  const blockId = workflow.addBlock("analysis", "Metrics").id;
  document.getElementById(box).innerHTML = statusPill("running");
  addLog("Computing complexity metrics...", 'info');
  
  try {
    const { job_id } = await postJSON("/api/metrics/run", {});
    const job = await pollJob(job_id, { blockId });
    if (job.status === "done") {
      workflow.updateBlockStatus(blockId, "done");
      const r = job.result;
      document.getElementById(box).innerHTML = `${statusPill("done")} <table><tr><th>Params</th><td>${r.params_millions}M</td></tr><tr><th>FLOPs</th><td>${r.flops_gflops} GFLOPs</td></tr></table>`;
      addLog(`✓ Metrics: ${r.params_millions}M params, ${r.flops_gflops} GFLOPs`, 'success');
    } else {
      workflow.updateBlockStatus(blockId, "error");
      renderError(box, { message: job.error });
    }
    refreshState();
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error");
    renderError(box, e);
  }
}

// ===================== 11. STATISTICS =====================

async function runStats() {
  const box = "result-stats";
  const blockId = workflow.addBlock("analysis", "Statistics").id;
  document.getElementById(box).innerHTML = statusPill("running");
  addLog("Running statistical analysis...", 'info');
  
  try {
    const { job_id } = await postJSON("/api/stats/run", {});
    const job = await pollJob(job_id, { blockId });
    if (job.status === "done") {
      workflow.updateBlockStatus(blockId, "done");
      document.getElementById(box).innerHTML = `${statusPill("done")} ${Object.keys(job.result).length} comparisons complete`;
      addLog(`✓ Statistics: ${Object.keys(job.result).length} comparisons`, 'success');
    } else {
      workflow.updateBlockStatus(blockId, "error");
      renderError(box, { message: job.error });
    }
    refreshState();
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error");
    renderError(box, e);
  }
}

// ===================== STATE REFRESH =====================

async function refreshState() {
  try {
    const s = await getJSON("/api/state");
    
    document.getElementById("device-label").textContent =
      `${s.device} · ${s.dataset || "no dataset"} · ${s.backbone || "no model"}${s.enhancement && s.enhancement !== "none" ? " +" + s.enhancement : ""}`;
    const dot = document.getElementById("device-dot");
    dot.className = "dot " + (s.device.includes("cuda") ? "gpu" : "ok");
    
    // Update results table
    let html = "<table><tr><th>Metric</th><th>Value</th></tr>";
    if (s.clean_eval) html += `<tr><td>Clean accuracy</td><td>${pct(s.clean_eval.accuracy)}</td></tr>`;
    Object.entries(s.attack_results || {}).forEach(([k, v]) => {
      html += `<tr><td>${k} accuracy</td><td>${pct(v.accuracy)}</td></tr>`;
    });
    if (s.smoothing_result) html += `<tr><td>Smoothed accuracy</td><td>${pct(s.smoothing_result.smoothed_accuracy)}</td></tr>`;
    html += "</table>";
    document.getElementById("results-summary").innerHTML = html;
    
    // Update logs
    const logDiv = document.getElementById("log-lines");
    if (s.recent_logs && s.recent_logs.length > 0) {
      logDiv.innerHTML = "";
      s.recent_logs.forEach(log => addLog(log, 'info'));
    }
  } catch (e) {
    console.error("Error refreshing state:", e);
  }
}

// ===================== INITIALIZATION =====================

document.addEventListener('DOMContentLoaded', () => {
  buildAttackGrid();
  refreshState();
  setInterval(refreshState, 3000);
  addLog("System initialized. Click buttons to build workflow.", 'success');
});
