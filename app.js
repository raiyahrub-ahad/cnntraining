// ===================== WORKFLOW VISUALIZATION SYSTEM =====================

class WorkflowManager {
  constructor() {
    this.canvas = document.getElementById('workflow-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.blocks = [];
    this.connections = [];
    this.selectedBlock = null;
    this.isDragging = false;
    this.dragOffset = { x: 0, y: 0 };
    
    // Colors for different block types
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
    this.canvas.width = window.innerWidth - 40;
    this.canvas.height = 600;
    window.addEventListener('resize', () => this.resizeCanvas());
  }

  resizeCanvas() {
    this.canvas.width = window.innerWidth - 40;
    this.redraw();
  }

  setupEventListeners() {
    this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
    this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
    this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
  }

  addBlock(type, label, x = null, y = null) {
    if (x === null) x = this.blocks.length * 180 + 50;
    if (y === null) y = 150;

    const block = {
      id: Date.now(),
      type: type,
      label: label,
      x: x,
      y: y,
      width: 140,
      height: 80,
      color: this.colors[type] || '#999',
      status: 'pending',
      progress: 0,
      result: null
    };

    this.blocks.push(block);
    this.redraw();
    return block;
  }

  connectBlocks(blockId1, blockId2) {
    this.connections.push({ from: blockId1, to: blockId2 });
    this.redraw();
  }

  updateBlockStatus(blockId, status, progress = 0, result = null) {
    const block = this.blocks.find(b => b.id === blockId);
    if (block) {
      block.status = status;
      block.progress = progress;
      block.result = result;
      this.redraw();
    }
  }

  handleMouseDown(e) {
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
    if (!this.isDragging || !this.selectedBlock) return;

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
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.drawConnections();
    this.drawBlocks();
  }

  drawConnections() {
    this.connections.forEach(conn => {
      const from = this.blocks.find(b => b.id === conn.from);
      const to = this.blocks.find(b => b.id === conn.to);
      if (from && to) {
        this.ctx.strokeStyle = '#999';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        this.ctx.beginPath();
        this.ctx.moveTo(from.x + from.width, from.y + from.height / 2);
        this.ctx.lineTo(to.x, to.y + to.height / 2);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
      }
    });
  }

  drawBlocks() {
    this.blocks.forEach(block => {
      // Draw block background
      this.ctx.fillStyle = block.color;
      this.ctx.globalAlpha = block.status === 'done' ? 1 : (block.status === 'running' ? 0.8 : 0.6);
      this.ctx.fillRect(block.x, block.y, block.width, block.height);

      // Draw border
      this.ctx.strokeStyle = block === this.selectedBlock ? '#000' : '#333';
      this.ctx.lineWidth = block === this.selectedBlock ? 3 : 1;
      this.ctx.strokeRect(block.x, block.y, block.width, block.height);

      this.ctx.globalAlpha = 1;

      // Draw progress bar
      if (block.status === 'running' && block.progress > 0) {
        this.ctx.fillStyle = 'rgba(0,255,0,0.3)';
        this.ctx.fillRect(block.x, block.y + block.height, block.width * block.progress, 5);
      }

      // Draw text
      this.ctx.fillStyle = '#fff';
      this.ctx.font = 'bold 12px Arial';
      this.ctx.textAlign = 'center';
      this.ctx.fillText(block.label, block.x + block.width / 2, block.y + 25);

      // Draw status
      this.ctx.font = '10px Arial';
      this.ctx.fillStyle = '#ddd';
      let statusText = block.status;
      if (block.status === 'running') statusText += ` (${Math.round(block.progress * 100)}%)`;
      this.ctx.fillText(statusText, block.x + block.width / 2, block.y + 50);

      // Draw result preview
      if (block.result) {
        this.ctx.font = '9px Arial';
        this.ctx.fillStyle = '#eee';
        const resultText = typeof block.result === 'string' ? block.result.substring(0, 20) : JSON.stringify(block.result).substring(0, 20);
        this.ctx.fillText(resultText + '...', block.x + block.width / 2, block.y + 65);
      }
    });
  }

  clearAll() {
    this.blocks = [];
    this.connections = [];
    this.redraw();
  }
}

// Initialize workflow manager
let workflow;

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

function addLog(message) {
  const logDiv = document.getElementById("log-lines");
  if (!logDiv) return;
  const logEntry = document.createElement("div");
  logEntry.className = "log-entry";
  logEntry.textContent = new Date().toLocaleTimeString() + " > " + message;
  logDiv.appendChild(logEntry);
  logDiv.scrollTop = logDiv.scrollHeight;
}

function statusPill(status) {
  return `<span class="status-pill ${status}">${status}</span>`;
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

// ===================== 1. DATASET =====================

async function runBuiltinDataset() {
  const name = document.getElementById("builtin-name").value;
  const batch_size = Number(document.getElementById("builtin-batch").value);

  const blockId = workflow.addBlock("dataset", `Dataset: ${name}`).id;
  addLog(`Loading dataset: ${name}...`);

  try {
    const { job_id } = await postJSON("/api/dataset/builtin", {
      name: name,
      batch_size: batch_size,
    });
    const job = await pollJob(job_id, { blockId });
    if (job.status === "done") {
      workflow.updateBlockStatus(blockId, "done", 1, job.result);
      addLog(`✓ Dataset loaded: ${job.result.num_classes} classes, ${job.result.img_size}x${job.result.img_size}`);
    } else {
      workflow.updateBlockStatus(blockId, "error", 0, job.error);
      addLog(`✗ Error loading dataset: ${job.error}`);
    }
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error", 0, e.message);
    addLog(`✗ Error: ${e.message}`);
  }
}

// ===================== 2. MODEL =====================

async function runModelSelect() {
  const name = document.getElementById("model-name").value;
  const pretrained = document.getElementById("model-pretrained").checked;

  const blockId = workflow.addBlock("model", `Model: ${name}`).id;
  addLog(`Selecting model: ${name}${pretrained ? ' (pretrained)' : ''}...`);

  try {
    const r = await postJSON("/api/model/select", {
      name: name,
      pretrained: pretrained,
    });
    workflow.updateBlockStatus(blockId, "done", 1, r);
    addLog(`✓ Model selected: ${(r.params / 1e6).toFixed(2)}M params`);
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error", 0, e.message);
    addLog(`✗ Error: ${e.message}`);
  }
}

// ===================== 3. ENHANCEMENT =====================

async function runEnhance() {
  const kind = document.getElementById("enhance-kind").value;
  const blockId = workflow.addBlock("model", `Enhancement: ${kind}`).id;
  addLog(`Applying enhancement: ${kind}...`);

  try {
    const r = await postJSON("/api/model/enhance", { kind: kind });
    workflow.updateBlockStatus(blockId, "done", 1, r);
    addLog(`✓ Enhancement applied: ${(r.params / 1e6).toFixed(2)}M params`);
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error", 0, e.message);
    addLog(`✗ Error: ${e.message}`);
  }
}

// ===================== 4. CLEAN TRAINING =====================

async function runTrainClean() {
  const epochs = Number(document.getElementById("clean-epochs").value);
  const lr = Number(document.getElementById("clean-lr").value);

  const blockId = workflow.addBlock("train", `Train Clean (${epochs} epochs)`).id;
  addLog(`Starting clean training: ${epochs} epochs, lr=${lr}...`);

  try {
    const { job_id } = await postJSON("/api/train/clean", {
      epochs: epochs,
      lr: lr,
    });
    const job = await pollJob(job_id, {
      blockId,
      onTick: (j) => {
        if (j.message) addLog(j.message);
      }
    });
    if (job.status === "done") {
      workflow.updateBlockStatus(blockId, "done", 1, job.result);
      const lastAcc = job.result.history.val_acc[job.result.history.val_acc.length - 1];
      addLog(`✓ Training complete! Final accuracy: ${pct(lastAcc)}`);
    } else {
      workflow.updateBlockStatus(blockId, "error", 0, job.error);
      addLog(`✗ Training error: ${job.error}`);
    }
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error", 0, e.message);
    addLog(`✗ Error: ${e.message}`);
  }
}

// ===================== 5. ADVERSARIAL TRAINING =====================

async function runTrainAdv() {
  const method = document.getElementById("adv-method").value;
  const epochs = Number(document.getElementById("adv-epochs").value);
  const lr = Number(document.getElementById("adv-lr").value);

  const blockId = workflow.addBlock("train", `Train Adv (${method})`).id;
  addLog(`Starting adversarial training: ${method}, ${epochs} epochs...`);

  try {
    const { job_id } = await postJSON("/api/train/adversarial", {
      method: method,
      epsilon: Number(document.getElementById("adv-eps").value),
      alpha: Number(document.getElementById("adv-alpha").value),
      steps: Number(document.getElementById("adv-steps").value),
      epochs: epochs,
      lr: lr,
      beta: Number(document.getElementById("adv-beta").value),
      awp_gamma: Number(document.getElementById("adv-awpgamma").value),
    });
    const job = await pollJob(job_id, {
      blockId,
      onTick: (j) => {
        if (j.message) addLog(j.message);
      }
    });
    if (job.status === "done") {
      workflow.updateBlockStatus(blockId, "done", 1, job.result);
      const lastAcc = job.result.history.val_acc[job.result.history.val_acc.length - 1];
      addLog(`✓ Training complete! Final accuracy: ${pct(lastAcc)}`);
    } else {
      workflow.updateBlockStatus(blockId, "error", 0, job.error);
      addLog(`✗ Training error: ${job.error}`);
    }
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error", 0, e.message);
    addLog(`✗ Error: ${e.message}`);
  }
}

// ===================== 6. CLEAN EVALUATION =====================

async function runEvalClean() {
  const blockId = workflow.addBlock("evaluate", "Eval: Clean").id;
  addLog("Evaluating clean accuracy...");

  try {
    const { job_id } = await postJSON("/api/eval/clean", {});
    const job = await pollJob(job_id, { blockId });
    if (job.status === "done") {
      workflow.updateBlockStatus(blockId, "done", 1, job.result);
      addLog(`✓ Clean accuracy: ${pct(job.result.accuracy)} (95% CI ${pct(job.result.ci_low)}–${pct(job.result.ci_high)})`);
    } else {
      workflow.updateBlockStatus(blockId, "error", 0, job.error);
      addLog(`✗ Error: ${job.error}`);
    }
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error", 0, e.message);
    addLog(`✗ Error: ${e.message}`);
  }
}

// ===================== 7. ATTACKS =====================

async function runAttack(key) {
  const blockId = workflow.addBlock("attack", `Attack: ${key.toUpperCase()}`).id;
  addLog(`Running ${key} attack...`);

  try {
    const { job_id } = await postJSON("/api/attack/run", {
      attack: key,
      epsilon: Number(document.getElementById(`atk-${key}-eps`).value),
      alpha: Number(document.getElementById(`atk-${key}-alpha`).value),
      steps: Number(document.getElementById(`atk-${key}-steps`).value),
      n_samples: Number(document.getElementById(`atk-${key}-n`).value),
    });
    const job = await pollJob(job_id, { blockId });
    if (job.status === "done") {
      workflow.updateBlockStatus(blockId, "done", 1, job.result);
      addLog(`✓ Attack complete! Adversarial accuracy: ${pct(job.result.accuracy)}`);
    } else {
      workflow.updateBlockStatus(blockId, "error", 0, job.error);
      addLog(`✗ Error: ${job.error}`);
    }
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error", 0, e.message);
    addLog(`✗ Error: ${e.message}`);
  }
}

// ===================== 8. SMOOTHING =====================

async function runSmoothing() {
  const blockId = workflow.addBlock("robustness", "Smoothing").id;
  addLog("Running randomized smoothing...");

  try {
    const { job_id } = await postJSON("/api/smoothing/run", {
      sigma: Number(document.getElementById("smooth-sigma").value),
      n_noise_samples: Number(document.getElementById("smooth-n").value),
      max_batches: Number(document.getElementById("smooth-maxbatch").value),
    });
    const job = await pollJob(job_id, { blockId });
    if (job.status === "done") {
      workflow.updateBlockStatus(blockId, "done", 1, job.result);
      addLog(`✓ Smoothing complete! Smoothed accuracy: ${pct(job.result.smoothed_accuracy)}, radius: ${job.result.mean_certified_radius.toFixed(4)}`);
    } else {
      workflow.updateBlockStatus(blockId, "error", 0, job.error);
      addLog(`✗ Error: ${job.error}`);
    }
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error", 0, e.message);
    addLog(`✗ Error: ${e.message}`);
  }
}

// ===================== 9. GRAD-CAM =====================

async function runGradcam() {
  const blockId = workflow.addBlock("analysis", "Grad-CAM").id;
  addLog("Generating Grad-CAM visualizations...");

  try {
    const { job_id } = await postJSON("/api/gradcam/run", {
      mode: document.getElementById("gradcam-mode").value,
      attack: document.getElementById("gradcam-attack").value,
      n_samples: Number(document.getElementById("gradcam-n").value),
    });
    const job = await pollJob(job_id, { blockId });
    if (job.status === "done") {
      workflow.updateBlockStatus(blockId, "done", 1, job.result);
      addLog(`✓ Grad-CAM complete! Generated ${job.result.count} visualizations`);
    } else {
      workflow.updateBlockStatus(blockId, "error", 0, job.error);
      addLog(`✗ Error: ${job.error}`);
    }
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error", 0, e.message);
    addLog(`✗ Error: ${e.message}`);
  }
}

// ===================== 10. METRICS =====================

async function runMetrics() {
  const blockId = workflow.addBlock("analysis", "Metrics").id;
  addLog("Computing complexity metrics...");

  try {
    const { job_id } = await postJSON("/api/metrics/run", {});
    const job = await pollJob(job_id, { blockId });
    if (job.status === "done") {
      workflow.updateBlockStatus(blockId, "done", 1, job.result);
      addLog(`✓ Metrics: ${job.result.params_millions}M params, ${job.result.flops_gflops} GFLOPs`);
    } else {
      workflow.updateBlockStatus(blockId, "error", 0, job.error);
      addLog(`✗ Error: ${job.error}`);
    }
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error", 0, e.message);
    addLog(`✗ Error: ${e.message}`);
  }
}

// ===================== 11. STATISTICS =====================

async function runStats() {
  const blockId = workflow.addBlock("analysis", "Statistics").id;
  addLog("Running statistical analysis...");

  try {
    const { job_id } = await postJSON("/api/stats/run", {});
    const job = await pollJob(job_id, { blockId });
    if (job.status === "done") {
      workflow.updateBlockStatus(blockId, "done", 1, job.result);
      addLog(`✓ Statistical analysis complete! ${Object.keys(job.result).length} comparisons`);
    } else {
      workflow.updateBlockStatus(blockId, "error", 0, job.error);
      addLog(`✗ Error: ${job.error}`);
    }
  } catch (e) {
    workflow.updateBlockStatus(blockId, "error", 0, e.message);
    addLog(`✗ Error: ${e.message}`);
  }
}

// ===================== STATE REFRESH =====================

async function refreshState() {
  try {
    const s = await getJSON("/api/state");
    document.getElementById("device-label").textContent =
      `${s.device} · ${s.dataset || "no dataset"} · ${s.backbone || "no model"}`;
    
    // Update results summary
    let html = "<table class='results-table'><tr><th>Metric</th><th>Value</th></tr>";
    if (s.clean_eval) html += `<tr><td>Clean accuracy</td><td><span class="badge success">${pct(s.clean_eval.accuracy)}</span></td></tr>`;
    Object.entries(s.attack_results || {}).forEach(([k, v]) => {
      html += `<tr><td>${k} adversarial accuracy</td><td><span class="badge warning">${pct(v.accuracy)}</span></td></tr>`;
    });
    if (s.smoothing_result) html += `<tr><td>Smoothed accuracy</td><td><span class="badge info">${pct(s.smoothing_result.smoothed_accuracy)}</span></td></tr>`;
    html += "</table>";
    document.getElementById("results-summary").innerHTML = html;
  } catch (e) {
    console.error("Error refreshing state:", e);
  }
}

// ===================== INITIALIZATION =====================

document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('workflow-canvas')) {
    workflow = new WorkflowManager();
    addLog("System initialized. Ready to build workflow...");
  }
  refreshState();
  setInterval(refreshState, 4000);
});
