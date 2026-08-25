"""Model complexity / efficiency metrics: parameter count, FLOPs, inference
latency — standard "required results" for a CNN research paper's methods
comparison table."""

import time
import numpy as np
import torch
from thop import profile


def flops_and_params(model, input_shape, device):
    model.eval()
    dummy = torch.randn(1, *input_shape).to(device)
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    return {
        "params": int(params),
        "params_millions": round(params / 1e6, 3),
        "macs": float(macs),
        "flops_gflops": round((macs * 2) / 1e9, 3),
    }


def inference_latency(model, input_shape, device, n_runs=50, batch_size=1, warmup=10):
    model.eval()
    dummy = torch.randn(batch_size, *input_shape).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            t0 = time.time()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }
