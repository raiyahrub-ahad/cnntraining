"""Grad-CAM: generic implementation that auto-locates the last Conv2d layer
of whatever model is currently active (works across all backbones and the
enhanced/wrapped variants)."""

import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm


def _find_last_conv(model):
    last = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last = module
    return last


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or _find_last_conv(model)
        if self.target_layer is None:
            raise ValueError("No Conv2d layer found for Grad-CAM.")
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, target_class=None):
        self.model.eval()
        x = x.clone().requires_grad_(True)
        out = self.model(x)
        if target_class is None:
            target_class = out.argmax(1)
        score = out.gather(1, target_class.view(-1, 1)).sum()
        self.model.zero_grad()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach().cpu(), out.detach()


def overlay_cam_on_image(img_tensor, cam, alpha=0.45):
    """img_tensor: CxHxW in [0,1]. cam: HxW in [0,1]. Returns base64 PNG string."""
    img = img_tensor.permute(1, 2, 0).numpy()
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    img = np.clip(img, 0, 1)
    heatmap = cm.jet(cam.numpy())[..., :3]
    overlay = (1 - alpha) * img + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    pil_img = Image.fromarray((overlay * 255).astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def tensor_to_base64(img_tensor):
    """Plain (non-overlaid) image tensor -> base64 PNG, for side-by-side
    clean/adversarial example display."""
    img = img_tensor.permute(1, 2, 0).numpy()
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    img = np.clip(img, 0, 1)
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
