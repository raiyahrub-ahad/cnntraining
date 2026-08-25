"""Architectural enhancements that can be spliced onto any backbone's feature
extractor with one click: classic Pyramid Pooling (PSPNet-style), an Adaptive
Pyramid Pooling variant with learned scale-gating, and a Feature Denoising
block (non-local means style, after Xie et al., 'Feature Denoising for
Improving Adversarial Robustness') as a robustness-oriented architectural
technique."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import get_feature_extractor


class PyramidPoolingModule(nn.Module):
    """Spatial Pyramid Pooling Module (PSPNet-style). Pools the feature map at
    several scales, projects each, upsamples and concatenates, then fuses."""

    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6), reduction=4):
        super().__init__()
        out_c = max(in_channels // reduction, 8)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ) for ps in pool_sizes
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + out_c * len(pool_sizes), in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = in_channels

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        feats = [x]
        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            feats.append(y)
        return self.fuse(torch.cat(feats, dim=1))


class AdaptivePyramidPooling(nn.Module):
    """Adaptive variant: the same multi-scale branches, but their contribution
    is combined with a learned, input-conditioned softmax gate instead of a
    fixed concatenation — the network learns which scale matters most for a
    given input, and the module is applied as a residual refinement."""

    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6), reduction=4):
        super().__init__()
        out_c = max(in_channels // reduction, 8)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ) for ps in pool_sizes
        ])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, len(pool_sizes), kernel_size=1),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_c, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = in_channels

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        weights = torch.softmax(self.gate(x), dim=1)  # B, n_branches, 1, 1
        acc = 0
        for i, branch in enumerate(self.branches):
            y = branch(x)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            acc = acc + y * weights[:, i:i + 1]
        return x + self.project(acc)


class FeatureDenoiseBlock(nn.Module):
    """Non-local-means style feature denoising block. Adversarial perturbations
    show up as noisy activations in deep feature maps; this block computes a
    non-local weighted average of each spatial location against all others and
    adds it back as a residual, which empirically suppresses adversarial noise
    (Xie et al., CVPR 2019, 'Feature Denoising for Improving Adversarial
    Robustness'). Included here as a 'novel' robustness-oriented architectural
    enhancement, distinct from the pyramid pooling modules."""

    def __init__(self, in_channels, key_channels=None):
        super().__init__()
        key_channels = key_channels or max(in_channels // 8, 8)
        self.theta = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.out_channels = in_channels

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        theta = self.theta(x).view(b, -1, n)              # B,K,N
        phi = self.phi(x).view(b, -1, n)                   # B,K,N
        g = self.g(x).view(b, c, n)                         # B,C,N

        attn = torch.softmax(torch.bmm(theta.transpose(1, 2), phi) / (theta.shape[1] ** 0.5), dim=-1)  # B,N,N
        y = torch.bmm(g, attn.transpose(1, 2)).view(b, c, h, w)  # non-local weighted average
        y = self.out_proj(y)
        return x + y  # residual: denoised signal added back to the original activations


ENHANCEMENT_REGISTRY = {
    "none": "No enhancement (baseline backbone)",
    "spp": "Pyramid Pooling Module (multi-scale context, PSPNet-style)",
    "adaptive_ppm": "Adaptive Pyramid Pooling (learned scale gating)",
    "feature_denoise": "Feature Denoising Block (non-local, robustness-oriented)",
}


class EnhancedCNN(nn.Module):
    def __init__(self, backbone, backbone_name, num_classes, enhancement="none"):
        super().__init__()
        feat_extractor, feat_channels = get_feature_extractor(backbone, backbone_name)
        self.feature_extractor = feat_extractor
        if enhancement == "spp":
            self.enhancement = PyramidPoolingModule(feat_channels)
        elif enhancement == "adaptive_ppm":
            self.enhancement = AdaptivePyramidPooling(feat_channels)
        elif enhancement == "feature_denoise":
            self.enhancement = FeatureDenoiseBlock(feat_channels)
        else:
            self.enhancement = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feat_channels, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.enhancement(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def apply_enhancement(backbone, backbone_name, num_classes, enhancement):
    if enhancement in (None, "none"):
        return backbone
    return EnhancedCNN(backbone, backbone_name, num_classes, enhancement)
