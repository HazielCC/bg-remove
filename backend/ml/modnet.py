"""
MODNet architecture adapted for Apple Silicon (MPS) / CPU / CUDA.

Based on ZHKKKe/MODNet (AAAI 2022).
Three branches:
  S (Semantic)  — LRBranch  — 1/16x resolution
  O (Detail)    — HRBranch  — full resolution
  C (Fusion)    — FusionBranch — full resolution → alpha matte

Key changes from original:
  - Removed all .cuda() calls → uses configurable device
  - Removed nn.DataParallel wrapper
  - GaussianBlurLayer uses .to(device) instead of .cuda()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import math


# ── Gaussian Blur ────────────────────────────────────────
class GaussianBlurLayer(nn.Module):
    """Apply gaussian smoothing on a tensor (1d, 2d, 3d)."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert kernel_size % 2 != 0, "kernel_size must be odd"
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        kernel = self._make_kernel(kernel_size, sigma)
        self.register_buffer("weight", kernel)
        self.groups = channels
        self.pad = kernel_size // 2

    @staticmethod
    def _make_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = torch.outer(g, g)
        g /= g.sum()
        return g.unsqueeze(0).unsqueeze(0)  # shape [1,1,k,k]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.expand(self.channels, -1, -1, -1).to(x.device)
        return F.conv2d(x, w, padding=self.pad, groups=self.groups)


# ── Encoder (MobileNetV2) ───────────────────────────────
class MobileNetV2Encoder(nn.Module):
    """
    MobileNetV2 backbone. Returns intermediate feature maps at
    1/2, 1/4, 1/8, 1/16, 1/32 scales.
    Channels: [16, 24, 32, 96, 1280]
    """

    STAGE_INDICES = [1, 3, 6, 13, 18]  # layer indices for feature extraction

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            net = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            net = mobilenet_v2(weights=None)
        self.features = net.features
        self.enc_channels = [16, 24, 32, 96, 1280]

    def forward(self, x: torch.Tensor):
        feats = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.STAGE_INDICES:
                feats[i] = x
        return [feats[i] for i in self.STAGE_INDICES]


# ── Low-Resolution Branch (Semantic) ────────────────────
class LRBranch(nn.Module):
    """Semantic estimation branch. Outputs a coarse segmentation at 1/16 scale."""

    def __init__(self, enc_channels: list[int]):
        super().__init__()
        self.se_block = nn.Sequential(
            nn.Conv2d(enc_channels[4], enc_channels[3], 1, bias=False),
            nn.BatchNorm2d(enc_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_channels[3], enc_channels[3], 1, bias=False),
            nn.BatchNorm2d(enc_channels[3]),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(enc_channels[3], 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, enc_feats: list[torch.Tensor]):
        f4 = enc_feats[4]  # 1/32 scale, 1280ch
        f3 = enc_feats[3]  # 1/16 scale, 96ch

        f4_up = F.interpolate(
            f4, size=f3.shape[2:], mode="bilinear", align_corners=False
        )
        x = self.se_block(f4_up)
        x = x + f3
        pred_semantic = self.head(x)
        return pred_semantic, x  # semantic pred + feature for fusion


# ── High-Resolution Branch (Detail) ─────────────────────
class HRBranch(nn.Module):
    """Detail estimation branch. Focuses on edges/boundary at full resolution."""

    def __init__(self, enc_channels: list[int], hr_channels: int = 32):
        super().__init__()
        self.conv_enc2 = nn.Sequential(
            nn.Conv2d(enc_channels[0], hr_channels, 1, bias=False),
            nn.BatchNorm2d(hr_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_enc3 = nn.Sequential(
            nn.Conv2d(enc_channels[1], hr_channels, 1, bias=False),
            nn.BatchNorm2d(hr_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(3 + hr_channels * 2, hr_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hr_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(hr_channels + 1, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(
        self,
        img: torch.Tensor,
        enc_feats: list[torch.Tensor],
        pred_semantic: torch.Tensor,
    ):
        h, w = img.shape[2:]
        f0 = self.conv_enc2(enc_feats[0])
        f0_up = F.interpolate(f0, size=(h, w), mode="bilinear", align_corners=False)

        f1 = self.conv_enc3(enc_feats[1])
        f1_up = F.interpolate(f1, size=(h, w), mode="bilinear", align_corners=False)

        sem_up = F.interpolate(
            pred_semantic, size=(h, w), mode="bilinear", align_corners=False
        )

        x = torch.cat([img, f0_up, f1_up], dim=1)
        x = self.conv_fuse(x)
        x = torch.cat([x, sem_up], dim=1)
        pred_detail = self.conv_out(x)
        return pred_detail


# ── Fusion Branch ────────────────────────────────────────
class FusionBranch(nn.Module):
    """Fuse semantic + detail → final alpha matte."""

    def __init__(self, enc_channels: list[int], hr_channels: int = 32):
        super().__init__()
        self.conv_lr = nn.Sequential(
            nn.Conv2d(enc_channels[3], hr_channels, 1, bias=False),
            nn.BatchNorm2d(hr_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(2 + hr_channels, hr_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hr_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hr_channels, hr_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hr_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hr_channels, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(
        self,
        pred_semantic: torch.Tensor,
        pred_detail: torch.Tensor,
        lr_feat: torch.Tensor,
    ):
        h, w = pred_detail.shape[2:]
        lr_up = F.interpolate(
            lr_feat, size=(h, w), mode="bilinear", align_corners=False
        )
        lr_up = self.conv_lr(lr_up)
        x = torch.cat(
            [
                (
                    pred_semantic.expand(-1, -1, h, w)
                    if pred_semantic.shape[2:] != (h, w)
                    else pred_semantic
                ),
                pred_detail,
                lr_up,
            ],
            dim=1,
        )
        # Ensure semantic is upsampled if needed
        if x.shape[2:] != (h, w):
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        # Re-concat properly
        sem_up = (
            F.interpolate(
                pred_semantic, size=(h, w), mode="bilinear", align_corners=False
            )
            if pred_semantic.shape[2:] != (h, w)
            else pred_semantic
        )
        x = torch.cat([sem_up, pred_detail, lr_up], dim=1)
        x = self.conv_fuse(x)
        pred_matte = self.head(x)
        return pred_matte


# ── MODNet ───────────────────────────────────────────────
class MODNet(nn.Module):
    """
    MODNet: Trimap-Free Portrait Matting in Real Time.

    Args:
        backbone_pretrained: load ImageNet weights for MobileNetV2
        hr_channels: channels for the high-resolution branch
    """

    def __init__(self, backbone_pretrained: bool = True, hr_channels: int = 32):
        super().__init__()
        self.encoder = MobileNetV2Encoder(pretrained=backbone_pretrained)
        enc_ch = self.encoder.enc_channels

        self.lr_branch = LRBranch(enc_ch)
        self.hr_branch = HRBranch(enc_ch, hr_channels)
        self.fusion_branch = FusionBranch(enc_ch, hr_channels)

        self.blurer = GaussianBlurLayer(1, 3)

    def freeze_norm(self):
        """Freeze all BatchNorm layers (used during SOC adaptation)."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, img: torch.Tensor, inference: bool = False):
        """
        Args:
            img: [B, 3, H, W] normalized to [-1, 1]
            inference: if True, only return pred_matte
        Returns:
            if inference: pred_matte [B, 1, H, W]
            else: (pred_semantic, pred_detail, pred_matte)
        """
        enc_feats = self.encoder(img)

        pred_semantic, lr_feat = self.lr_branch(enc_feats)
        pred_detail = self.hr_branch(img, enc_feats, pred_semantic)
        pred_matte = self.fusion_branch(pred_semantic, pred_detail, lr_feat)

        if inference:
            return pred_matte

        return pred_semantic, pred_detail, pred_matte

    @classmethod
    def from_pretrained(
        cls, ckpt_path: str, device: str = "cpu", backbone_pretrained: bool = False
    ) -> "MODNet":
        """Load from a .ckpt file (official MODNet format)."""
        model = cls(backbone_pretrained=backbone_pretrained)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

        # Handle DataParallel wrapped state_dict (keys start with 'module.')
        cleaned = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            cleaned[new_key] = v

        model.load_state_dict(cleaned, strict=False)
        model.to(device)
        model.eval()
        return model


# ── Inference-only wrapper (for ONNX export) ────────────
class MODNetInference(nn.Module):
    """Thin wrapper that only returns pred_matte (for ONNX export)."""

    def __init__(self, modnet: MODNet):
        super().__init__()
        self.modnet = modnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.modnet(x, inference=True)
