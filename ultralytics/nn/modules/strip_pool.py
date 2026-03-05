# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Lightweight strip pooling modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class StripPoolingLite(nn.Module):
    """Lightweight strip pooling attention with horizontal and vertical context branches."""

    def __init__(self, c: int, reduction: int = 4):
        """Initialize a lightweight strip pooling module.

        Args:
            c (int): Input/output channels.
            reduction (int): Channel reduction ratio.
        """
        super().__init__()
        c_mid = max(8, c // reduction)
        self.conv_reduce = Conv(c, c_mid, k=1, s=1)
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_h = Conv(c_mid, c_mid, k=(1, 3), s=1)
        self.conv_w = Conv(c_mid, c_mid, k=(3, 1), s=1)
        self.conv_attn = Conv(c_mid, c, k=1, s=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply strip pooling attention and preserve input shape."""
        _, _, h, w = x.shape
        y = self.conv_reduce(x)
        h_feat = self.conv_h(self.pool_h(y))
        w_feat = self.conv_w(self.pool_w(y))
        h_feat = F.interpolate(h_feat, size=(h, w), mode="bilinear", align_corners=False)
        w_feat = F.interpolate(w_feat, size=(h, w), mode="bilinear", align_corners=False)
        attn = torch.sigmoid(self.conv_attn(h_feat + w_feat))
        return x * attn


class StripPoolingLiteRes(nn.Module):
    """Residual-safe lightweight strip pooling."""

    def __init__(self, c: int, reduction: int = 4, alpha: float = 0.5):
        """Initialize residual strip pooling.

        Args:
            c (int): Input/output channels.
            reduction (int): Channel reduction ratio.
            alpha (float): Residual interpolation factor.
        """
        super().__init__()
        self.sp = StripPoolingLite(c, reduction=reduction)
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual strip pooling interpolation."""
        y = self.sp(x)
        return x + self.alpha * (y - x)


class StripPoolingAtrous(nn.Module):
    """Residual strip pooling followed by lightweight depthwise atrous enhancement."""

    def __init__(self, c: int, reduction: int = 4, dilations: tuple[int, int] = (2, 3)):
        """Initialize atrous strip pooling enhancement.

        Args:
            c (int): Input/output channels.
            reduction (int): Channel reduction ratio.
            dilations (tuple[int, int]): Depthwise dilation rates.
        """
        super().__init__()
        d2, d3 = dilations
        self.pre = StripPoolingLiteRes(c, reduction=reduction, alpha=0.5)
        self.dw_d2 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=d2, dilation=d2, groups=c, bias=False)
        self.dw_d3 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=d3, dilation=d3, groups=c, bias=False)
        self.fuse = Conv(c, c, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply strip pooling and atrous fusion while preserving shape."""
        y = self.pre(x)
        return self.fuse(self.dw_d2(y) + self.dw_d3(y))
