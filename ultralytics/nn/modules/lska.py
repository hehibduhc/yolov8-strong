# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Large Separable Kernel Attention (LSKA) modules."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSKA(nn.Module):
    """Large Separable Kernel Attention with depthwise and depthwise-dilated kernels."""

    def __init__(self, c: int, k: int = 23, d: int = 3):
        super().__init__()
        k_local = 2 * d - 1
        k_global = max(1, k // d)
        p_global = ((k_global - 1) // 2) * d

        self.dw1 = nn.Conv2d(c, c, kernel_size=(1, k_local), padding=(0, k_local // 2), groups=c, bias=False)
        self.dw2 = nn.Conv2d(c, c, kernel_size=(k_local, 1), padding=(k_local // 2, 0), groups=c, bias=False)
        self.dwd1 = nn.Conv2d(
            c, c, kernel_size=(1, k_global), padding=(0, p_global), dilation=(1, d), groups=c, bias=False
        )
        self.dwd2 = nn.Conv2d(
            c, c, kernel_size=(k_global, 1), padding=(p_global, 0), dilation=(d, 1), groups=c, bias=False
        )
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LSKA attention and keep input tensor shape unchanged."""
        attn = self.dw1(x)
        attn = self.dw2(attn)
        attn = self.dwd1(attn)
        attn = self.dwd2(attn)
        attn = self.pw(attn)
        return x * attn


class LSKARes(nn.Module):
    """Residual-safe LSKA variant for weak target preservation."""

    def __init__(self, c: int, k: int = 23, d: int = 3, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.attn = LSKA(c, k=k, d=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Blend attention output with identity using residual interpolation."""
        y = self.attn(x)
        return x + self.alpha * (y - x)


class DirectionalLSKA(nn.Module):
    """Directional LSKA with horizontal/vertical branches and softmax gating."""

    def __init__(self, c: int, k: int = 23, d: int = 3, alpha: float = 0.5):
        super().__init__()
        k_local = 2 * d - 1
        k_global = max(1, k // d)
        p_global = ((k_global - 1) // 2) * d

        self.h_dw = nn.Conv2d(c, c, kernel_size=(1, k_local), padding=(0, k_local // 2), groups=c, bias=False)
        self.h_dwd = nn.Conv2d(
            c, c, kernel_size=(1, k_global), padding=(0, p_global), dilation=(1, d), groups=c, bias=False
        )

        self.v_dw = nn.Conv2d(c, c, kernel_size=(k_local, 1), padding=(k_local // 2, 0), groups=c, bias=False)
        self.v_dwd = nn.Conv2d(
            c, c, kernel_size=(k_global, 1), padding=(p_global, 0), dilation=(d, 1), groups=c, bias=False
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Conv2d(2 * c, 2, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply directional attention and residual-safe fusion with identity."""
        h_attn = self.h_dwd(self.h_dw(x))
        v_attn = self.v_dwd(self.v_dw(x))
        hv = torch.cat((h_attn, v_attn), dim=1)
        w = self.softmax(self.gate(self.gap(hv)))
        attn = h_attn * w[:, 0:1] + v_attn * w[:, 1:2]
        attn = self.pw(attn)
        y = x * attn
        return x + self.alpha * (y - x)

