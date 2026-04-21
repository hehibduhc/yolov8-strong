# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Custom segmentation building blocks for YOLOv8 variants."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv


class PConv(nn.Module):
    """FasterNet-style partial convolution operating on a channel subset."""

    def __init__(self, c1: int, kernel_size: int = 3, n_div: int = 4, forward_mode: str = "split_cat"):
        super().__init__()
        if forward_mode != "split_cat":
            raise ValueError(f"PConv only supports forward_mode='split_cat', but got {forward_mode}.")
        if n_div <= 0:
            raise ValueError(f"n_div must be > 0, but got {n_div}.")
        self.partial_channels = c1 // n_div
        if self.partial_channels < 1:
            raise ValueError(f"c1 // n_div must be >= 1, but got c1={c1}, n_div={n_div}.")
        self.identity_channels = c1 - self.partial_channels
        self.partial_conv = nn.Conv2d(
            self.partial_channels,
            self.partial_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial conv on a partial channel split and concatenate with identity channels."""
        x1, x2 = torch.split(x, [self.partial_channels, self.identity_channels], dim=1)
        return torch.cat((self.partial_conv(x1), x2), dim=1)


class FasterBlock(nn.Module):
    """FasterNet-style block: partial spatial mixing + MLP-like 1x1 expansion/projection."""

    def __init__(
        self,
        c1: int,
        c2: int,
        mlp_ratio: float = 2.0,
        n_div: int = 4,
        shortcut: bool = True,
        act: bool = True,
    ):
        super().__init__()
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be > 0, but got {mlp_ratio}.")
        hidden = max(int(c2 * mlp_ratio), c2)
        self.spatial_mixing = PConv(c1, kernel_size=3, n_div=n_div, forward_mode="split_cat")
        self.expand = Conv(c1, hidden, k=1, s=1, act=act)
        self.project = Conv(hidden, c2, k=1, s=1, act=False)
        self.use_shortcut = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with optional residual connection."""
        y = self.project(self.expand(self.spatial_mixing(x)))
        return x + y if self.use_shortcut else y


class C2fFaster(nn.Module):
    """C2f-compatible module with FasterBlock stack."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        mlp_ratio: float = 2.0,
        n_div: int = 4,
    ):
        super().__init__()
        _ = g  # kept for parser/API compatibility
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            FasterBlock(self.c, self.c, mlp_ratio=mlp_ratio, n_div=n_div, shortcut=shortcut) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Match C2f forward behavior."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class GSConv(nn.Module):
    """GSConv with standard + depthwise branch and channel shuffle."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1, act: bool = True):
        super().__init__()
        _ = g  # API compatibility
        c_ = c2 // 2
        if c_ < 1:
            raise ValueError(f"GSConv requires c2 >= 2, but got c2={c2}.")
        self.c2 = c2
        self.cv1 = Conv(c1, c_, k, s, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, g=c_, act=act)

    @staticmethod
    def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
        """Shuffle channels between groups."""
        b, c, h, w = x.shape
        if c % groups != 0:
            raise ValueError(f"channels={c} not divisible by groups={groups}.")
        x = x.view(b, groups, c // groups, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward GSConv and keep exact output channels."""
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        y = torch.cat((x1, x2), dim=1)
        y = self.channel_shuffle(y, groups=2)
        return y[:, : self.c2, :, :]


class GSBottleneck(nn.Module):
    """Bottleneck based on GSConv."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = GSConv(c1, c_, k=1, s=1)
        self.cv2 = GSConv(c_, c2, k=3, s=1)
        self.use_shortcut = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with optional residual."""
        y = self.cv2(self.cv1(x))
        return x + y if self.use_shortcut else y


class VoVGSCSP(nn.Module):
    """VoV-GSCSP style CSP fusion block for lightweight necks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        _ = g  # API compatibility
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.blocks = nn.Sequential(*(GSBottleneck(c_, c_, shortcut=shortcut, e=1.0) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse transformed and shortcut branches."""
        y1 = self.blocks(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class EMA(nn.Module):
    """Efficient Multi-Scale Attention module."""

    def __init__(self, c1: int, factor: int = 8):
        super().__init__()
        if factor <= 0:
            raise ValueError(f"factor must be > 0, but got {factor}.")
        groups = max(1, min(factor, c1))
        while c1 % groups != 0 and groups > 1:
            groups -= 1
        self.groups = groups
        gc = c1 // self.groups

        self.conv1x1 = nn.Conv2d(gc, gc, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(gc, gc, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn = nn.GroupNorm(1, gc)
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply grouped cross-spatial attention."""
        b, c, h, w = x.shape
        xg = x.reshape(b * self.groups, c // self.groups, h, w)

        x_h = xg.mean(dim=3, keepdim=True)
        x_w = xg.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat((x_h, x_w), dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        gate_h = x_h.sigmoid()
        gate_w = x_w.permute(0, 1, 3, 2).sigmoid()

        x1 = self.gn(xg * gate_h * gate_w)
        x2 = self.conv3x3(xg)

        w1 = self.softmax(self.agp(x1).reshape(b * self.groups, 1, -1))
        w2 = self.softmax(self.agp(x2).reshape(b * self.groups, 1, -1))

        x1_flat = x1.reshape(b * self.groups, c // self.groups, -1)
        x2_flat = x2.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.bmm(w1, x2_flat) + torch.bmm(w2, x1_flat)).reshape(b * self.groups, 1, h, w)

        out = (xg * weights.sigmoid()).reshape(b, c, h, w)
        return out
