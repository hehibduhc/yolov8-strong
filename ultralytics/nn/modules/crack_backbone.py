# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Crack-segmentation-oriented backbone modules (SPD-Conv + Dynamic Snake Conv)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class SpaceToDepth(nn.Module):
    """Rearrange spatial blocks into channels."""

    def __init__(self, block_size: int = 2):
        super().__init__()
        if block_size <= 1:
            raise ValueError(f"block_size must be > 1, but got {block_size}")
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        bs = self.block_size
        if h % bs != 0 or w % bs != 0:
            raise ValueError(f"Input spatial size {(h, w)} must be divisible by block_size={bs}")
        x = x.view(b, c, h // bs, bs, w // bs, bs)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(b, c * bs * bs, h // bs, w // bs)


class SPDConv(nn.Module):
    """Space-to-depth followed by non-strided convolution, used to replace stride-2 conv."""

    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True, block_size=2):
        super().__init__()
        if s != 2:
            raise ValueError(f"SPDConv only supports s=2 for downsampling replacement, but got s={s}")
        self.stod = SpaceToDepth(block_size=block_size)
        self.conv = Conv(c1 * block_size * block_size, c2, k=k, s=1, p=p, g=g, d=d, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.stod(x))


class DSConv2d(nn.Module):
    """Dynamic Snake Convolution in pure PyTorch with grid_sample-based deformation."""

    def __init__(
        self,
        c1,
        c2,
        kernel_size=7,
        extend_scope=1.0,
        morph=0,
        if_offset=True,
        act=True,
    ):
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError(f"kernel_size must be odd and >= 3, got {kernel_size}")
        if morph not in (0, 1):
            raise ValueError(f"morph must be 0 (x-axis) or 1 (y-axis), got {morph}")
        self.kernel_size = kernel_size
        self.extend_scope = float(extend_scope)
        self.morph = morph
        self.if_offset = if_offset

        self.offset_conv = nn.Conv2d(c1, 2 * kernel_size, kernel_size=3, stride=1, padding=1, bias=True)
        self.offset_norm = nn.BatchNorm2d(2 * kernel_size)
        self.proj = nn.Conv3d(c1, c2, kernel_size=(kernel_size, 1, 1), stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def _progressive(self, delta: torch.Tensor) -> torch.Tensor:
        """Progressively build snake offsets from center towards both directions."""
        b, k, h, w = delta.shape
        center = k // 2
        out = delta.new_zeros((b, k, h, w))
        for i in range(center + 1, k):
            out[:, i] = out[:, i - 1] + delta[:, i]
        for i in range(center - 1, -1, -1):
            out[:, i] = out[:, i + 1] + delta[:, i]
        return out

    def _sample(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        k = self.kernel_size
        center = k // 2

        delta_a, delta_b = torch.chunk(offsets, 2, dim=1)  # [B, k, H, W], [B, k, H, W]
        delta_a = self._progressive(delta_a) * self.extend_scope
        delta_b = delta_b * self.extend_scope

        yy, xx = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype),
            torch.arange(w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        yy = yy.view(1, 1, h, w)
        xx = xx.view(1, 1, h, w)
        base = torch.arange(-center, center + 1, device=x.device, dtype=x.dtype).view(1, k, 1, 1)

        if self.morph == 0:  # snake along x-axis
            grid_x = xx + base + delta_a
            grid_y = yy + delta_b
        else:  # snake along y-axis
            grid_x = xx + delta_b
            grid_y = yy + base + delta_a

        grid_x = 2.0 * grid_x / max(w - 1, 1) - 1.0
        grid_y = 2.0 * grid_y / max(h - 1, 1) - 1.0

        grid = torch.stack((grid_x, grid_y), dim=-1).reshape(b * k, h, w, 2)
        x_rep = x.repeat_interleave(k, dim=0)
        sampled = F.grid_sample(x_rep, grid, mode="bilinear", padding_mode="border", align_corners=True)
        sampled = sampled.view(b, k, x.shape[1], h, w).permute(0, 2, 1, 3, 4).contiguous()  # [B, C, K, H, W]
        return sampled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offsets = torch.tanh(self.offset_norm(self.offset_conv(x))) if self.if_offset else x.new_zeros(
            (x.shape[0], 2 * self.kernel_size, x.shape[2], x.shape[3])
        )
        sampled = self._sample(x, offsets)
        y = self.proj(sampled).squeeze(2)
        return self.act(self.bn(y))


class DSCBottleNeck(nn.Module):
    """Bottleneck block with dual-axis Dynamic Snake Convolution branches."""

    def __init__(self, c1, c2, shortcut=True, e=0.5, kernel_size=7, extend_scope=1.0):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.ds_x = DSConv2d(c_, c_, kernel_size=kernel_size, extend_scope=extend_scope, morph=0, if_offset=True)
        self.ds_y = DSConv2d(c_, c_, kernel_size=kernel_size, extend_scope=extend_scope, morph=1, if_offset=True)
        self.cv2 = Conv(2 * c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        y = self.cv2(torch.cat((self.ds_x(y), self.ds_y(y)), dim=1))
        return x + y if self.add else y


class C2fDSC(nn.Module):
    """C2f-compatible block using DSCBottleNeck internally."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        g=1,
        e=0.5,
        kernel_size=7,
        extend_scope=1.0,
    ):
        super().__init__()
        _ = g  # parser compatibility
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            DSCBottleNeck(
                self.c,
                self.c,
                shortcut=shortcut,
                e=1.0,
                kernel_size=kernel_size,
                extend_scope=extend_scope,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
