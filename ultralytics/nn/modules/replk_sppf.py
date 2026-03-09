# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn

from .conv import Conv


class LargeKernelDW(nn.Module):
    """Depthwise large-kernel convolution followed by pointwise channel mixing."""

    def __init__(self, c: int, k: int = 13, bias: bool = False):
        """Initialize depthwise large-kernel block with shape-preserving output."""
        super().__init__()
        self.dw = nn.Conv2d(c, c, kernel_size=k, stride=1, padding=k // 2, groups=c, bias=bias)
        self.pw = Conv(c, c, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply large-kernel depthwise convolution then pointwise fusion."""
        return self.pw(self.dw(x))


class SPPFRepLKLite(nn.Module):
    """SPPF followed by lightweight RepLK-style large-kernel residual branch."""

    def __init__(self, c1: int, c2: int, k: int = 5, lk: int = 13, alpha: float = 1.0):
        """Initialize SPPF backbone block with a single large-kernel depthwise branch."""
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lk = LargeKernelDW(c2, k=lk)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SPPF then add scaled large-kernel residual refinement."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        y = self.cv2(torch.cat(y, 1))
        lk_out = self.lk(y)
        return y + self.alpha * lk_out


class RepLKBranch(nn.Module):
    """RepLK-inspired parallel depthwise large/small kernel branch with pointwise fusion."""

    def __init__(self, c: int, large_k: int = 17, small_k: int = 5):
        """Initialize parallel depthwise branches and 1x1 channel fusion."""
        super().__init__()
        self.large = nn.Conv2d(c, c, kernel_size=large_k, stride=1, padding=large_k // 2, groups=c, bias=False)
        self.small = nn.Conv2d(c, c, kernel_size=small_k, stride=1, padding=small_k // 2, groups=c, bias=False)
        self.pw = Conv(c, c, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse large/small depthwise features and mix channels with pointwise conv."""
        return self.pw(self.large(x) + self.small(x))


class SPPFRepLKFull(nn.Module):
    """SPPF followed by full RepLK-style parallel branch residual refinement."""

    def __init__(self, c1: int, c2: int, k: int = 5, large_k: int = 17, small_k: int = 5, alpha: float = 1.0):
        """Initialize SPPF plus parallel large/small-kernel residual branch."""
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.replk = RepLKBranch(c2, large_k=large_k, small_k=small_k)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SPPF then add scaled RepLK parallel-branch residual output."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        y = self.cv2(torch.cat(y, 1))
        z = self.replk(y)
        return y + self.alpha * z
