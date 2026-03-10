# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class HaarTransform2D(nn.Module):
    """Fixed 2D Haar wavelet decomposition with stride=2.

    Input shape: [B, C, H, W] Output shape: [B, 4C, H/2, W/2]
    """

    def __init__(self):
        super().__init__()
        haar = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],  # LL
                [[1.0, -1.0], [1.0, -1.0]],  # LH
                [[1.0, 1.0], [-1.0, -1.0]],  # HL
                [[1.0, -1.0], [-1.0, 1.0]],  # HH
            ],
            dtype=torch.float32,
        )
        self.register_buffer("haar_weight", (haar / 2.0).unsqueeze(1), persistent=False)  # [4, 1, 2, 2]

    def forward(self, x):
        _b, c, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0, "HaarTransform2D requires even H and W"
        weight = self.haar_weight.repeat(c, 1, 1, 1)  # [4C, 1, 2, 2]
        return F.conv2d(x, weight, bias=None, stride=2, padding=0, groups=c)


class HWDDown(nn.Module):
    """Haar Wavelet Downsampling block to replace stride-2 Conv in YOLOv8 backbone."""

    def __init__(self, c1, c2):
        super().__init__()
        self.haar = HaarTransform2D()
        self.cv = Conv(4 * c1, c2, k=1, s=1)

    def forward(self, x):
        return self.cv(self.haar(x))
