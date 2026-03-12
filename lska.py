# ultralytics/nn/modules/lska.py
# 修改理由：提取 LSKA 论文核心结构，用于在 YOLOv8 的 SPPF 后进行全局方向性上下文增强。
# 修改理由：同时提供原始版、残差安全版、裂缝方向增强版，便于通过不同 YAML 做消融实验。

import torch
import torch.nn as nn

try:
    pass
except Exception:
    # 兼容某些 ultralytics 版本路径差异
    pass


class LSKA(nn.Module):
    """原始 LSKA: 1 x (2d-1) DW-Conv (2d-1) x 1 DW-Conv 1 x floor(k/d) DW-D-Conv floor(k/d) x 1 DW-D-Conv 1x1 Conv
    最后与输入逐元素相乘.
    """

    def __init__(self, c, k=23, d=3):
        super().__init__()
        assert c > 0
        assert k >= d and d >= 1
        k_d = max(1, k // d)
        s = 2 * d - 1

        self.dw1 = nn.Conv2d(c, c, kernel_size=(1, s), stride=1, padding=(0, s // 2), groups=c, bias=False)
        self.dw2 = nn.Conv2d(c, c, kernel_size=(s, 1), stride=1, padding=(s // 2, 0), groups=c, bias=False)

        self.dw_d1 = nn.Conv2d(
            c,
            c,
            kernel_size=(1, k_d),
            stride=1,
            padding=(0, ((k_d - 1) * d) // 2),
            dilation=(1, d),
            groups=c,
            bias=False,
        )
        self.dw_d2 = nn.Conv2d(
            c,
            c,
            kernel_size=(k_d, 1),
            stride=1,
            padding=(((k_d - 1) * d) // 2, 0),
            dilation=(d, 1),
            groups=c,
            bias=False,
        )

        self.pw = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        attn = self.dw1(x)
        attn = self.dw2(attn)
        attn = self.dw_d1(attn)
        attn = self.dw_d2(attn)
        attn = self.pw(attn)
        return x * attn


class LSKARes(nn.Module):
    """残差安全版 LSKA： out = x * (1 + alpha * attn) 修改理由：避免直接 x * attn 对弱裂缝通道造成过度抑制。.
    """

    def __init__(self, c, k=23, d=3, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.attn = LSKA(c, k=k, d=d)

    def forward(self, x):
        # 先得到原始乘法结果 y = x * A
        y = self.attn(x)
        # 反推出注意力 A 的相对缩放趋势，用残差方式减弱抑制
        # 这里用 (y - x) 作为增量项更稳
        return x + self.alpha * (y - x)


class DirectionalLSKA(nn.Module):
    """裂缝导向方向版 LSKA： 将横向与纵向响应显式分开建模，再用轻量门控融合。 修改理由：裂缝是细长、方向性强的结构，显式方向建模比单一路径更贴合任务。.
    """

    def __init__(self, c, k=23, d=3, alpha=0.5):
        super().__init__()
        assert c > 0
        k_d = max(1, k // d)
        s = 2 * d - 1
        self.alpha = alpha

        # 横向分支
        self.h1 = nn.Conv2d(c, c, kernel_size=(1, s), stride=1, padding=(0, s // 2), groups=c, bias=False)
        self.h2 = nn.Conv2d(
            c,
            c,
            kernel_size=(1, k_d),
            stride=1,
            padding=(0, ((k_d - 1) * d) // 2),
            dilation=(1, d),
            groups=c,
            bias=False,
        )

        # 纵向分支
        self.v1 = nn.Conv2d(c, c, kernel_size=(s, 1), stride=1, padding=(s // 2, 0), groups=c, bias=False)
        self.v2 = nn.Conv2d(
            c,
            c,
            kernel_size=(k_d, 1),
            stride=1,
            padding=(((k_d - 1) * d) // 2, 0),
            dilation=(d, 1),
            groups=c,
            bias=False,
        )

        # 方向融合门控
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * 2, c // 4 if c >= 4 else 1, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(c // 4 if c >= 4 else 1, 2, 1, bias=True),
            nn.Softmax(dim=1),
        )

        self.fuse = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        h = self.h2(self.h1(x))
        v = self.v2(self.v1(x))

        cat = torch.cat([h, v], dim=1)
        w = self.gate(cat)  # [B,2,1,1]
        wh = w[:, 0:1]
        wv = w[:, 1:2]

        attn = wh * h + wv * v
        attn = self.fuse(attn)

        y = x * attn
        return x + self.alpha * (y - x)
