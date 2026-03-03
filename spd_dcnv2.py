# spd_dcnv2.py
# 修改理由（对应你的目标）：
# 1) 用 Space-to-Depth(SPD) 完成下采样，替代 stride=2/pooling，避免细裂缝小目标信息被池化/步长采样丢失（与SPDConv思想一致，但不照搬其“仅1x1”形式）。
# 2) 在下采样后接 DCNv2（offset + modulation mask），提升对细长、弯折、低对比裂缝的自适应对齐与背景抑制能力。
# 3) DCNv2 使用 stride=1：下采样只发生一次（SPD），进一步减少信息损失，利于实例分割边界与细裂缝连续性。

import torch
import torch.nn as nn

try:
    # 最推荐：mmcv 的 ModulatedDeformConv2d 就是 DCNv2 的常用实现
    from mmcv.ops import ModulatedDeformConv2d

    _HAS_MMCV = True
except Exception:
    _HAS_MMCV = False


class SpaceToDepth(nn.Module):
    """Space-to-Depth (pixel unshuffle variant): scale=2 => H,W /2 and C*4."""

    def __init__(self, scale: int = 2):
        super().__init__()
        assert scale >= 2 and isinstance(scale, int)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        s = self.scale
        assert h % s == 0 and w % s == 0, f"Input (H,W)=({h},{w}) must be divisible by scale={s}"
        # 等价于论文里四路采样并在通道维 concat（scale=2 时是4路）
        # 更一般化：用 view/permute 实现任意scale
        x = x.view(b, c, h // s, s, w // s, s)  # b,c,h/s,s,w/s,s
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()  # b,c,s,s,h/s,w/s
        x = x.view(b, c * s * s, h // s, w // s)  # b,c*s^2,h/s,w/s
        return x


class ConvBNAct(nn.Module):
    """Conv + BN + SiLU (与YOLOv8风格一致)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DCNv2BNAct(nn.Module):
    """DCNv2 block: Modulated Deformable Conv + BN + SiLU 说明： - offset+mask 通过一个普通卷积从输入特征预测（标准DCNv2做法） - deform conv
    stride=1（下采样由SPD完成）.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, groups=1, deform_groups=1, act=True):
        super().__init__()
        if not _HAS_MMCV:
            raise ImportError(
                "mmcv is required for ModulatedDeformConv2d (DCNv2). Install mmcv-full matching your CUDA/PyTorch."
            )
        if p is None:
            p = k // 2
        self.k = k
        self.deform_groups = deform_groups

        # offset(2*k*k*dg) + mask(k*k*dg)
        out_off = deform_groups * (2 * k * k + k * k)
        self.conv_offset_mask = nn.Conv2d(c1, out_off, kernel_size=3, stride=1, padding=1)

        self.dcn = ModulatedDeformConv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=1,
            groups=groups,
            deform_groups=deform_groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        out = self.conv_offset_mask(x)
        dg = self.deform_groups
        k2 = self.k * self.k

        o1 = 2 * k2 * dg
        offset = out[:, :o1, :, :]
        mask = out[:, o1 : o1 + k2 * dg, :, :].sigmoid()

        x = self.dcn(x, offset, mask)
        x = self.act(self.bn(x))
        return x


class SPD_DCNv2Down(nn.Module):
    """无池化下采样：SPD(scale=2) -> 1x1瓶颈(压通道) -> DCNv2(3x3, s=1) -> (可选1x1融合) - 输入: (B, Cin, H, W) - 输出: (B, Cout, H/2, W/2).
    """

    def __init__(self, c1, c2, scale=2, bottleneck_ratio=0.5, deform_groups=1, fuse_1x1=False):
        super().__init__()
        self.spd = SpaceToDepth(scale=scale)
        c_spd = c1 * (scale * scale)

        # 瓶颈：控制计算量，避免SPD后通道爆炸导致DCNv2太重
        c_mid = max(16, int(c2 * bottleneck_ratio))
        self.reduce = ConvBNAct(c_spd, c_mid, k=1, s=1)

        # 主特征提取：DCNv2
        self.dcn = DCNv2BNAct(c_mid, c2, k=3, s=1, deform_groups=deform_groups)

        # 可选：再加一个1x1融合，提升mask边界细节（分割任务有时有益）
        self.fuse = ConvBNAct(c2, c2, k=1, s=1) if fuse_1x1 else nn.Identity()

    def forward(self, x):
        x = self.spd(x)
        x = self.reduce(x)
        x = self.dcn(x)
        x = self.fuse(x)
        return x
