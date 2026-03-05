# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""FcaNet-inspired frequency channel attention modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_freq_indices(method):
    """Return selected frequency indices according to the given method."""
    assert method in [
        "top1",
        "top2",
        "top4",
        "top8",
        "top16",
        "top32",
        "bot1",
        "bot2",
        "bot4",
        "bot8",
        "bot16",
        "bot32",
        "low1",
        "low2",
        "low4",
        "low8",
        "low16",
        "low32",
    ]
    num_freq = int(method[3:])
    if "top" in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif "low" in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif "bot" in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(nn.Module):
    """FcaNet multi-spectral channel attention."""

    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method="top16"):
        super().__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Scale input channels with frequency-aware attention weights."""
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """Generate fixed DCT filters for channel attention."""

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super().__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)
        self.register_buffer("weight", self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        """Apply precomputed DCT filters and sum over spatial dimensions."""
        assert len(x.shape) == 4, "x must been 4 dimensions, but got " + str(len(x.shape))

        x = x * self.weight
        return torch.sum(x, dim=[2, 3])

    def build_filter(self, pos, freq, POS):
        """Build one DCT basis scalar."""
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        """Build DCT filter tensor for all channels."""
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part : (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y
                    )

        return dct_filter


class FCALayer(nn.Module):
    """YOLO-friendly wrapper around FcaNet MultiSpectralAttentionLayer."""

    def __init__(self, c, dct_h=7, dct_w=7, reduction=16, freq_sel_method="top16"):
        super().__init__()
        self.att = MultiSpectralAttentionLayer(c, dct_h, dct_w, reduction=reduction, freq_sel_method=freq_sel_method)

    def forward(self, x):
        """Apply FCA channel reweighting."""
        return self.att(x)


class MultiSpectralAttentionLayerResidual(MultiSpectralAttentionLayer):
    """Residual FcaNet multi-spectral channel attention for weaker targets."""

    def __init__(self, channel, dct_h, dct_w, reduction=32, freq_sel_method="low16", alpha=0.5):
        super().__init__(channel, dct_h, dct_w, reduction=reduction, freq_sel_method=freq_sel_method)
        self.alpha = alpha

    def forward(self, x):
        """Apply residual channel scaling to preserve weak target distributions."""
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        y = self.dct_layer(x_pooled)
        y = self.fc(y).view(n, c, 1, 1)
        scale = 1.0 + self.alpha * (y - 1.0)
        return x * scale


class FCALayerResidual(nn.Module):
    """YOLO-friendly wrapper around residual FcaNet attention."""

    def __init__(self, c, dct_h=7, dct_w=7, reduction=32, freq_sel_method="low16", alpha=0.5):
        super().__init__()
        self.att = MultiSpectralAttentionLayerResidual(
            c, dct_h, dct_w, reduction=reduction, freq_sel_method=freq_sel_method, alpha=alpha
        )

    def forward(self, x):
        """Apply residual FCA channel reweighting."""
        return self.att(x)
