"""The four building blocks of CSPDarknet. See textbook Ch.4."""

from __future__ import annotations

import torch
from torch import nn


class ConvBNAct(nn.Module):
    """Convolution -> BatchNorm -> SiLU activation. The atom of the whole network."""

    def __init__(self, in_ch, out_ch, k=1, stride=1, groups=1):
        super().__init__()
        pad = (k - 1) // 2  # 'same' padding for odd kernels
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, pad, groups=groups, bias=False)
        # Match the official YOLOX Exp initialization. These values are part of
        # the detector base, so every fusion arm must inherit the same setting.
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Squeeze channels, expand back. Optional skip connection."""

    def __init__(self, in_ch, out_ch, shortcut=True, expansion=0.5):
        super().__init__()
        hidden = int(out_ch * expansion)
        self.conv1 = ConvBNAct(in_ch, hidden, k=1)
        self.conv2 = ConvBNAct(hidden, out_ch, k=3)
        self.use_add = shortcut and in_ch == out_ch

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.use_add else y


class CSPLayer(nn.Module):
    """Cross-Stage Partial: split channels, send HALF through the bottlenecks,
    concat back. Most of the benefit of a deep stack at ~half the compute."""

    def __init__(self, in_ch, out_ch, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        hidden = int(out_ch * expansion)
        self.conv1 = ConvBNAct(in_ch, hidden, k=1)  # branch that gets processed
        self.conv2 = ConvBNAct(in_ch, hidden, k=1)  # branch that skips ahead
        self.conv3 = ConvBNAct(2 * hidden, out_ch, k=1)  # merge the two branches
        self.blocks = nn.Sequential(
            *[Bottleneck(hidden, hidden, shortcut, expansion=1.0) for _ in range(n)]
        )

    def forward(self, x):
        a = self.blocks(self.conv1(x))
        b = self.conv2(x)
        return self.conv3(torch.cat((a, b), dim=1))


class SPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling: pool at 5/9/13 and concat, so one layer sees
    several receptive-field sizes at once. Used at the end of dark5."""

    def __init__(self, in_ch, out_ch, kernels=(5, 9, 13)):
        super().__init__()
        hidden = in_ch // 2
        self.conv1 = ConvBNAct(in_ch, hidden, k=1)
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernels]
        )
        self.conv2 = ConvBNAct(hidden * (len(kernels) + 1), out_ch, k=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [p(x) for p in self.pools], dim=1)
        return self.conv2(x)
