"""Single-stream YOLOX: backbone -> neck -> head, wired into one module.
This is the RGB-only / thermal-only baseline model. See textbook Ch.4."""
from __future__ import annotations

import torch.nn as nn

from prism.models.darknet import CSPDarknet
from prism.models.pafpn import PAFPN
from prism.models.head import DecoupledHead


class YOLOX(nn.Module):
    def __init__(self, num_classes=1, base_ch=32, base_depth=1):
        super().__init__()
        self.backbone = CSPDarknet(base_ch=base_ch, base_depth=base_depth)
        self.neck = PAFPN(base_ch=base_ch, base_depth=base_depth)
        self.head = DecoupledHead(
            num_classes=num_classes,
            in_channels=(base_ch * 4, base_ch * 8, base_ch * 16),
        )

    def forward(self, x):
        return self.head(self.neck(self.backbone(x)))
