"""PAFPN neck: lets dark3/dark4/dark5 share information across scales. See Ch.4/5.

Two passes:
  top-down  -- meaning flows DOWN from dark5 to dark3
  bottom-up -- location flows UP from dark3 to dark5
Outputs three maps at the SAME shapes as the inputs, now each carrying both.
"""

from __future__ import annotations

import torch
from torch import nn

from prism.models.blocks import ConvBNAct, CSPLayer


class PAFPN(nn.Module):
    def __init__(self, base_ch=32, base_depth=1):
        super().__init__()
        c3, c4, c5 = base_ch * 4, base_ch * 8, base_ch * 16  # 128, 256, 512

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")  # double H,W

        # --- top-down: meaning flows down (dark5 -> dark4 -> dark3) ---
        self.reduce_c5 = ConvBNAct(c5, c4, k=1)  # 512 -> 256 before merging into dark4
        self.csp_td4 = CSPLayer(2 * c4, c4, n=base_depth, shortcut=False)

        self.reduce_c4 = ConvBNAct(c4, c3, k=1)  # 256 -> 128 before merging into dark3
        self.csp_td3 = CSPLayer(2 * c3, c3, n=base_depth, shortcut=False)  # -> OUT dark3

        # --- bottom-up: location flows up (dark3 -> dark4 -> dark5) ---
        self.down_p3 = ConvBNAct(c3, c3, k=3, stride=2)  # shrink dark3 to dark4 size
        self.csp_bu4 = CSPLayer(2 * c3, c4, n=base_depth, shortcut=False)  # -> OUT dark4

        self.down_p4 = ConvBNAct(c4, c4, k=3, stride=2)  # shrink to dark5 size
        self.csp_bu5 = CSPLayer(2 * c4, c5, n=base_depth, shortcut=False)  # -> OUT dark5

    def forward(self, feats):
        c3, c4, c5 = feats  # from the backbone

        # top-down
        p5 = self.reduce_c5(c5)  # 512 -> 256
        p4 = self.csp_td4(torch.cat([self.upsample(p5), c4], dim=1))
        p4_red = self.reduce_c4(p4)  # 256 -> 128
        p3 = self.csp_td3(torch.cat([self.upsample(p4_red), c3], dim=1))  # OUT dark3

        # bottom-up
        n4 = self.csp_bu4(torch.cat([self.down_p3(p3), p4_red], dim=1))  # OUT dark4
        n5 = self.csp_bu5(torch.cat([self.down_p4(n4), p5], dim=1))  # OUT dark5

        return p3, n4, n5
