"""Early fusion: the two cameras are concatenated at the INPUT, before the
first conv, and one shared backbone/neck/head runs on the 6-channel stack.
This is family B1 of the experiment matrix (docs/EXPERIMENT_PLAN.md) -- the
"meet at depth 0" end of the fusion-depth ladder, and the cheapest graph there
is: a single stream whose capacity is exactly one YOLOX plus the extra stem
input channels.

Graph:

    Z_0 = F_0(x_r, x_t)          # input concat, before any conv
    Z_3, Z_4, Z_5 = B_shared(Z_0) # ONE backbone, never split
    y = H(N(Z_3, Z_4, Z_5))

Unlike the middle-fusion graphs there is no anchor branch and no adapter: the
streams are indistinguishable after the input, so there are no per-modality
parameters anywhere. (TODO.md 6.3 / blocker #10.)
"""

from __future__ import annotations

import torch
from torch import nn

from prism.models.darknet import CSPDarknet
from prism.models.head import DecoupledHead
from prism.models.pafpn import PAFPN

_NECK_LEVELS = (3, 4, 5)


class EarlyFusionYOLOX(nn.Module):
    def __init__(self, num_classes=1, base_ch=32, base_depth=1):
        super().__init__()
        self.level_channels = {3: base_ch * 4, 4: base_ch * 8, 5: base_ch * 16}
        self.backbone = CSPDarknet(base_ch=base_ch, base_depth=base_depth, in_ch=6)
        self.neck = PAFPN(base_ch=base_ch, base_depth=base_depth)
        self.head = DecoupledHead(
            num_classes=num_classes,
            in_channels=tuple(self.level_channels[level] for level in _NECK_LEVELS),
        )

    def fusion_levels(self):
        return (0,)  # the fusion happens at the input, before any conv

    def forward_features(self, rgb, thermal):
        x = torch.cat([rgb, thermal], dim=1)  # F_0: the only fusion
        return self.backbone(x)

    def forward(self, rgb, thermal):
        return self.head(self.neck(self.forward_features(rgb, thermal)))

    def graph_table(self) -> str:
        rows = ["  input: C=6 (rgb 3 + thermal 3 concatenated), stride=1, owned by [fused]"]
        for lvl in _NECK_LEVELS:
            ch = self.level_channels[lvl]
            stride = 2**lvl
            rows.append(f"  p{lvl}: C={ch}, stride={stride}, owned by [fused]")
        return "\n".join(rows)
