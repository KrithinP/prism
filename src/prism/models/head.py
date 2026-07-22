
"""Decoupled anchor-free detection head. See textbook Ch.4.

At every cell of every scale, three separate branches predict:
  class      (n_classes numbers)   -- what is it
  objectness (1 number)            -- is anything here at all
  box        (4 numbers)           -- center offset (x,y) and size (w,h)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from prism.models.blocks import ConvBNAct


class DecoupledHead(nn.Module):
    def __init__(self, num_classes=1, in_channels=(128, 256, 512), width=128):
        super().__init__()
        self.num_classes = num_classes
        self.n_scales = len(in_channels)

        # per scale: a shared stem, then a class branch and a box/obj branch
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()   # -> class scores
        self.reg_preds = nn.ModuleList()   # -> 4 box numbers
        self.obj_preds = nn.ModuleList()   # -> 1 objectness number

        for ch in in_channels:
            self.stems.append(ConvBNAct(ch, width, k=1))       # bring every scale to same width
            self.cls_convs.append(nn.Sequential(
                ConvBNAct(width, width, k=3), ConvBNAct(width, width, k=3)))
            self.reg_convs.append(nn.Sequential(
                ConvBNAct(width, width, k=3), ConvBNAct(width, width, k=3)))
            self.cls_preds.append(nn.Conv2d(width, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(width, 4, 1))
            self.obj_preds.append(nn.Conv2d(width, 1, 1))

        # Prior-probability bias init (RetinaNet/YOLOX). At birth every cell should
        # predict "almost certainly background" (p ~= 0.01), or the from-scratch
        # objectness loss starts ~50x too large and SGD diverges to NaN. See Ch.7.
        prior = 1e-2
        bias = -math.log((1 - prior) / prior)   # ~= -4.595
        for m in self.cls_preds:
            nn.init.constant_(m.bias, bias)
        for m in self.obj_preds:
            nn.init.constant_(m.bias, bias)

    def forward(self, feats):
        """feats: (p3, p4, p5) from the neck.
        Returns a list of per-scale outputs, each (B, 4+1+num_classes, H, W)."""
        outputs = []
        for i, x in enumerate(feats):
            x = self.stems[i](x)
            cls_feat = self.cls_convs[i](x)
            reg_feat = self.reg_convs[i](x)
            cls = self.cls_preds[i](cls_feat)     # (B, num_classes, H, W)
            reg = self.reg_preds[i](reg_feat)     # (B, 4, H, W)
            obj = self.obj_preds[i](reg_feat)     # (B, 1, H, W)
            # order: box(4), objectness(1), class(num_classes)
            outputs.append(torch.cat([reg, obj, cls], dim=1))
        return outputs

